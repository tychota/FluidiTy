#include "fluidity/host/adaptive_liquid_remap_3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "mac_pressure_3d_arithmetic.h"

namespace fluidity {
namespace {

using pressure_internal::arithmetic::CompensatedSum;
using FaceKey = std::tuple<FaceNormalAxis, std::int64_t, std::int64_t, std::int64_t, std::uint8_t>;

// Topology family ordinals are x-fast and the remap kernel is z-fast.
constexpr std::array<std::size_t, 8> kRemapIndexForChild{0, 4, 2, 6, 1, 5, 3, 7};

absl::Status ValidateFields(const AdaptiveMacTopology3D& topology, const AdaptiveLiquidRemapInput3D& fields) {
  if (fields.inventories.size() != topology.cells().size()
      || fields.interior_fluxes_m3_per_s.size() != topology.interior_faces().size()) {
    return absl::InvalidArgumentError("adaptive remap fields must cover the committed topology");
  }
  for (const auto& row : fields.inventories) {
    const Vec3 momentum = row.momentum_kg_m_per_s;
    const std::array amounts{
        row.mass_kg,
        row.reference_volume_m3,
        row.current_volume_m3,
        momentum.x,
        momentum.y,
        momentum.z,
        row.enthalpy_j,
    };
    if (row.mass_kg < 0 || row.reference_volume_m3 < 0 || row.current_volume_m3 < 0
        || !std::ranges::all_of(
            amounts,
            [](double value) {
              return std::isfinite(value);
            }
        )) {
      return absl::InvalidArgumentError("adaptive remap inventories must be finite with nonnegative mass and volumes");
    }
  }
  if (!std::ranges::all_of(fields.interior_fluxes_m3_per_s, [](double value) {
        return std::isfinite(value);
      })) {
    return absl::InvalidArgumentError("adaptive remap face fluxes must be finite");
  }
  return absl::OkStatus();
}

FaceKey Key(const auto& face) {
  const auto index = face.face_index.lattice_index;
  return {face.face_index.axis, index.z, index.y, index.x, face.width_exponent};
}

template <typename Face>
std::optional<std::size_t> FindFace(std::span<const Face> faces, const FaceKey& key) {
  const auto face = std::ranges::lower_bound(faces, key, {}, Key<Face>);
  if (face == faces.end() || Key(*face) != key) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(face - faces.begin());
}

std::array<FaceKey, 4> SubfaceKeys(const auto& face) {
  const auto normal = face.face_index.axis;
  const auto axis = static_cast<std::size_t>(normal);
  const auto exponent = static_cast<std::uint8_t>(face.width_exponent - 1);
  const std::int64_t width = std::int64_t{1} << exponent;
  const auto index = face.face_index.lattice_index;
  std::array<FaceKey, 4> subfaces{};
  for (std::size_t slot = 0; slot < subfaces.size(); ++slot) {
    std::array<std::int64_t, 3> lower{index.x, index.y, index.z};
    lower[(axis + 1) % 3] += static_cast<std::int64_t>(slot & 1) * width;
    lower[(axis + 2) % 3] += static_cast<std::int64_t>(slot >> 1) * width;
    subfaces[slot] = {normal, lower[2], lower[1], lower[0], exponent};
  }
  return subfaces;
}

bool IsSiblingFace(const AdaptiveMacInteriorFace3D& face, std::span<const int> child_index) {
  return child_index[face.tail_cell_ordinal] >= 0 && child_index[face.head_cell_ordinal] >= 0;
}

// The coarse scalar must represent the entire partition. Accumulate the
// residual in nonoverlapping binary64 partials via ordered FastTwoSum so a
// small lost term remains visible even next to much larger terms.
bool RepresentsPartitionSum(const std::array<double, 4>& values, double coarse_value) {
  const std::array terms{values[0], values[1], values[2], values[3], -coarse_value};
  std::array<double, terms.size()> partials{};
  std::size_t count = 0;
  for (double term : terms) {
    std::size_t retained = 0;
    for (std::size_t index = 0; index < count; ++index) {
      double partial = partials[index];
      if (std::abs(term) < std::abs(partial)) {
        std::swap(term, partial);
      }
      const double sum = term + partial;
      if (!std::isfinite(sum)) {
        return false;
      }
      const double remainder = partial - (sum - term);
      if (remainder != 0) {
        partials[retained++] = remainder;
      }
      term = sum;
    }
    partials[retained] = term;
    count = retained + 1;
  }
  return std::ranges::all_of(std::span(partials).first(count), [](double value) {
    return value == 0;
  });
}

bool RepresentsSubfaceSum(const std::array<LiquidRemapSubface3D, 4>& subfaces, double coarse_flux) {
  return RepresentsPartitionSum(
      {subfaces[0].flux_m3_per_s, subfaces[1].flux_m3_per_s, subfaces[2].flux_m3_per_s, subfaces[3].flux_m3_per_s},
      coarse_flux
  );
}

// Overlay the two face partitions by integer lattice rectangles. In a single
// aligned 2:1 family edit, every coarse rectangle is unchanged or partitions
// into exactly four fine rectangles. Fine sibling faces have no coarse image.
absl::Status TransferOuterFluxes(
    std::span<const AdaptiveMacInteriorFace3D> coarse_faces,
    std::span<const AdaptiveMacInteriorFace3D> fine_faces,
    std::span<const int> child_index,
    std::span<const double> committed_fluxes,
    std::span<double> proposed_fluxes,
    bool refining
) {
  std::vector<bool> fine_used(fine_faces.size());
  for (std::size_t face = 0; face < fine_faces.size(); ++face) {
    fine_used[face] = IsSiblingFace(fine_faces[face], child_index);
  }
  for (std::size_t coarse = 0; coarse < coarse_faces.size(); ++coarse) {
    const auto& face = coarse_faces[coarse];
    if (const auto exact = FindFace(fine_faces, Key(face))) {
      if (fine_used[*exact]) {
        return absl::FailedPreconditionError("adaptive face coverage overlaps");
      }
      fine_used[*exact] = true;
      if (refining) {
        proposed_fluxes[*exact] = committed_fluxes[coarse];
      } else {
        proposed_fluxes[coarse] = committed_fluxes[*exact];
      }
      continue;
    }
    if (face.width_exponent == 0) {
      return absl::FailedPreconditionError("adaptive face coverage is incomplete");
    }

    std::array<LiquidRemapFaceSlot3D, 4> slots{};
    const auto keys = SubfaceKeys(face);
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      const auto fine = FindFace(fine_faces, keys[slot]);
      if (!fine || fine_used[*fine]) {
        return absl::FailedPreconditionError("adaptive face requires four distinct covered subfaces");
      }
      fine_used[*fine] = true;
      // Both topologies orient the face along its positive world axis.
      slots[slot] = {.owner_slot = *fine, .normal_sign = 1};
    }
    if (refining) {
      const auto split = ProlongLiquidFaceFlux3D(committed_fluxes[coarse], slots);
      if (!split.ok()) {
        return split.status();
      }
      for (const auto& subface : *split) {
        proposed_fluxes[subface.slot.owner_slot] = subface.flux_m3_per_s;
      }
    } else {
      std::array<LiquidRemapSubface3D, 4> subfaces{};
      for (std::size_t slot = 0; slot < slots.size(); ++slot) {
        subfaces[slot] = {.slot = slots[slot], .flux_m3_per_s = committed_fluxes[slots[slot].owner_slot]};
      }
      const auto sum = RestrictLiquidFaceFlux3D(subfaces);
      if (!sum.ok()) {
        return sum.status();
      }
      if (!RepresentsSubfaceSum(subfaces, *sum)) {
        return absl::OutOfRangeError("adaptive outer flux is not exactly representable on the coarse face");
      }
      proposed_fluxes[coarse] = *sum;
    }
  }
  if (std::ranges::find(fine_used, false) != fine_used.end()) {
    return absl::FailedPreconditionError("adaptive fine face has no coarse coverage");
  }
  return absl::OkStatus();
}

struct ExteriorFluxView {
  std::span<const AdaptiveMacExteriorFace3D> faces;
  std::span<const double> outward_fluxes;
};

absl::Status ValidateExteriorFields(
    const AdaptiveLiquidExteriorRemapInput3D& fields,
    std::span<const AdaptiveMacExteriorFace3D> committed_faces,
    std::span<const AdaptiveMacExteriorFace3D> candidate_faces
) {
  if (fields.committed_open_areas_m2.size() != committed_faces.size()
      || fields.committed_outward_fluxes_m3_per_s.size() != committed_faces.size()
      || fields.candidate_open_areas_m2.size() != candidate_faces.size()) {
    return absl::InvalidArgumentError("adaptive exterior fields must cover every canonical exterior face");
  }
  const auto valid_areas = [](std::span<const AdaptiveMacExteriorFace3D> faces, std::span<const double> areas) {
    for (std::size_t index = 0; index < faces.size(); ++index) {
      if (!std::isfinite(areas[index]) || areas[index] < 0 || areas[index] > faces[index].face_area) {
        return false;
      }
    }
    return true;
  };
  if (!valid_areas(committed_faces, fields.committed_open_areas_m2)
      || !valid_areas(candidate_faces, fields.candidate_open_areas_m2)) {
    return absl::InvalidArgumentError("adaptive exterior aperture must be within the geometric face area");
  }
  for (std::size_t index = 0; index < committed_faces.size(); ++index) {
    const double flux = fields.committed_outward_fluxes_m3_per_s[index];
    if (!std::isfinite(flux) || (fields.committed_open_areas_m2[index] == 0 && flux != 0)) {
      return absl::InvalidArgumentError("adaptive exterior flux must be finite and zero on a sealed face");
    }
  }
  return absl::OkStatus();
}

absl::Status SplitExteriorFlux(
    double coarse_flux,
    double coarse_area,
    const std::array<double, 4>& fine_areas,
    std::array<LiquidRemapSubface3D, 4>& subfaces
) {
  if (coarse_flux == 0) {
    return absl::OkStatus();
  }
  if (std::ranges::all_of(fine_areas, [quarter_area = coarse_area / 4](double area) {
        return area == quarter_area;
      })) {
    std::array<LiquidRemapFaceSlot3D, 4> slots{};
    for (std::size_t index = 0; index < slots.size(); ++index) {
      slots[index] = subfaces[index].slot;
    }
    const auto uniform_split = ProlongLiquidFaceFlux3D(coarse_flux, slots);
    if (!uniform_split.ok()) {
      return uniform_split.status();
    }
    subfaces = *uniform_split;
    return absl::OkStatus();
  }

  // A direct proportional split can retain shares smaller than the coarse
  // flux quantum. Use it only when the exact signed sum is representable.
  CompensatedSum proportional_sum;
  for (std::size_t index = 0; index < fine_areas.size(); ++index) {
    const double aperture_fraction = fine_areas[index] / coarse_area;
    const double proportional_flux = coarse_flux * aperture_fraction;
    subfaces[index].flux_m3_per_s = proportional_flux;
    proportional_sum.Add(proportional_flux);
  }
  if (proportional_sum.Total() == coarse_flux && RepresentsSubfaceSum(subfaces, coarse_flux)) {
    return absl::OkStatus();
  }

  // Round cumulative aperture shares in units of the coarse flux's
  // binary64 quantum. Their consecutive differences sum exactly and
  // assign no flux to a zero-aperture subface.
  int flux_exponent = 0;
  std::frexp(coarse_flux, &flux_exponent);
  constexpr int kDigits = std::numeric_limits<double>::digits;
  constexpr int kDenormalExponent = std::numeric_limits<double>::min_exponent - kDigits;
  const int quantum_exponent = std::max(flux_exponent - kDigits, kDenormalExponent);
  const double flux_quanta = std::scalbn(std::abs(coarse_flux), -quantum_exponent);
  CompensatedSum cumulative_area;
  double assigned_quanta = 0;
  for (std::size_t index = 0; index < fine_areas.size(); ++index) {
    cumulative_area.Add(fine_areas[index]);
    const double cumulative_quanta = std::round((cumulative_area.Total() / coarse_area) * flux_quanta);
    const double subface_flux = std::scalbn(cumulative_quanta - assigned_quanta, quantum_exponent);
    subfaces[index].flux_m3_per_s = std::copysign(subface_flux, coarse_flux);
    assigned_quanta = cumulative_quanta;
  }
  if (!RepresentsSubfaceSum(subfaces, coarse_flux)) {
    return absl::OutOfRangeError("adaptive exterior flux split cannot conserve the coarse face");
  }
  return absl::OkStatus();
}

// Match exterior rectangles independently of the interior partition. Both
// sides of a match must describe the same physical owner side and aperture.
absl::StatusOr<std::vector<double>> TransferExteriorFluxes(
    const AdaptiveLiquidExteriorRemapInput3D& fields,
    std::span<const AdaptiveMacExteriorFace3D> coarse_faces,
    std::span<const AdaptiveMacExteriorFace3D> fine_faces,
    const AdaptiveFamilyChange3D& family,
    std::size_t coarse_cell_count,
    std::span<const int> child_index,
    bool refining
) {
  const auto committed_faces = refining ? coarse_faces : fine_faces;
  const auto candidate_faces = refining ? fine_faces : coarse_faces;
  const auto validation = ValidateExteriorFields(fields, committed_faces, candidate_faces);
  if (!validation.ok()) {
    return validation;
  }
  const auto coarse_areas = refining ? fields.committed_open_areas_m2 : fields.candidate_open_areas_m2;
  const auto fine_areas = refining ? fields.candidate_open_areas_m2 : fields.committed_open_areas_m2;
  const auto committed_fluxes = fields.committed_outward_fluxes_m3_per_s;
  std::vector<double> proposed_fluxes(candidate_faces.size());
  std::vector<std::optional<std::size_t>> fine_owner_for_coarse(coarse_cell_count);
  for (const auto retained : family.retained_leaves) {
    const auto coarse_owner = refining ? retained.old_cell_ordinal : retained.new_cell_ordinal;
    fine_owner_for_coarse[coarse_owner] = refining ? retained.new_cell_ordinal : retained.old_cell_ordinal;
  }

  std::vector<bool> fine_used(fine_faces.size());
  for (std::size_t coarse = 0; coarse < coarse_faces.size(); ++coarse) {
    const auto& coarse_face = coarse_faces[coarse];
    const auto same_owner_side = [&](std::size_t fine) {
      const auto& fine_face = fine_faces[fine];
      const bool owner_matches = coarse_face.cell_ordinal == family.parent_ordinal
          ? child_index[fine_face.cell_ordinal] >= 0
          : fine_owner_for_coarse[coarse_face.cell_ordinal] == fine_face.cell_ordinal;
      return owner_matches && coarse_face.cell_is_tail == fine_face.cell_is_tail;
    };

    if (const auto exact = FindFace(fine_faces, Key(coarse_face))) {
      if (fine_used[*exact] || !same_owner_side(*exact) || coarse_face.face_area != fine_faces[*exact].face_area
          || coarse_areas[coarse] != fine_areas[*exact]) {
        return absl::FailedPreconditionError("adaptive exterior face has incompatible ownership or aperture");
      }
      fine_used[*exact] = true;
      proposed_fluxes[refining ? *exact : coarse] = committed_fluxes[refining ? coarse : *exact];
      continue;
    }
    if (coarse_face.width_exponent == 0) {
      return absl::FailedPreconditionError("adaptive exterior face coverage is incomplete");
    }

    std::array<LiquidRemapSubface3D, 4> subfaces{};
    std::array<double, 4> subface_areas{};
    const auto keys = SubfaceKeys(coarse_face);
    for (std::size_t slot = 0; slot < keys.size(); ++slot) {
      const auto fine = FindFace(fine_faces, keys[slot]);
      if (!fine || fine_used[*fine] || !same_owner_side(*fine)
          || fine_faces[*fine].face_area * 4 != coarse_face.face_area) {
        return absl::FailedPreconditionError("adaptive exterior face requires four distinct owned subfaces");
      }
      fine_used[*fine] = true;
      // Matching outward normals keep the same sign even on the low side.
      subfaces[slot].slot = {.owner_slot = *fine, .normal_sign = 1};
    }
    std::ranges::sort(subfaces, {}, [](const auto& subface) {
      return subface.slot.owner_slot;
    });
    for (std::size_t slot = 0; slot < subfaces.size(); ++slot) {
      subface_areas[slot] = fine_areas[subfaces[slot].slot.owner_slot];
    }
    if (!RepresentsPartitionSum(subface_areas, coarse_areas[coarse])) {
      return absl::FailedPreconditionError("adaptive exterior subfaces do not conserve the open area");
    }
    if (refining) {
      const auto split = SplitExteriorFlux(committed_fluxes[coarse], coarse_areas[coarse], subface_areas, subfaces);
      if (!split.ok()) {
        return split;
      }
      for (const auto& subface : subfaces) {
        proposed_fluxes[subface.slot.owner_slot] = subface.flux_m3_per_s;
      }
    } else {
      for (auto& subface : subfaces) {
        subface.flux_m3_per_s = committed_fluxes[subface.slot.owner_slot];
      }
      const auto sum = RestrictLiquidFaceFlux3D(subfaces);
      if (!sum.ok()) {
        return sum.status();
      }
      if (!RepresentsSubfaceSum(subfaces, *sum)) {
        return absl::OutOfRangeError("adaptive exterior flux is not exactly representable on the coarse face");
      }
      proposed_fluxes[coarse] = *sum;
    }
  }
  if (std::ranges::find(fine_used, false) != fine_used.end()) {
    return absl::FailedPreconditionError("adaptive exterior fine face has no coarse coverage");
  }
  return proposed_fluxes;
}

struct ChildOutflow {
  CompensatedSum sum;
  double largest_term = 0;

  void Add(double signed_flux) {
    sum.Add(signed_flux);
    largest_term = std::max(largest_term, std::abs(signed_flux));
  }
};

std::array<ChildOutflow, 8> ChildOutflows(
    std::span<const AdaptiveMacInteriorFace3D> fine_faces,
    std::span<const double> fine_fluxes,
    std::span<const int> child_index,
    ExteriorFluxView fine_exterior
) {
  std::array<ChildOutflow, 8> outflows{};
  for (std::size_t face = 0; face < fine_faces.size(); ++face) {
    const int tail = child_index[fine_faces[face].tail_cell_ordinal];
    const int head = child_index[fine_faces[face].head_cell_ordinal];
    if (tail >= 0) {
      outflows[tail].Add(fine_fluxes[face]);
    }
    if (head >= 0) {
      outflows[head].Add(-fine_fluxes[face]);
    }
  }
  for (std::size_t face = 0; face < fine_exterior.faces.size(); ++face) {
    const int child = child_index[fine_exterior.faces[face].cell_ordinal];
    if (child >= 0) {
      outflows[child].Add(fine_exterior.outward_fluxes[face]);
    }
  }
  return outflows;
}

absl::Status RedistributeSiblingFluxes(
    const AdaptiveMacTopology3D& coarse,
    const AdaptiveFamilyChange3D& family,
    std::span<const double> coarse_fluxes,
    std::span<const int> child_index,
    std::span<double> fine_fluxes,
    ExteriorFluxView coarse_exterior,
    ExteriorFluxView fine_exterior
) {
  CompensatedSum parent_outflow;
  const auto coarse_faces = coarse.interior_faces();
  for (std::size_t face = 0; face < coarse_faces.size(); ++face) {
    if (coarse_faces[face].tail_cell_ordinal == family.parent_ordinal) {
      parent_outflow.Add(coarse_fluxes[face]);
    }
    if (coarse_faces[face].head_cell_ordinal == family.parent_ordinal) {
      parent_outflow.Add(-coarse_fluxes[face]);
    }
  }
  for (std::size_t face = 0; face < coarse_exterior.faces.size(); ++face) {
    if (coarse_exterior.faces[face].cell_ordinal == family.parent_ordinal) {
      parent_outflow.Add(coarse_exterior.outward_fluxes[face]);
    }
  }
  const double parent_flux = parent_outflow.Total();
  const double child_target = parent_flux / 8;
  const double parent_volume = coarse.cells()[family.parent_ordinal].cell_volume;
  const auto fine_cells = family.topology.cells();
  if (!std::isfinite(parent_flux / parent_volume) || child_target * 8 != parent_flux
      || fine_cells[family.child_ordinals.front()].cell_volume * 8 != parent_volume) {
    return absl::OutOfRangeError("adaptive parent divergence cannot be represented on the children");
  }

  const auto fine_faces = family.topology.interior_faces();
  std::array<std::optional<std::size_t>, 8> tree_face{};
  for (std::size_t face = 0; face < fine_faces.size(); ++face) {
    const int tail = child_index[fine_faces[face].tail_cell_ordinal];
    const int head = child_index[fine_faces[face].head_cell_ordinal];
    if (head > 0 && tail == (head & (head - 1))) {
      tree_face[head] = face;
    }
  }

  // Each non-root octant connects to the octant with its lowest set bit
  // cleared. Descending ordinals drain each subtree into its tree edge.
  // A positive outgoing subtree contribution is negative along that edge.
  const auto outer = ChildOutflows(fine_faces, fine_fluxes, child_index, fine_exterior);
  std::array<CompensatedSum, 8> subtree{};
  for (std::size_t child = 0; child < subtree.size(); ++child) {
    subtree[child].Add(child_target);
    subtree[child].Add(-outer[child].sum.Total());
  }
  for (std::size_t child = subtree.size() - 1; child > 0; --child) {
    if (!tree_face[child]) {
      return absl::FailedPreconditionError("adaptive sibling flux tree is incomplete");
    }
    const double required = subtree[child].Total();
    if (!std::isfinite(required)) {
      return absl::OutOfRangeError("adaptive sibling flux is unrepresentable");
    }
    fine_fluxes[*tree_face[child]] = -required;
    subtree[child & (child - 1)].Add(required);
  }

  constexpr double kRoundoff = 64 * std::numeric_limits<double>::epsilon();
  for (const auto& child : ChildOutflows(fine_faces, fine_fluxes, child_index, fine_exterior)) {
    const double outflow = child.sum.Total();
    const double scale = std::max(child.largest_term, std::abs(child_target));
    const bool subnormal_target = child_target != 0 && std::abs(child_target) < std::numeric_limits<double>::min();
    if (!std::isfinite(outflow) || (subnormal_target && outflow != child_target)
        || std::abs(outflow - child_target) > kRoundoff * scale) {
      return absl::OutOfRangeError("adaptive sibling fluxes cannot preserve the parent divergence");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<AdaptiveLiquidRemapProposal3D> Prepare(
    const AdaptiveMacTopology3D& committed_topology,
    const AdaptiveLiquidRemapInput3D& committed,
    AdaptiveMacCellKey3D parent,
    const LiquidRemapStencil3D* parent_stencil
) {
  const auto validation = ValidateFields(committed_topology, committed);
  if (!validation.ok()) {
    return validation;
  }
  const bool refining = parent_stencil != nullptr;
  auto family = refining ? PrepareAdaptiveRefinement3D(committed_topology, committed.previous_pressures_pa, parent)
                         : PrepareAdaptiveCoarsening3D(committed_topology, committed.previous_pressures_pa, parent);
  if (!family.ok()) {
    return family.status();
  }

  std::vector<LiquidRemapInventory3D> inventories(family->topology.cells().size());
  for (const auto retained : family->retained_leaves) {
    inventories[retained.new_cell_ordinal] = committed.inventories[retained.old_cell_ordinal];
  }
  if (refining) {
    const auto children = ProlongLiquidInventory3D(committed.inventories[family->parent_ordinal], *parent_stencil);
    if (!children.ok()) {
      return children.status();
    }
    for (std::size_t child = 0; child < family->child_ordinals.size(); ++child) {
      inventories[family->child_ordinals[child]] = (*children)[kRemapIndexForChild[child]];
    }
  } else {
    LiquidRemapChildren3D children{};
    for (std::size_t child = 0; child < family->child_ordinals.size(); ++child) {
      children[kRemapIndexForChild[child]] = committed.inventories[family->child_ordinals[child]];
    }
    const auto restricted = RestrictLiquidInventory3D(children);
    if (!restricted.ok()) {
      return restricted.status();
    }
    inventories[family->parent_ordinal] = *restricted;
  }

  const auto& coarse = refining ? committed_topology : family->topology;
  const auto& fine = refining ? family->topology : committed_topology;
  std::vector<int> child_index(fine.cells().size(), -1);
  for (std::size_t child = 0; child < family->child_ordinals.size(); ++child) {
    child_index[family->child_ordinals[child]] = static_cast<int>(child);
  }
  std::vector<double> fluxes(family->topology.interior_faces().size());
  const auto transfer = TransferOuterFluxes(
      coarse.interior_faces(),
      fine.interior_faces(),
      child_index,
      committed.interior_fluxes_m3_per_s,
      fluxes,
      refining
  );
  if (!transfer.ok()) {
    return transfer;
  }
  AdaptiveMacFaceOwnership3D coarse_ownership;
  AdaptiveMacFaceOwnership3D fine_ownership;
  std::vector<double> exterior_outward_fluxes;
  if (committed.exterior) {
    auto built_coarse_ownership = BuildAdaptiveMacFaceOwnership3D(coarse);
    if (!built_coarse_ownership.ok()) {
      return built_coarse_ownership.status();
    }
    coarse_ownership = std::move(*built_coarse_ownership);
    auto built_fine_ownership = BuildAdaptiveMacFaceOwnership3D(fine);
    if (!built_fine_ownership.ok()) {
      return built_fine_ownership.status();
    }
    fine_ownership = std::move(*built_fine_ownership);

    auto exterior_transfer = TransferExteriorFluxes(
        *committed.exterior,
        coarse_ownership.exterior_faces,
        fine_ownership.exterior_faces,
        *family,
        coarse.cells().size(),
        child_index,
        refining
    );
    if (!exterior_transfer.ok()) {
      return exterior_transfer.status();
    }
    exterior_outward_fluxes = std::move(*exterior_transfer);
  }
  if (refining) {
    const auto committed_outward =
        committed.exterior ? committed.exterior->committed_outward_fluxes_m3_per_s : std::span<const double>{};
    const auto redistribution = RedistributeSiblingFluxes(
        coarse,
        *family,
        committed.interior_fluxes_m3_per_s,
        child_index,
        fluxes,
        {.faces = coarse_ownership.exterior_faces, .outward_fluxes = committed_outward},
        {.faces = fine_ownership.exterior_faces, .outward_fluxes = exterior_outward_fluxes}
    );
    if (!redistribution.ok()) {
      return redistribution;
    }
  }
  return AdaptiveLiquidRemapProposal3D{
      .family = std::move(*family),
      .inventories = std::move(inventories),
      .interior_fluxes_m3_per_s = std::move(fluxes),
      .exterior_outward_fluxes_m3_per_s = std::move(exterior_outward_fluxes),
  };
}

}  // namespace

absl::StatusOr<AdaptiveLiquidRemapProposal3D> PrepareAdaptiveLiquidRefinement3D(
    const AdaptiveMacTopology3D& committed_topology,
    const AdaptiveLiquidRemapInput3D& committed,
    AdaptiveMacCellKey3D parent,
    const LiquidRemapStencil3D& parent_stencil
) {
  return Prepare(committed_topology, committed, parent, &parent_stencil);
}

absl::StatusOr<AdaptiveLiquidRemapProposal3D> PrepareAdaptiveLiquidCoarsening3D(
    const AdaptiveMacTopology3D& committed_topology,
    const AdaptiveLiquidRemapInput3D& committed,
    AdaptiveMacCellKey3D parent
) {
  return Prepare(committed_topology, committed, parent, nullptr);
}

}  // namespace fluidity
