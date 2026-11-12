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

FaceKey Key(const AdaptiveMacInteriorFace3D& face) {
  const auto index = face.face_index.lattice_index;
  return {face.face_index.axis, index.z, index.y, index.x, face.width_exponent};
}

std::optional<std::size_t> FindFace(std::span<const AdaptiveMacInteriorFace3D> faces, const FaceKey& key) {
  const auto face = std::ranges::lower_bound(faces, key, {}, Key);
  if (face == faces.end() || Key(*face) != key) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(face - faces.begin());
}

std::array<FaceKey, 4> SubfaceKeys(const AdaptiveMacInteriorFace3D& face) {
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

// The rounded coarse scalar must carry the entire outer-face flux. Accumulate
// the residual in nonoverlapping binary64 partials via ordered FastTwoSum so
// a small lost term remains visible even next to much larger signed terms.
bool RepresentsSubfaceSum(const std::array<LiquidRemapSubface3D, 4>& subfaces, double coarse_flux) {
  const std::array terms{
      subfaces[0].flux_m3_per_s,
      subfaces[1].flux_m3_per_s,
      subfaces[2].flux_m3_per_s,
      subfaces[3].flux_m3_per_s,
      -coarse_flux,
  };
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
    std::span<const int> child_index
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
  return outflows;
}

absl::Status RedistributeSiblingFluxes(
    const AdaptiveMacTopology3D& coarse,
    const AdaptiveFamilyChange3D& family,
    std::span<const double> coarse_fluxes,
    std::span<const int> child_index,
    std::span<double> fine_fluxes
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
  const auto outer = ChildOutflows(fine_faces, fine_fluxes, child_index);
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
  for (const auto& child : ChildOutflows(fine_faces, fine_fluxes, child_index)) {
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
  if (refining) {
    const auto redistribution =
        RedistributeSiblingFluxes(coarse, *family, committed.interior_fluxes_m3_per_s, child_index, fluxes);
    if (!redistribution.ok()) {
      return redistribution;
    }
  }
  return AdaptiveLiquidRemapProposal3D{
      .family = std::move(*family),
      .inventories = std::move(inventories),
      .interior_fluxes_m3_per_s = std::move(fluxes),
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
