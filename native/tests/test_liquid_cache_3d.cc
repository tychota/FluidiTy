#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "fluidity/host/liquid_cache_3d.h"
#include "fluidity/host/solid_interaction_checkpoint_3d.h"
#include "frame_manifest_generated.h"
#include "gtest/gtest.h"
#include "liquid_state_generated.h"
#include "point_frame_generated.h"
#include "resolved_scene_generated.h"
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

namespace fluidity {
namespace {
namespace wire = cache::v1;
template <typename T>
LiquidRecords3D<T> Records(std::size_t count) {
  auto result = LiquidRecords3D<T>::Create(count);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(result->Resize(count).ok());
  return std::move(*result);
}
LiquidState3D PopulatedState() {
  LiquidState3D state;
  state.geometry = {{1.25, -2.5, 3.75}, 0.125};
  state.clock = {2.5, 3, 0, 9, 12};
  state.random_seed = 0xfedcba9876543210;
  state.next_particle_id = 23;
  state.particles = Records<LiquidParticleState3D>(2);
  state.particles[0] = {
      .id = 22,
      .tag = 0xfedcba98,
      .world_position = {-1.25, 2.5, 3.75},
      .world_velocity = {4.5, -5.25, 6.75},
      .velocity_slopes = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
      .affine_enabled = false,
      .mass = 0.125,
      .reference_volume = 0.0625,
      .volume_ratio = 1.25,
      .side_tag = -19,
      .pic_initialized = true,
  };
  state.particles[1].id = 3;
  state.blocks = Records<LiquidBlockState3D>(2);
  state.blocks[0] = {{-1, 0, 0}, 5, 7, 11};
  state.blocks[1] = {{0, 0, 0}, 6, 8, 12};
  state.cells = Records<LiquidCellState3D>(2);
  state.cells[0] = {{-4, 0, 0}, LiquidCellOccupancy3D::kSolid, -13.25};
  state.cells[1] = {{0, 0, 0}, LiquidCellOccupancy3D::kAir, 14.5};
  state.faces = Records<LiquidFaceState3D>(2);
  state.faces[0] = {{FaceNormalAxis::kY, {-3, 0, 0}}, 65, -4.5, LiquidFaceValidity3D::kPredicted, -7, false};
  state.faces[1] = {{FaceNormalAxis::kZ, {0, 1, 2}}, 356, 5.75, LiquidFaceValidity3D::kLowMass, 9, true};
  state.sources = Records<LiquidSourceState3D>(2);
  state.sources[0] = {5, 0x123456789abcdef0, 0.25};
  state.sources[1] = {17, 29, 0.75};
  EXPECT_TRUE(ValidateLiquidState3D(state).ok());
  return state;
}
std::vector<std::uint8_t> Copy(const LiquidCacheBytes3D& bytes) {
  return {bytes.values().begin(), bytes.values().end()};
}
LiquidResumeMetadata3D Resume() {
  LiquidResumeMetadata3D resume;
  resume.resolved_fingerprint[0] = 13;
  resume.execution_id = LiquidExecutionId3D(resume.resolved_fingerprint);
  resume.capacities = {4, 4, 4, 4, 4};
  resume.capacity_replay_reservations = {8, 8, 8, 8, 8};
  return resume;
}
TEST(LiquidCache3D, ResolvedPlasticBodiesAreVersionedAndCanonical) {
  std::array bindings{
      PlasticSolidCheckpointBinding3D{9, 11, SnowParameters3D{1200, 700, 0.025, 0.01, 4}},
      PlasticSolidCheckpointBinding3D{12, 13, GranularParameters3D{450, 700, 0.5235987755982989, 0}},
  };
  LiquidResolvedSceneView3D scene;
  const auto baseline = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  scene.plastic_bodies = bindings;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  scene.model_version = kPlasticBodyResolvedModel3D;
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  ASSERT_EQ(decoded->model_version, kPlasticBodyResolvedModel3D);
  const auto view = decoded->View();
  ASSERT_EQ(view.plastic_bodies.size(), bindings.size());
  EXPECT_EQ(view.plastic_bodies[0].body_id, 9);
  EXPECT_EQ(view.plastic_bodies[1].material_id, 13);
  EXPECT_DOUBLE_EQ(std::get<SnowParameters3D>(view.plastic_bodies[0].material).shear_modulus, 1200);
  EXPECT_DOUBLE_EQ(std::get<GranularParameters3D>(view.plastic_bodies[1].material).lame_lambda, 700);
  const auto repeated = EncodeLiquidResolvedScene3D(view);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(Copy(*repeated), Copy(*encoded));
  const auto fingerprint = ResolvedSceneFingerprint(view);
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  EXPECT_NE(*fingerprint, *baseline);

  const auto& snow_binding = view.plastic_bodies[0];
  const DrySolidParticle3D compacted_snow{
      .world_position = {1, 2, 3},
      .mass = 300,
      .reference_volume = 1,
      .elastic_deformation = {{1, 0, 0}, {0, 0.975, 0}, {0, 0, 1}},
      .log_plastic_jacobian = std::log(0.8),
  };
  const auto particle = CapturePlasticSolidCheckpointParticle3D(
      5,
      snow_binding.body_id,
      snow_binding.material_id,
      snow_binding.material,
      compacted_snow
  );
  ASSERT_TRUE(particle.ok()) << particle.status();
  const SolidInteractionCheckpoint3D solid{.accepted_step = 0, .plastic_particles = {*particle}};
  const LiquidState3D liquid;
  const auto checkpoint = EncodeCoupledInteractionCheckpoint3D(liquid, solid, view.plastic_bodies);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  const auto restored = DecodeCoupledInteractionCheckpoint3D(*checkpoint, view.plastic_bodies);
  ASSERT_TRUE(restored.ok()) << restored.status();
  ASSERT_EQ(restored->solid.plastic_particles.size(), 1);
  EXPECT_DOUBLE_EQ(restored->solid.plastic_particles[0].log_plastic_jacobian, std::log(0.8));

  const auto bound_checkpoint = EncodeResolvedCoupledCheckpoint3D(view, liquid, solid);
  ASSERT_TRUE(bound_checkpoint.ok()) << bound_checkpoint.status();
  const auto bound_restored = DecodeResolvedCoupledCheckpoint3D(view, *bound_checkpoint);
  ASSERT_TRUE(bound_restored.ok()) << bound_restored.status();
  ASSERT_EQ(bound_restored->solid.plastic_particles.size(), 1);
  EXPECT_DOUBLE_EQ(bound_restored->solid.plastic_particles[0].log_plastic_jacobian, std::log(0.8));
  EXPECT_EQ(
      DecodeResolvedCoupledCheckpoint3D(view, *checkpoint).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  EXPECT_EQ(
      DecodeCoupledInteractionCheckpoint3D(*bound_checkpoint, view.plastic_bodies).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  auto alternate_bindings = bindings;
  std::get<SnowParameters3D>(alternate_bindings[0].material).hardening = 5;
  auto alternate_scene = scene;
  alternate_scene.plastic_bodies = alternate_bindings;
  ASSERT_TRUE(ValidatePlasticSolidCheckpointParticles3D(solid.plastic_particles, alternate_bindings).ok());
  EXPECT_EQ(
      DecodeResolvedCoupledCheckpoint3D(alternate_scene, *bound_checkpoint).status().code(),
      absl::StatusCode::kFailedPrecondition
  );
  alternate_scene = scene;
  ++alternate_scene.random_seed;
  EXPECT_EQ(
      EncodeResolvedCoupledCheckpoint3D(alternate_scene, liquid, solid).status().code(),
      absl::StatusCode::kInvalidArgument
  );

  auto wrong_version = Copy(*encoded);
  ASSERT_TRUE(wire::GetMutableResolvedScene(wrong_version.data())->mutate_model_version(kLiquidOnlyResolvedModel3D));
  EXPECT_EQ(DecodeLiquidResolvedScene3D(wrong_version).status().code(), absl::StatusCode::kDataLoss);
  std::swap(bindings[0], bindings[1]);
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  std::swap(bindings[0], bindings[1]);
  std::get<SnowParameters3D>(bindings[0].material).compression_limit = 1;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  scene.plastic_bodies = {};
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, ResolvedPhasePairIsCanonicalAndChangesFingerprint) {
  constexpr std::array<LiquidConstituentMaterial3D, 3> kMaterials{{
      {8, 4, 1000.0},
      {14, 4, 1020.0},
      {25, 5, 820.0},
  }};
  LiquidResolvedSceneView3D scene;
  scene.scene.constituent_materials = kMaterials;
  const auto baseline = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(baseline.ok()) << baseline.status();

  std::array<LiquidPhasePair3D, 1> pairs{{{4, 5, 0.031}}};
  scene.scene.phase_pairs = pairs;
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  const auto view = decoded->View();
  ASSERT_EQ(view.scene.phase_pairs.size(), 1);
  const auto* forward = FindLiquidPhasePair3D(view.scene.phase_pairs, 4, 5);
  ASSERT_NE(forward, nullptr);
  EXPECT_EQ(FindLiquidPhasePair3D(view.scene.phase_pairs, 5, 4), forward);
  EXPECT_EQ(FindLiquidPhasePair3D(view.scene.phase_pairs, 4, 4), nullptr);
  EXPECT_DOUBLE_EQ(forward->surface_tension, 0.031);
  const auto fingerprint = ResolvedSceneFingerprint(view);
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  EXPECT_NE(*fingerprint, *baseline);

  pairs[0] = {5, 4, 0.031};
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  pairs[0] = {4, 6, 0.031};
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  pairs[0] = {4, 5, -0.031};
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, ParticleCompositionSurvivesStateAndSourceEncoding) {
  auto state = PopulatedState();
  auto& particle = state.particles[0];
  particle.phase_id = 4;
  particle.constituent_count = 2;
  particle.constituents[0] = {8, 0.25};
  particle.constituents[1] = {14, 0.75};
  const auto bytes = EncodeLiquidState3D(state);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto decoded = DecodeLiquidState3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_TRUE(decoded->particles[0] == particle);
  EXPECT_EQ(decoded->particles[1].constituent_count, 0);

  const auto points = BuildLiquidPointFrame3D(state);
  ASSERT_TRUE(points.ok()) << points.status();
  const auto point_bytes = EncodeLiquidPointFrame3D(*points);
  ASSERT_TRUE(point_bytes.ok()) << point_bytes.status();
  const auto decoded_points = DecodeLiquidPointFrame3D(point_bytes->values());
  ASSERT_TRUE(decoded_points.ok()) << decoded_points.status();
  EXPECT_EQ(decoded_points->points[0].phase_id, 4);
  ASSERT_EQ(decoded_points->points[0].constituent_count, 2);
  EXPECT_EQ(decoded_points->points[0].constituents[0], (LiquidConstituentFraction3D{8, 0.25}));
  EXPECT_EQ(decoded_points->points[0].constituents[1], (LiquidConstituentFraction3D{14, 0.75}));
  EXPECT_EQ(decoded_points->points[1].constituent_count, 0);

  LiquidEmissionSource3D source;
  source.emitted_particle_template = particle;
  LiquidResolvedSceneView3D scene;
  scene.scene.sources = std::span(&source, 1);
  std::array<LiquidConstituentMaterial3D, 3> materials{{
      {8, 4, 1000.0},
      {14, 4, 1020.0},
      {25, 5, 820.0},
  }};
  scene.scene.constituent_materials = materials;
  const auto resolved = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  const auto loaded = DecodeLiquidResolvedScene3D(resolved->values());
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  ASSERT_EQ(loaded->View().scene.sources.size(), 1);
  ASSERT_EQ(loaded->View().scene.constituent_materials.size(), 3);
  EXPECT_EQ(loaded->View().scene.constituent_materials[2].phase_id, 5);
  EXPECT_DOUBLE_EQ(loaded->View().scene.constituent_materials[2].density_kg_per_m3, 820.0);
  const auto encoded_again = EncodeLiquidResolvedScene3D(loaded->View());
  ASSERT_TRUE(encoded_again.ok()) << encoded_again.status();
  EXPECT_TRUE(std::ranges::equal(resolved->values(), encoded_again->values()));
  EXPECT_TRUE(loaded->View().scene.sources[0].emitted_particle_template == particle);

  particle.constituents[1].substance_id = 8;
  EXPECT_EQ(EncodeLiquidState3D(state).status().code(), absl::StatusCode::kInvalidArgument);
  materials[1].phase_id = 5;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  materials[1].phase_id = 4;
  source.emitted_particle_template.constituents[1].volume_fraction = 0.5;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, CheckpointPreservesCavitySnapshot) {
  auto state = PopulatedState();
  state.next_cavity_id = 12;
  state.cavity_gas = *LiquidRecords3D<LiquidCavityGasState3D>::Create(2);
  state.cavity_components = *LiquidRecords3D<LiquidCavityComponentState3D>::Create(2);
  ASSERT_TRUE(state.cavity_gas.Resize(2).ok());
  ASSERT_TRUE(state.cavity_components.Resize(2).ok());
  state.cavity_gas[0] = {{-1, 0, 0}, 0.75, 0};
  state.cavity_gas[1] = {{1, 0, 0}, 0.25, 1};
  state.cavity_components[0] = {11, 1.0, 104000.0};

  const auto bytes = EncodeLiquidCheckpoint3D(state, Resume());
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  const auto decoded = DecodeLiquidCheckpoint3D(bytes->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_TRUE(decoded->state == state);
  const auto cloned = CloneLiquidState3D(decoded->state, Resume().capacities);
  ASSERT_TRUE(cloned.ok()) << cloned.status();
  EXPECT_TRUE(*cloned == state);

  state.cavity_components[1] = state.cavity_components[0];
  EXPECT_EQ(EncodeLiquidCheckpoint3D(state, Resume()).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, Blake3KnownAnswersAndExactByteIdentity) {
  EXPECT_EQ(
      std::string(LiquidDigestHex3D(LiquidContentDigest3D({})).data()),
      "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
  );
  constexpr std::array<std::uint8_t, 3> abc{'a', 'b', 'c'};
  EXPECT_EQ(
      std::string(LiquidDigestHex3D(LiquidContentDigest3D(abc)).data()),
      "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
  );
}

TEST(LiquidCache3D, PointBuilderPreservesPhysicalFieldsAndEquivalentVolumeRadius) {
  LiquidState3D state;
  state.particles = Records<LiquidParticleState3D>(1);
  state.next_particle_id = 2;
  state.particles[0].id = 1;
  state.particles[0].tag = 19;
  state.particles[0].world_position = {-0.0, -2.5, 3.5};
  state.particles[0].world_velocity = {1, -2, 3};
  const auto original = EncodeLiquidState3D(state);
  ASSERT_TRUE(original.ok());
  auto points = BuildLiquidPointFrame3D(state);
  ASSERT_TRUE(points.ok()) << points.status();
  ASSERT_EQ(points->points.size(), 1);
  const auto& point = points->points[0];
  EXPECT_EQ(point.id, 1);
  EXPECT_EQ(point.tag, 19);
  EXPECT_TRUE(std::signbit(point.world_position.x));
  EXPECT_EQ(point.world_position.y, -2.5);
  EXPECT_EQ(point.world_velocity.z, 3);
  EXPECT_NEAR(point.radius, 0.6203504908994001, 1e-15);
  EXPECT_EQ(point.volume_ratio, 1);
  EXPECT_EQ(point.level, 0);
  const auto after = EncodeLiquidState3D(state);
  ASSERT_TRUE(after.ok());
  EXPECT_TRUE(std::ranges::equal(original->values(), after->values()));
}

TEST(LiquidCache3D, PointBuilderRejectsIncompleteClockBudgetsAndUnrepresentableRadius) {
  auto state = PopulatedState();
  LiquidCacheLimits3D limits;
  limits.maximum_records.particles = 1;
  EXPECT_EQ(BuildLiquidPointFrame3D(state, limits).status().code(), absl::StatusCode::kResourceExhausted);
  state.clock.substep_ordinal = 1;
  EXPECT_EQ(BuildLiquidPointFrame3D(state).status().code(), absl::StatusCode::kInvalidArgument);
  state.clock.substep_ordinal = 0;
  state.particles[0].reference_volume = std::numeric_limits<double>::max();
  EXPECT_EQ(BuildLiquidPointFrame3D(state).status().code(), absl::StatusCode::kOutOfRange);
}
TEST(LiquidCache3D, StateRoundTripPreservesEverySemanticFieldAndStableOrder) {
  auto state = PopulatedState();
  auto encoded = EncodeLiquidState3D(state);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto decoded = DecodeLiquidState3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, state);
  EXPECT_NE(decoded->particles.values().data(), state.particles.values().data());
  auto larger = CloneLiquidState3D(state, {9, 9, 9, 9, 9});
  ASSERT_TRUE(larger.ok());
  auto repeated = EncodeLiquidState3D(*larger);
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(Copy(*encoded), Copy(*repeated));
}
TEST(LiquidCache3D, CheckpointSeparatesResumePlanAndRequiresCompletedFrame) {
  auto state = PopulatedState();
  auto encoded = EncodeLiquidCheckpoint3D(state, Resume());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto decoded = DecodeLiquidCheckpoint3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->state, state);
  EXPECT_EQ(decoded->resume, Resume());
  auto plain = EncodeLiquidState3D(state);
  ASSERT_TRUE(plain.ok());
  EXPECT_EQ(DecodeLiquidCheckpoint3D(plain->values()).status().code(), absl::StatusCode::kDataLoss);
  state.clock.substep_ordinal = 1;
  EXPECT_EQ(EncodeLiquidCheckpoint3D(state, Resume()).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(LiquidCache3D, PointAndManifestRoundTripKeepPortableContentReference) {
  LiquidPointFrame3D frame;
  frame.clock = {1, 1, 0, 2, 2};
  frame.points = Records<LiquidPoint3D>(1);
  frame.points[0] = {29, 0xff0011, {-1, 2, 3}, {4, -5, 6}, 0.125, 1.25, 0};
  auto encoded = EncodeLiquidPointFrame3D(frame);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto decoded = DecodeLiquidPointFrame3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->clock, frame.clock);
  ASSERT_EQ(decoded->points.size(), 1);
  const auto& point = decoded->points[0];
  EXPECT_EQ(point.id, 29);
  EXPECT_EQ(point.tag, 0xff0011);
  EXPECT_EQ(point.world_position.x, -1);
  EXPECT_EQ(point.world_position.y, 2);
  EXPECT_EQ(point.world_position.z, 3);
  EXPECT_EQ(point.world_velocity.x, 4);
  EXPECT_EQ(point.world_velocity.y, -5);
  EXPECT_EQ(point.world_velocity.z, 6);
  EXPECT_EQ(point.radius, 0.125);
  EXPECT_EQ(point.volume_ratio, 1.25);
  EXPECT_EQ(point.level, 0);
  LiquidFrameManifest3D manifest{
      frame.clock,
      Resume().resolved_fingerprint,
      Resume().execution_id,
      {LiquidContentKind3D::kPoints, encoded->size(), LiquidContentDigest3D(encoded->values())},
  };
  auto manifest_bytes = EncodeLiquidFrameManifest3D(manifest);
  ASSERT_TRUE(manifest_bytes.ok()) << manifest_bytes.status();
  auto read_manifest = DecodeLiquidFrameManifest3D(manifest_bytes->values());
  ASSERT_TRUE(read_manifest.ok()) << read_manifest.status();
  EXPECT_EQ(read_manifest->clock, manifest.clock);
  EXPECT_EQ(read_manifest->resolved_fingerprint, manifest.resolved_fingerprint);
  EXPECT_EQ(read_manifest->execution_id, manifest.execution_id);
  EXPECT_EQ(read_manifest->points, manifest.points);
  EXPECT_TRUE(VerifyLiquidContent3D(encoded->values(), read_manifest->points, LiquidContentKind3D::kPoints).ok());
  auto changed = Copy(*encoded);
  changed.back() ^= 1;
  EXPECT_EQ(
      VerifyLiquidContent3D(changed, read_manifest->points, LiquidContentKind3D::kPoints).code(),
      absl::StatusCode::kDataLoss
  );
}
TEST(LiquidCache3D, ContentNamesRejectTraversalAlternateSpellingAndWrongKind) {
  LiquidContentReference3D ref{LiquidContentKind3D::kPoints, 8, {}};
  auto name = LiquidContentName3D(ref);
  ASSERT_TRUE(name.ok()) << name.status();
  EXPECT_EQ(std::string(name->data()), "blobs/" + std::string(64, '0') + ".flpt");
  EXPECT_TRUE(ValidateLiquidContentName3D(name->data(), ref).ok());
  for (const std::string bad : {"../x", "/tmp/x", "C:/x", "blobs/../x", "blobs\\x", ".", ""}) {
    EXPECT_EQ(ValidateLiquidContentName3D(bad, ref).code(), absl::StatusCode::kInvalidArgument);
  }
  auto nul_name = std::string(name->data()) + std::string(1, '\0');
  EXPECT_FALSE(ValidateLiquidContentName3D(nul_name, ref).ok());
  ref.kind = static_cast<LiquidContentKind3D>(255);
  EXPECT_EQ(LiquidContentName3D(ref).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(LiquidCache3D, MalformedWireRejectsBeforeExposingRecords) {
  EXPECT_EQ(DecodeLiquidState3D({}).status().code(), absl::StatusCode::kDataLoss);
  auto encoded = EncodeLiquidState3D(PopulatedState());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  for (std::size_t size = 0; size < encoded->size(); ++size) {
    EXPECT_FALSE(DecodeLiquidState3D(encoded->values().first(size)).ok()) << size;
  }
  auto changed = Copy(*encoded);
  changed[4] ^= 1;
  EXPECT_EQ(DecodeLiquidState3D(changed).status().code(), absl::StatusCode::kDataLoss);
  changed = Copy(*encoded);
  ASSERT_TRUE(wire::GetMutableLiquidState(changed.data())->mutate_schema_version(2));
  EXPECT_EQ(DecodeLiquidState3D(changed).status().code(), absl::StatusCode::kFailedPrecondition);
  changed = Copy(*encoded);
  ASSERT_TRUE(wire::GetMutableLiquidState(changed.data())->mutate_byte_length(encoded->size() + 1));
  EXPECT_EQ(DecodeLiquidState3D(changed).status().code(), absl::StatusCode::kDataLoss);
  LiquidCacheLimits3D limits;
  limits.maximum_records.particles = 1;
  EXPECT_EQ(DecodeLiquidState3D(encoded->values(), limits).status().code(), absl::StatusCode::kResourceExhausted);
  limits.maximum_bytes = encoded->size() - 1;
  EXPECT_EQ(DecodeLiquidState3D(encoded->values(), limits).status().code(), absl::StatusCode::kResourceExhausted);
}
TEST(LiquidCache3D, StateSemanticCorruptionRejectsEnumsOrderAndDomain) {
  auto encoded = EncodeLiquidState3D(PopulatedState());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  for (int mutation = 0; mutation < 12; ++mutation) {
    auto changed = Copy(*encoded);
    auto* state = wire::GetMutableLiquidState(changed.data());
    switch (mutation) {
      case 0:
        state->mutable_cells()->GetMutableObject(0)->mutate_occupancy(static_cast<wire::Occupancy>(255));
        break;
      case 1:
        state->mutable_faces()->GetMutableObject(0)->mutate_axis(static_cast<wire::Axis>(255));
        break;
      case 2:
        state->mutable_faces()->GetMutableObject(0)->mutate_validity(static_cast<wire::Validity>(255));
        break;
      case 3:
        state->mutable_particles()->GetMutableObject(0)->mutate_volume_ratio(0);
        break;
      case 4:
        state->mutable_particles()->GetMutableObject(0)->mutate_mass(std::numeric_limits<double>::infinity());
        break;
      case 5:
        state->mutable_particles()->GetMutableObject(0)->mutate_id(3);
        break;
      case 6:
        state->mutable_faces()->GetMutableObject(0)->mutate_slot(0);
        break;
      case 7:
        state->mutable_sources()->GetMutableObject(1)->mutate_ordinal(5);
        break;
      case 8:
        state->mutable_blocks()->GetMutableObject(0)->mutable_index()->mutate_x(INT32_MAX);
        break;
      case 9:
        state->mutable_clock()->mutate_accepted_step(0);
        break;
      case 10:
        state->mutable_cells()->GetMutableObject(0)->mutable_index()->mutate_z(99);
        break;
      case 11:
        flatbuffers::WriteScalar<flatbuffers::uoffset_t>(
            reinterpret_cast<std::uint8_t*>(state->mutable_particles()),
            UINT32_MAX
        );
        break;
    }
    EXPECT_EQ(DecodeLiquidState3D(changed).status().code(), absl::StatusCode::kDataLoss) << mutation;
  }
}
TEST(LiquidCache3D, ResolvedOpenTankRoundTrips) {
  LiquidResolvedSceneView3D scene;
  scene.scene.obstacle = AxisAlignedLiquidTank3D{-1.25, 1.5, 0, -0.5, 0.75, 0.125};
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  const auto* tank = std::get_if<AxisAlignedLiquidTank3D>(&decoded->scene.obstacle);
  ASSERT_NE(tank, nullptr);
  EXPECT_EQ(tank->start_low_x, -1.25);
  EXPECT_EQ(tank->high_x, 1.5);
  EXPECT_EQ(tank->floor_y, 0);
  EXPECT_EQ(tank->low_z, -0.5);
  EXPECT_EQ(tank->high_z, 0.75);
  EXPECT_EQ(tank->piston_velocity_x, 0.125);
  const auto repeated = EncodeLiquidResolvedScene3D(decoded->View());
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(Copy(*encoded), Copy(*repeated));
}

TEST(LiquidCache3D, ResolvedHorizontalPlaneRequiresTankAndMaterializedPayloads) {
  LiquidResolvedSceneView3D scene;
  scene.scene.field = LiquidHorizontalPlaneSurface3D{-0.0};
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  scene.scene.obstacle = AxisAlignedLiquidTank3D{-1, 1, -1, -1, 1, 0};
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  const auto* plane = std::get_if<LiquidHorizontalPlaneSurface3D>(&decoded->scene.field);
  ASSERT_NE(plane, nullptr);
  EXPECT_DOUBLE_EQ(plane->world_surface_height_y, 0);
  EXPECT_FALSE(std::signbit(plane->world_surface_height_y));
  const auto repeated = EncodeLiquidResolvedScene3D(decoded->View());
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(Copy(*encoded), Copy(*repeated));

  const auto omit_field = [](auto* record, flatbuffers::voffset_t field) {
    auto* table = reinterpret_cast<std::uint8_t*>(record);
    auto* vtable = table - flatbuffers::ReadScalar<flatbuffers::soffset_t>(table);
    flatbuffers::WriteScalar<flatbuffers::voffset_t>(vtable + field, 0);
  };
  for (int mutation = 0; mutation < 10; ++mutation) {
    auto changed = Copy(*encoded);
    auto* root = wire::GetMutableResolvedScene(changed.data());
    auto* field = root->mutable_field();
    auto* obstacle = root->mutable_obstacle();
    switch (mutation) {
      case 0:
        omit_field(field, wire::Field::VT_HORIZONTAL_PLANE);
        break;
      case 1:
        omit_field(field->mutable_horizontal_plane(), wire::HorizontalPlane::VT_SURFACE_HEIGHT_Y);
        break;
      case 2:
        ASSERT_TRUE(field->mutate_kind(wire::FieldKind::AnalyticSphere));
        break;
      case 3:
        omit_field(obstacle, wire::Obstacle::VT_OPEN_TANK);
        break;
      case 4:
        omit_field(obstacle->mutable_open_tank(), wire::OpenTank::VT_PISTON_VELOCITY_X);
        break;
      case 5:
        ASSERT_TRUE(obstacle->mutate_kind(wire::ObstacleKind::None));
        break;
      case 6:
        ASSERT_TRUE(field->mutable_horizontal_plane()->mutate_surface_height_y(std::numeric_limits<double>::infinity())
        );
        break;
      case 7:
        ASSERT_TRUE(obstacle->mutable_open_tank()->mutate_high_x(-1));
        break;
      case 8:
        ASSERT_TRUE(root->mutable_material()->mutate_friction(0.125));
        break;
      case 9:
        ASSERT_TRUE(field->mutate_radius(2));
        break;
    }
    EXPECT_EQ(DecodeLiquidResolvedScene3D(changed).status().code(), absl::StatusCode::kDataLoss) << mutation;
  }

  scene.scene.friction_coefficient = 0.125;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, ResolvedViscosityRoundTripsAndChangesFingerprint) {
  LiquidResolvedSceneView3D scene;
  const auto baseline = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  scene.scene.dynamic_viscosity = 12.5;
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_DOUBLE_EQ(decoded->View().scene.dynamic_viscosity, 12.5);
  const auto fingerprint = ResolvedSceneFingerprint(decoded->View());
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  EXPECT_NE(*baseline, *fingerprint);
  scene.scene.dynamic_viscosity = -1.0;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, ResolvedCavityEquilibriumRoundTripsAndChangesFingerprint) {
  LiquidResolvedSceneView3D scene;
  const auto baseline = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  scene.scene.cavity = LiquidSceneCavity3D{
      .atmospheric_absolute_pressure = 100000.0,
      .birth_hydrostatic_gauge_pressure = 4000.0,
  };
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  ASSERT_TRUE(decoded->scene.cavity);
  EXPECT_DOUBLE_EQ(decoded->scene.cavity->atmospheric_absolute_pressure, 100000.0);
  EXPECT_DOUBLE_EQ(decoded->scene.cavity->birth_hydrostatic_gauge_pressure, 4000.0);
  const auto fingerprint = ResolvedSceneFingerprint(decoded->View());
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  EXPECT_NE(*fingerprint, *baseline);
  scene.scene.cavity->birth_hydrostatic_gauge_pressure = -1.0;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, ResolvedEosRoundTripsAndChangesFingerprint) {
  LiquidResolvedSceneView3D scene;
  const auto baseline = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  scene.scene.eos = LiquidSceneEos3D{.reference_sound_speed = 2.0, .law = LiquidEosLaw::kTait};
  scene.scene.drift = LiquidSceneDrift3D{.correction_time = 0.2, .maximum_rate = 0.3};
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  const auto round_trip = decoded->View();
  ASSERT_TRUE(round_trip.scene.eos.has_value());
  EXPECT_EQ(round_trip.scene.eos->reference_sound_speed, 2.0);
  EXPECT_EQ(round_trip.scene.eos->law, LiquidEosLaw::kTait);
  ASSERT_TRUE(round_trip.scene.drift.has_value());
  EXPECT_EQ(round_trip.scene.drift->correction_time, 0.2);
  EXPECT_EQ(round_trip.scene.drift->maximum_rate, 0.3);
  const auto fingerprint = ResolvedSceneFingerprint(round_trip);
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  EXPECT_NE(*baseline, *fingerprint);
  scene.scene.bulk_modulus = 4000.0;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(LiquidCache3D, ResolvedDefaultsMaterializeAndRoundTrip) {
  LiquidResolvedSceneView3D scene;
  auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto* root = wire::GetResolvedScene(encoded->values().data());
  EXPECT_TRUE(flatbuffers::IsFieldPresent(root, wire::ResolvedScene::VT_BACKEND));
  EXPECT_TRUE(flatbuffers::IsFieldPresent(root->material(), wire::Material::VT_SURFACE_TENSION));
  EXPECT_TRUE(flatbuffers::IsFieldPresent(root->material(), wire::Material::VT_BULK_MODULUS));
  EXPECT_TRUE(flatbuffers::IsFieldPresent(root->options(), wire::StepOptions::VT_ENERGY_BLEND));
  EXPECT_TRUE(flatbuffers::IsFieldPresent(root->field(), wire::Field::VT_KIND));
  EXPECT_TRUE(flatbuffers::IsFieldPresent(root->obstacle(), wire::Obstacle::VT_KIND));
  auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  auto repeated = EncodeLiquidResolvedScene3D(decoded->View());
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(Copy(*encoded), Copy(*repeated));
  scene.scene.liquid_density = 1000;
  scene.scene.surface_tension = 0;
  scene.scene.bulk_modulus = std::numeric_limits<double>::infinity();
  scene.options.particle_energy_blend = 0.6;
  scene.scene.world_acceleration.x = -0.0;
  auto explicit_defaults = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(explicit_defaults.ok());
  EXPECT_EQ(Copy(*encoded), Copy(*explicit_defaults));
  scene.options.capacities = {100, 100, 100, 100, 100};
  auto capacity_only = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(capacity_only.ok());
  EXPECT_EQ(Copy(*encoded), Copy(*capacity_only));
}

TEST(LiquidCache3D, EffectsCaptureSettingsAreOptionalAndFingerprintTheirSamplePolicy) {
  LiquidResolvedSceneView3D scene;
  const auto baseline = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(baseline.ok());
  const auto original = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(original.ok());
  scene.options.motion_sample_intervals = 16;
  const auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok());
  auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ(decoded->options.motion_sample_intervals, 16);
  EXPECT_NE(*original, *ResolvedSceneFingerprint(scene));
  EXPECT_EQ(Copy(*encoded), Copy(*EncodeLiquidResolvedScene3D(decoded->View())));
  scene.options.motion_sample_intervals = 65;
  EXPECT_EQ(EncodeLiquidResolvedScene3D(scene).status().code(), absl::StatusCode::kInvalidArgument);
  scene.options.motion_sample_intervals = 0;
  EXPECT_EQ(Copy(*baseline), Copy(*EncodeLiquidResolvedScene3D(scene)));
}
TEST(LiquidCache3D, ResolvedFingerprintChangesWithSettingsAndExactUpstreamContent) {
  LiquidResolvedSceneView3D scene;
  auto baseline = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  for (int change = 0; change < 7; ++change) {
    auto variant = scene;
    LiquidUpstreamContent3D upstream;
    switch (change) {
      case 0:
        variant.scene.surface_tension = 0.072;
        break;
      case 1:
        variant.backend = LiquidBackend3D::kCuda;
        break;
      case 2:
        variant.build_digest[0] = 1;
        break;
      case 3:
        variant.random_seed = 1;
        break;
      case 4:
        variant.options.maximum_step_duration = 0.02;
        break;
      case 5:
        variant.scene.field = LiquidAnalyticSphere3D{{1, 2, 3}, 0.25};
        break;
      case 6:
        upstream.digest[0] = 1;
        variant.upstream = {&upstream, 1};
        break;
    }
    auto fingerprint = ResolvedSceneFingerprint(variant);
    ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
    EXPECT_NE(*fingerprint, *baseline) << change;
    EXPECT_NE(LiquidExecutionId3D(*fingerprint), LiquidExecutionId3D(*baseline)) << change;
  }
}
TEST(LiquidCache3D, CanonicalResolvedBytesMatchIndependentFlatcGolden) {
  std::ifstream file(FLUIDITY_CACHE_GOLDEN_PATH);
  std::string hex;
  ASSERT_TRUE(static_cast<bool>(file >> hex));
  std::vector<std::uint8_t> golden;
  for (std::size_t index = 0; index < hex.size(); index += 2) {
    golden.push_back(static_cast<std::uint8_t>(std::stoul(hex.substr(index, 2), nullptr, 16)));
  }
  auto encoded = EncodeLiquidResolvedScene3D({});
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(Copy(*encoded), golden);
  ASSERT_EQ(encoded->size(), golden.size());
  for (std::size_t index = 0; index < golden.size(); ++index) {
    EXPECT_EQ((*encoded)[index], golden[index]) << "golden byte " << index;
  }
  auto fingerprint = ResolvedSceneFingerprint({});
  ASSERT_TRUE(fingerprint.ok()) << fingerprint.status();
  EXPECT_EQ(
      std::string(LiquidDigestHex3D(*fingerprint).data()),
      "f9970683b8708e8963451711f5f03a62d1ec6830c5815f43dda539fec94eddd0"
  );
}
TEST(LiquidCache3D, ResolvedSourceWallAndAllMaterialOptionsRoundTrip) {
  LiquidResolvedSceneView3D scene;
  scene.geometry = {{1, -2, 3}, 0.25};
  scene.random_seed = 123;
  scene.backend = LiquidBackend3D::kCuda;
  scene.build_digest.fill(42);
  scene.scene.field = LiquidAnalyticSphere3D{{4, 5, -6}, 0.75};
  scene.scene.obstacle = TranslatingSlab3D{{0, 1, 0}, -2.5, 0.125, {1, 2, 3}};
  scene.scene.world_acceleration = {4, 5, 6};
  scene.scene.liquid_density = 997;
  scene.scene.surface_tension = 0.072;
  scene.scene.ambient_pressure = -7;
  scene.scene.bulk_modulus = 8000;
  scene.scene.target_divergence = 0.125;
  scene.scene.friction_coefficient = 0.25;
  scene.scene.penetration_tolerance = 0.03125;
  scene.options.particle_energy_blend = 0.75;
  scene.options.maximum_step_duration = 0.002;
  scene.options.minimum_face_mass = 0.001;
  scene.options.mass_per_quantum = 0x1p-30;
  scene.options.momentum_per_quantum = 0x1p-35;
  scene.options.pressure_solver_options.max_iterations = 17;
  LiquidEmissionSource3D source;
  source.source_ordinal = 99;
  source.particles_per_second = 125;
  source.world_position = {1, 2, 3};
  source.world_position_jitter = {0.125, 0.25, 0.5};
  source.emitted_particle_template = PopulatedState().particles[0];
  scene.scene.sources = {&source, 1};
  const LiquidUpstreamContent3D upstream{LiquidUpstreamRole3D::kSource, 99, 12345, scene.build_digest};
  scene.upstream = {&upstream, 1};
  auto encoded = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  const auto* root = wire::GetResolvedScene(encoded->values().data());
  EXPECT_EQ(root->sources()->Get(0)->particle_template()->tag(), source.emitted_particle_template.tag);
  EXPECT_TRUE(
      flatbuffers::IsFieldPresent(root->sources()->Get(0)->particle_template(), wire::Particle::VT_AFFINE_ENABLED)
  );
  auto decoded = DecodeLiquidResolvedScene3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->sources[0].emitted_particle_template, source.emitted_particle_template);
  EXPECT_EQ(decoded->sources[0].source_ordinal, 99);
  EXPECT_EQ(decoded->sources[0].particles_per_second, 125);
  EXPECT_EQ(decoded->sources[0].world_position_jitter.z, 0.5);
  EXPECT_EQ(decoded->upstream[0].digest, upstream.digest);
  EXPECT_EQ(decoded->geometry.world_origin.x, 1);
  EXPECT_EQ(decoded->geometry.world_origin.y, -2);
  EXPECT_EQ(decoded->geometry.world_origin.z, 3);
  EXPECT_EQ(decoded->geometry.cell_width, 0.25);
  EXPECT_EQ(decoded->random_seed, 123);
  EXPECT_EQ(decoded->backend, LiquidBackend3D::kCuda);
  EXPECT_EQ(decoded->model_version, 1);
  EXPECT_EQ(decoded->build_digest, scene.build_digest);
  EXPECT_EQ(decoded->scene.world_acceleration.x, 4);
  EXPECT_EQ(decoded->scene.world_acceleration.y, 5);
  EXPECT_EQ(decoded->scene.world_acceleration.z, 6);
  EXPECT_EQ(decoded->scene.liquid_density, 997);
  EXPECT_EQ(decoded->scene.surface_tension, 0.072);
  EXPECT_EQ(decoded->scene.ambient_pressure, -7);
  EXPECT_EQ(decoded->scene.bulk_modulus, 8000);
  EXPECT_EQ(decoded->scene.target_divergence, 0.125);
  EXPECT_EQ(decoded->scene.friction_coefficient, 0.25);
  EXPECT_EQ(decoded->scene.penetration_tolerance, 0.03125);
  EXPECT_EQ(decoded->options.particle_energy_blend, 0.75);
  EXPECT_EQ(decoded->options.maximum_step_duration, 0.002);
  EXPECT_EQ(decoded->options.minimum_face_mass, 0.001);
  EXPECT_EQ(decoded->options.mass_per_quantum, 0x1p-30);
  EXPECT_EQ(decoded->options.momentum_per_quantum, 0x1p-35);
  EXPECT_EQ(decoded->options.pressure_solver_options.max_iterations, 17);
  const auto* sphere = std::get_if<LiquidAnalyticSphere3D>(&decoded->scene.field);
  ASSERT_NE(sphere, nullptr);
  EXPECT_EQ(sphere->world_center.x, 4);
  EXPECT_EQ(sphere->world_center.y, 5);
  EXPECT_EQ(sphere->world_center.z, -6);
  EXPECT_EQ(sphere->radius, 0.75);
  const auto* slab = std::get_if<TranslatingSlab3D>(&decoded->scene.obstacle);
  ASSERT_NE(slab, nullptr);
  EXPECT_EQ(slab->unit_normal.x, 0);
  EXPECT_EQ(slab->unit_normal.y, 1);
  EXPECT_EQ(slab->unit_normal.z, 0);
  EXPECT_EQ(slab->start_center_coordinate, -2.5);
  EXPECT_EQ(slab->half_width, 0.125);
  EXPECT_EQ(slab->wall_velocity.x, 1);
  EXPECT_EQ(slab->wall_velocity.y, 2);
  EXPECT_EQ(slab->wall_velocity.z, 3);
  EXPECT_EQ(decoded->sources[0].world_position.x, 1);
  EXPECT_EQ(decoded->sources[0].world_position.y, 2);
  EXPECT_EQ(decoded->sources[0].world_position.z, 3);
  EXPECT_EQ(decoded->sources[0].world_position_jitter.x, 0.125);
  EXPECT_EQ(decoded->sources[0].world_position_jitter.y, 0.25);
  EXPECT_EQ(decoded->upstream[0].role, LiquidUpstreamRole3D::kSource);
  EXPECT_EQ(decoded->upstream[0].ordinal, 99);
  EXPECT_EQ(decoded->upstream[0].byte_length, 12345);
  auto repeated = EncodeLiquidResolvedScene3D(decoded->View());
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(Copy(*encoded), Copy(*repeated));
}
TEST(LiquidCache3D, SameLengthUpstreamByteChangeInvalidatesFingerprintAndExecution) {
  LiquidResolvedSceneView3D scene;
  constexpr std::array<std::uint8_t, 3> first_bytes{'a', 'b', 'c'};
  constexpr std::array<std::uint8_t, 3> changed_bytes{'a', 'b', 'd'};
  LiquidUpstreamContent3D upstream{LiquidUpstreamRole3D::kInitialParticles, 7, 3, LiquidContentDigest3D(first_bytes)};
  scene.upstream = {&upstream, 1};
  auto first = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(first.ok());
  upstream.digest = LiquidContentDigest3D(changed_bytes);
  auto changed = ResolvedSceneFingerprint(scene);
  ASSERT_TRUE(changed.ok());
  EXPECT_NE(*first, *changed);
  EXPECT_NE(LiquidExecutionId3D(*first), LiquidExecutionId3D(*changed));
}
TEST(LiquidCache3D, HalfspacePoseAndParticleUnionRadiusRoundTrip) {
  LiquidResolvedSceneView3D scene;
  scene.scene.field = LiquidParticleSphereUnion3D{0.375};
  scene.scene.obstacle = TranslatingHalfspace3D{{0, 0, -1}, 0.25, {-1, -2, -3}};
  auto bytes = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(bytes.ok());
  auto decoded = DecodeLiquidResolvedScene3D(bytes->values());
  ASSERT_TRUE(decoded.ok());
  const auto* halfspace = std::get_if<TranslatingHalfspace3D>(&decoded->scene.obstacle);
  ASSERT_NE(halfspace, nullptr);
  EXPECT_EQ(halfspace->unit_normal.x, 0);
  EXPECT_EQ(halfspace->unit_normal.y, 0);
  EXPECT_EQ(halfspace->unit_normal.z, -1);
  EXPECT_EQ(halfspace->start_plane_coordinate, 0.25);
  EXPECT_EQ(halfspace->wall_velocity.x, -1);
  EXPECT_EQ(halfspace->wall_velocity.y, -2);
  EXPECT_EQ(halfspace->wall_velocity.z, -3);
  const auto* field = std::get_if<LiquidParticleSphereUnion3D>(&decoded->scene.field);
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->radius, 0.375);
}
TEST(LiquidCache3D, ResolvedInvalidEnumsAndDomainsAreRejected) {
  auto encoded = EncodeLiquidResolvedScene3D({});
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto changed = Copy(*encoded);
    auto* scene = wire::GetMutableResolvedScene(changed.data());
    switch (mutation) {
      case 0:
        scene->mutate_backend(static_cast<wire::Backend>(255));
        break;
      case 1:
        scene->mutable_field()->mutate_kind(static_cast<wire::FieldKind>(255));
        break;
      case 2:
        scene->mutable_obstacle()->mutate_kind(static_cast<wire::ObstacleKind>(255));
        break;
      case 3:
        scene->mutable_material()->mutate_density(0);
        break;
      case 4:
        scene->mutable_options()->mutate_energy_blend(1.5);
        break;
      case 5:
        scene->mutable_options()->mutate_mass_per_quantum(0.3);
        break;
      case 6:
        scene->mutable_field()->mutate_radius(std::numeric_limits<double>::quiet_NaN());
        break;
    }
    EXPECT_EQ(DecodeLiquidResolvedScene3D(changed).status().code(), absl::StatusCode::kDataLoss) << mutation;
  }
}
TEST(LiquidCache3D, ConfiguredMaximumFamiliesFitTheBoundedEncoderArena) {
  LiquidState3D state;
  state.particles = Records<LiquidParticleState3D>(1024);
  state.blocks = Records<LiquidBlockState3D>(4096);
  state.cells = Records<LiquidCellState3D>(4096);
  state.faces = Records<LiquidFaceState3D>(24576);
  state.sources = Records<LiquidSourceState3D>(256);
  state.next_particle_id = 1025;
  for (std::size_t index = 0; index < state.particles.size(); ++index) {
    state.particles[index].id = 1024 - index;
  }
  for (std::size_t index = 0; index < state.blocks.size(); ++index) {
    state.blocks[index].block_index = {static_cast<std::int32_t>(index), 0, 0};
    state.cells[index].cell_index = {static_cast<std::int32_t>(4 * index), 0, 0};
  }
  for (std::size_t slot = 0; slot < state.faces.size(); ++slot) {
    const auto block = static_cast<std::int32_t>(slot / 192);
    const auto local = static_cast<std::int32_t>(slot % 64);
    state.faces[slot].face_index = {
        static_cast<FaceNormalAxis>((slot % 192) / 64),
        {4 * block + local % 4, (local / 4) % 4, local / 16},
    };
    state.faces[slot].face_slot = slot;
  }
  for (std::size_t index = 0; index < state.sources.size(); ++index) {
    state.sources[index].source_ordinal = index;
  }
  auto encoded = EncodeLiquidState3D(state);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  auto decoded = DecodeLiquidState3D(encoded->values());
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, state);
  LiquidResolvedSceneView3D scene;
  auto sources = Records<LiquidEmissionSource3D>(256);
  auto upstream = Records<LiquidUpstreamContent3D>(256);
  for (std::size_t index = 0; index < 256; ++index) {
    sources[index].source_ordinal = index;
    upstream[index].ordinal = index;
  }
  scene.scene.sources = sources.values();
  scene.upstream = upstream.values();
  auto resolved = EncodeLiquidResolvedScene3D(scene);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  auto read_scene = DecodeLiquidResolvedScene3D(resolved->values());
  ASSERT_TRUE(read_scene.ok()) << read_scene.status();
  EXPECT_EQ(read_scene->sources.size(), 256);
  EXPECT_EQ(read_scene->upstream.size(), 256);
}
TEST(LiquidCache3D, MalformedPointManifestAndResolvedRecordsRejectIdentifiersAndLengths) {
  LiquidPointFrame3D point;
  auto point_bytes = EncodeLiquidPointFrame3D(point);
  ASSERT_TRUE(point_bytes.ok());
  LiquidFrameManifest3D manifest{
      {},
      {},
      LiquidExecutionId3D({}),
      {LiquidContentKind3D::kPoints, point_bytes->size(), LiquidContentDigest3D(point_bytes->values())},
  };
  auto manifest_bytes = EncodeLiquidFrameManifest3D(manifest);
  auto scene_bytes = EncodeLiquidResolvedScene3D({});
  ASSERT_TRUE(manifest_bytes.ok());
  ASSERT_TRUE(scene_bytes.ok());
  const auto exercise = [](const LiquidCacheBytes3D& encoded, auto decode) {
    EXPECT_EQ(decode(std::span<const std::uint8_t>{}).code(), absl::StatusCode::kDataLoss);
    for (std::size_t size = 0; size < encoded.size(); ++size) {
      EXPECT_FALSE(decode(encoded.values().first(size)).ok()) << size;
    }
    auto changed = Copy(encoded);
    changed[4] ^= 1;
    EXPECT_EQ(decode(changed).code(), absl::StatusCode::kDataLoss);
    changed = Copy(encoded);
    std::fill_n(changed.begin(), 4, 255);
    EXPECT_EQ(decode(changed).code(), absl::StatusCode::kDataLoss);
    changed = Copy(encoded);
    changed.push_back(0);
    EXPECT_EQ(decode(changed).code(), absl::StatusCode::kDataLoss);
  };
  exercise(*point_bytes, [](auto bytes) {
    return DecodeLiquidPointFrame3D(bytes).status();
  });
  exercise(*manifest_bytes, [](auto bytes) {
    return DecodeLiquidFrameManifest3D(bytes).status();
  });
  exercise(*scene_bytes, [](auto bytes) {
    return DecodeLiquidResolvedScene3D(bytes).status();
  });
}
TEST(LiquidCache3D, ManifestRejectsStatusKindsAndContentPathTampering) {
  const LiquidFrameManifest3D manifest{{}, {}, LiquidExecutionId3D({}), {LiquidContentKind3D::kPoints, 128, {}}};
  auto encoded = EncodeLiquidFrameManifest3D(manifest);
  ASSERT_TRUE(encoded.ok());
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto bytes = Copy(*encoded);
    auto* root = wire::GetMutableFrameManifest(bytes.data());
    switch (mutation) {
      case 0:
        root->mutate_status(static_cast<wire::FrameStatus>(255));
        break;
      case 1:
        root->mutable_points()->mutate_kind(wire::ContentKind::State);
        break;
      case 2:
        root->mutable_points()->mutable_relative_name()->Mutate(0, '.');
        break;
      case 3:
        root->mutable_points()->mutate_byte_length(0);
        break;
    }
    EXPECT_EQ(DecodeLiquidFrameManifest3D(bytes).status().code(), absl::StatusCode::kDataLoss) << mutation;
  }
}
TEST(LiquidCache3D, CheckpointRejectsUnrepresentableOrInsufficientCapacityAndBadClock) {
  auto encoded = EncodeLiquidCheckpoint3D(PopulatedState(), Resume());
  ASSERT_TRUE(encoded.ok());
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto bytes = Copy(*encoded);
    auto* root = wire::GetMutableLiquidState(bytes.data());
    switch (mutation) {
      case 0:
        root->mutable_resume()->mutable_capacities()->mutate_particles(UINT64_MAX);
        break;
      case 1:
        root->mutable_resume()->mutable_capacities()->mutate_particles(0);
        break;
      case 2:
        root->mutable_resume()->mutable_replay_capacities()->mutate_particles(1);
        break;
      case 3:
        root->mutable_clock()->mutate_substep(1);
        break;
    }
    EXPECT_EQ(DecodeLiquidCheckpoint3D(bytes).status().code(), absl::StatusCode::kDataLoss) << mutation;
  }
}
TEST(LiquidCache3D, CheckpointEncoderRejectsInconsistentExecutionIdentity) {
  auto resume = Resume();
  resume.execution_id[0] ^= 1;
  EXPECT_EQ(EncodeLiquidCheckpoint3D(PopulatedState(), resume).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(LiquidCache3D, CheckpointReaderRejectsInconsistentExecutionIdentity) {
  auto bytes = EncodeLiquidCheckpoint3D(PopulatedState(), Resume());
  ASSERT_TRUE(bytes.ok());
  auto* root = wire::GetMutableLiquidState(bytes->values().data());
  auto* digest = root->mutable_resume()->mutable_execution_id()->mutable_bytes();
  digest->Mutate(0, static_cast<std::uint8_t>(digest->Get(0) ^ 1));
  EXPECT_EQ(DecodeLiquidCheckpoint3D(bytes->values()).status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(DecodeLiquidState3D(bytes->values()).status().code(), absl::StatusCode::kDataLoss);
}
TEST(LiquidCache3D, ManifestEncoderRejectsInconsistentExecutionIdentity) {
  LiquidFrameManifest3D manifest{
      {},
      Resume().resolved_fingerprint,
      Resume().execution_id,
      {LiquidContentKind3D::kPoints, 128, {}},
  };
  manifest.resolved_fingerprint[0] ^= 1;
  EXPECT_EQ(EncodeLiquidFrameManifest3D(manifest).status().code(), absl::StatusCode::kInvalidArgument);
}
TEST(LiquidCache3D, ManifestReaderRejectsInconsistentExecutionIdentity) {
  const LiquidFrameManifest3D manifest{
      {},
      Resume().resolved_fingerprint,
      Resume().execution_id,
      {LiquidContentKind3D::kPoints, 128, {}},
  };
  auto bytes = EncodeLiquidFrameManifest3D(manifest);
  ASSERT_TRUE(bytes.ok());
  auto* root = wire::GetMutableFrameManifest(bytes->values().data());
  auto* digest = root->mutable_resolved_fingerprint()->mutable_bytes();
  digest->Mutate(0, static_cast<std::uint8_t>(digest->Get(0) ^ 1));
  EXPECT_EQ(DecodeLiquidFrameManifest3D(bytes->values()).status().code(), absl::StatusCode::kDataLoss);
}
TEST(LiquidCache3D, InvalidBooleanBytesCannotNormalizeIntoValidState) {
  auto encoded = EncodeLiquidState3D(PopulatedState());
  ASSERT_TRUE(encoded.ok());
  auto bytes = Copy(*encoded);
  auto* root = wire::GetMutableLiquidState(bytes.data());
  auto* particle = reinterpret_cast<flatbuffers::Table*>(root->mutable_particles()->GetMutableObject(0));
  *particle->GetAddressOf(wire::Particle::VT_AFFINE_ENABLED) = 2;
  EXPECT_EQ(DecodeLiquidState3D(bytes).status().code(), absl::StatusCode::kDataLoss);
}
TEST(LiquidCache3D, ResolvedReaderRequiresMaterializedScalarDefaults) {
  auto encoded = EncodeLiquidResolvedScene3D({});
  ASSERT_TRUE(encoded.ok());
  auto bytes = Copy(*encoded);
  auto* material = wire::GetMutableResolvedScene(bytes.data())->mutable_material();
  auto* table = reinterpret_cast<std::uint8_t*>(material);
  auto* vtable = table - flatbuffers::ReadScalar<flatbuffers::soffset_t>(table);
  flatbuffers::WriteScalar<flatbuffers::voffset_t>(vtable + wire::Material::VT_SURFACE_TENSION, 0);
  EXPECT_EQ(DecodeLiquidResolvedScene3D(bytes).status().code(), absl::StatusCode::kDataLoss);
}
#if defined(_MSC_VER) && defined(_DEBUG)
std::size_t codec_allocations = 0;
std::size_t codec_frees = 0;
std::size_t fail_codec_allocation = 0;
int CodecAllocationHook(int kind, void*, std::size_t, int block, long, const unsigned char*, int) {
  if (block == _CRT_BLOCK) {
    return 1;
  }
  if (kind == _HOOK_ALLOC || kind == _HOOK_REALLOC) {
    ++codec_allocations;
    if (fail_codec_allocation == codec_allocations) {
      return 0;
    }
  }
  if (kind == _HOOK_FREE) {
    ++codec_frees;
  }
  return 1;
}
TEST(LiquidCache3D, InvalidSemanticInputAllocatesNoDecodedOwners) {
  auto encoded = EncodeLiquidState3D(PopulatedState());
  ASSERT_TRUE(encoded.ok());
  auto bytes = Copy(*encoded);
  wire::GetMutableLiquidState(bytes.data())->mutable_particles()->GetMutableObject(0)->mutate_volume_ratio(0);
  codec_allocations = 0;
  codec_frees = 0;
  const auto previous = _CrtSetAllocHook(CodecAllocationHook);
  absl::StatusCode code;
  {
    auto decoded = DecodeLiquidState3D(bytes);
    code = decoded.status().code();
  }
  _CrtSetAllocHook(previous);
  EXPECT_EQ(code, absl::StatusCode::kDataLoss);
  EXPECT_EQ(codec_allocations, 0);
  EXPECT_EQ(codec_frees, 0);
}
TEST(LiquidCache3D, ActualPartialAllocationFailureReturnsNoDecodedState) {
  auto state = PopulatedState();
  auto encoded = EncodeLiquidState3D(state);
  ASSERT_TRUE(encoded.ok());
  codec_allocations = 0;
  codec_frees = 0;
  fail_codec_allocation = 3;
  const auto previous = _CrtSetAllocHook(CodecAllocationHook);
  absl::StatusCode code;
  {
    auto decoded = DecodeLiquidState3D(encoded->values());
    code = decoded.status().code();
  }
  _CrtSetAllocHook(previous);
  fail_codec_allocation = 0;
  EXPECT_EQ(code, absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(codec_allocations, 3);
  EXPECT_EQ(codec_frees, 2);
  EXPECT_EQ(*DecodeLiquidState3D(encoded->values()), state);
}
#endif
}  // namespace
}  // namespace fluidity
