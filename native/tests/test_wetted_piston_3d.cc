#include "wetted_piston_3d_cases.h"

namespace fluidity {
namespace {
TEST(WettedPiston3D, ClipsHorizontalPlaneAndKeepsBothFluxSigns) {
  wetted_piston_test::PlaneClippingAndSignedFlux(ComputeWettedPiston3D, 0);
}
TEST(WettedPiston3D, EnclosesThinOverlappingCapAndAttachesOnce) {
  wetted_piston_test::OverlappingThinCapEnclosesAnalyticArea(ComputeWettedPiston3D, 0);
}
TEST(WettedPiston3D, CertifiesTangencyAndRejectsUncertainPositiveArea) {
  wetted_piston_test::TangencyAndUncertifiableThinWall(ComputeWettedPiston3D, 0);
}
TEST(WettedPiston3D, RejectsDisconnectedAndUnownedWetRows) {
  wetted_piston_test::RefusesUnsupportedOwnershipAndKeepsPublication(ComputeWettedPiston3D);
}
TEST(WettedPiston3D, UsesSignedHalfOpenPistonCellAndChecksQueryCapacity) {
  wetted_piston_test::PreservesSignedLatticeAndChecksCapacity(ComputeWettedPiston3D, 0);
}
}  // namespace
}  // namespace fluidity
