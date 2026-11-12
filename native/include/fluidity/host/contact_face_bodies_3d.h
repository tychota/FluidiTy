#pragma once

#include "absl/status/status.h"
#include "fluidity/base/contact_face_bodies_3d.h"

namespace fluidity {

// Converts a face deposit result to the checked host status. A fifth body
// maps to ResourceExhausted and E-CONTACT-BODY-CAPACITY.
inline absl::Status ContactFaceDepositStatus(ContactFaceDepositStatus3D result) {
  switch (result) {
    case ContactFaceDepositStatus3D::kAccepted:
      return absl::OkStatus();
    case ContactFaceDepositStatus3D::kInvalid:
      return absl::InvalidArgumentError("invalid contact face contribution");
    case ContactFaceDepositStatus3D::kRange:
      return absl::OutOfRangeError("unrepresentable contact face body state");
    case ContactFaceDepositStatus3D::kBodyCapacity:
      return absl::ResourceExhaustedError("E-CONTACT-BODY-CAPACITY");
  }
  return absl::InternalError("unknown contact face deposit result");
}

}  // namespace fluidity
