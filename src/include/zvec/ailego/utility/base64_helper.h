// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <string>
#include <zvec/export.h>

namespace zvec {
namespace ailego {

/*! Base64 Helper
 *
 *  Minimal base64 encode/decode for safe binary-to-text transport,
 *  e.g. embedding binary blobs into JSON or Params strings.
 */
struct ZVEC_AILEGO_API Base64Helper {
  //! Encode a binary buffer into a base64 string
  static std::string Encode(const void *data, size_t len);

  //! Decode a base64 string into the original binary data.
  //! Characters outside the base64 alphabet are silently skipped.
  static std::string Decode(const std::string &in);
};

}  // namespace ailego
}  // namespace zvec
