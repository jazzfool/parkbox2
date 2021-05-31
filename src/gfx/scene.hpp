#pragma once

#include "indirect.hpp"

namespace gfx {

class Scene final {
  public:
    IndirectStorage storage;
    IndirectMeshPass pass;
};

} // namespace gfx
