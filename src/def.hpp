#pragma once

#include <cassert>

#if !defined(NDEBUG)
#define PK_DEBUG
#endif

#define PK_ASSERT(x) assert(x)
