// Minimal glog shim — Sputnik only uses CHECK-style macros from glog. This avoids a
// hard dependency on a specific glog version (the conda glog 0.7 is ABI/source-
// incompatible with Sputnik's expectations).
#pragma once
#include <cstdlib>
#include <iostream>

namespace glogshim {
struct Fatal {
  ~Fatal() {
    std::cerr << std::endl;
    std::abort();
  }
  template <class T>
  Fatal& operator<<(const T& v) {
    std::cerr << v;
    return *this;
  }
};
}  // namespace glogshim

#define CHECK(cond) for (bool _gls = static_cast<bool>(cond); !_gls; _gls = true) ::glogshim::Fatal()
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define DCHECK(cond) CHECK(cond)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)

namespace google {
inline void InitGoogleLogging(const char*) {}
}  // namespace google
