#include "common.h"
#include <algorithm>

matrix allocate(size_t n) {
  matrix m;
  auto matrix = new float*[n];
  float* buf = new float[n * n];
  std::fill_n(buf, n * n, 0);
  for (size_t i = 0; i < n; ++i) {
    matrix[i] = buf + i * n;
  }

  m.size = n;
  m.elems = matrix;
  return m;
}
