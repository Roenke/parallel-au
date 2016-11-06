#include <utility>
#include <vector>
#include <iostream>
#include "common.h"
#include "conv.h"

static const float FLOAT_DELTA = 1e-3;

matrix generate_filled_matrix(int n, float value = 1.f) {
  matrix m = allocate(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      m.elems[i][j] = value;
    }
  }

  return m;
}

bool check_eq(matrix const& a, matrix const& b) {
  if(a.size != b.size) {
    return false;
  }

  for(int i = 0; i < a.size; ++i) {
    for(int j = 0; j < a.size; ++j) {
      if (abs(a.elems[i][j] - b.elems[i][j]) >= FLOAT_DELTA) {
        return false;
      }
    }
  }

  return true;
}

bool run_tests(std::vector<std::pair<size_t, size_t>> test_descriptions) {
  size_t failed_count = 0;

  for (auto desc : test_descriptions) {
    std::cout << "N = " << desc.first << ", M = " << desc.second << ". ";
    matrix a = generate_filled_matrix(desc.first);
    matrix b = generate_filled_matrix(desc.second);

    auto res_seq = sequential_evaluation(a, b);
    auto res_par = parallel_evaluation(a, b);

    if (res_seq.second == 0 || res_par.second == 0 || !check_eq(res_seq.first, res_par.first)) {
      ++failed_count;
      std::cout << "failed";
    }
    else {
      std::cout << "passed";
    }

    std::cout << " (seq_time = " << res_seq.second << " ns, " << "par_time = " << res_par.second << " ns, speedup = " << static_cast<float>(res_seq.second) / res_par.second << std::endl;
  }

  std::cout << "Test summary: " << (test_descriptions.size() - failed_count) << " / " << test_descriptions.size() << " passed" << std::endl;

  return failed_count == 0;
}
