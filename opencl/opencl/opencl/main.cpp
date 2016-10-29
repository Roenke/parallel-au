#include<stdio.h>
#include<CL/cl.hpp>
#include <vector>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <chrono>

float FLOAT_DELTA = 1e-5f;

struct matrix {
  float** elems;
  size_t size;
};

matrix allocate(size_t n) {
  matrix m;
  auto matrix = new float*[n];
  for (size_t i = 0; i < n; ++i) {
    matrix[i] = new float[n];
    std::fill_n(matrix[i], n, 0);
  }

  m.size = n;
  m.elems = matrix;
  return m;
}

void read_square_matrix(std::istream& is, matrix& m) {
  for (size_t i = 0; i < m.size; ++i) {
    for (size_t j = 0; j < m.size; ++j) {
      is >> m.elems[i][j];
    }
  }
}

std::pair<matrix, matrix> read_matrix(std::string const& filename) {
  std::ifstream is(filename);
  size_t n, m;
  is >> n >> m;
  assert(m % 2 == 1 && "m must be an odd number");
  matrix a = allocate(n);
  matrix b = allocate(m);

  read_square_matrix(is, a);
  read_square_matrix(is, b);

  return std::make_pair(a, b);
}

void assert_equals(matrix const& left, matrix const& right) {
  assert(left.size == right.size);
  for (size_t i = 0; i < left.size; ++i) {
    for (size_t j = 0; j < left.size; ++j) {
      assert(abs(left.elems[i][j] - right.elems[i][j]) < FLOAT_DELTA);
    }
  }
}

std::pair<matrix, size_t> sequential_evaluation(matrix const& a, matrix const& b) {
  auto start = std::chrono::high_resolution_clock::now();
  int hm = (b.size - 1) / 2;
  auto res = allocate(a.size);
  for (int i = 0; i < a.size; ++i) {
    for (int j = 0; j < a.size; ++j) {
      for (int k = -hm; k <= hm; ++k) {
        int a_i = i + k;
        if (a_i < 0 || a_i >= a.size) {
          continue;
        }
        int b_i = k + hm;
        for (int l = -hm; l <= hm; ++l) {
          int a_j = j + l;
          
          if (a_j < 0 || a_j >= a.size) {
            continue;
          }

          int b_j = l + hm;
          res.elems[i][j] += a.elems[a_i][a_j] * b.elems[b_i][b_j];
        }
      }
    }
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start);

  return std::make_pair(res, elapsed.count());
}

std::pair<matrix, size_t> parallel_evaluation(matrix const& a, matrix const& b) {
  return std::make_pair(a, 1);
}

void print_matrix(matrix const& m, std::ostream& os) {
  for (size_t i = 0; i < m.size; ++i) {
    for (size_t j = 0; j < m.size; ++j) {
      os << " " << m.elems[i][j];
    }

    os << std::endl;
  }
}

int main(void) {
  auto p = read_matrix("test2.txt");
  matrix a = p.first;
  matrix b = p.second;

  std::pair<matrix, size_t> seq_res = sequential_evaluation(a, b);
  std::cout << "time in sequential evaluation = " << seq_res.second << " ns" << std::endl;;
  std::pair<matrix, size_t> par_res = parallel_evaluation(a, b);
  std::cout << "time in parallel evaluation = " << par_res.second << " ns" << std::endl;
  std::cout << "acceleration = " << static_cast<double>(seq_res.second) / par_res.second << std::endl;

  print_matrix(seq_res.first, std::cout);
  assert_equals(seq_res.first, par_res.first);
  std::cout << "parallel and sequential algorithms produce same matrices" << std::endl;
}
