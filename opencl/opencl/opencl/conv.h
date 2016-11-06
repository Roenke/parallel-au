#pragma once
#include "common.h"
#include <utility>

std::pair<matrix, size_t> sequential_evaluation(matrix const& a, matrix const& b);
std::pair<matrix, size_t> parallel_evaluation(matrix const& a, matrix const& b);
