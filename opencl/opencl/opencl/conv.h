#pragma once
#include "common.h"
#include <utility>

auto constexpr error = std::make_pair(matrix{}, 0);

std::pair<matrix, size_t> sequential_evaluation(matrix const& a, matrix const& b);
std::pair<matrix, size_t> parallel_evaluation(matrix const& a, matrix const& b);
