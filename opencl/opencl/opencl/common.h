#pragma once
struct matrix {
  float** elems;
  size_t size;
};

matrix allocate(size_t n);
