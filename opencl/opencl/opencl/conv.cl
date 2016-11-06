void kernel eval(global const float * a, global const float * b,
  global float * const c, const int n, const int m)
{
  int idx = get_global_id(0);
  int i = idx / n;
  int j = idx % n;
  int hm = (m - 1) / 2;
  float res = 0.f;

  int upper_k = min(hm + 1, n - i);
  for (int k = max(-i, -hm); k < upper_k; ++k) {
    int upper_l = min(hm + 1, n - j);
    for (int l = max(-j, -hm); l < upper_l; ++l) {
      res += a[(i + k) * n + (j + l)] * b[(k + hm) * m + (l + hm)];
    }
  }

  c[idx] = res;
}
