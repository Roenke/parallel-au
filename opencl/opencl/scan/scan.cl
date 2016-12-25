#define SWAP(a,b) {__local float* tmp=a; a=b; b=tmp;}

__kernel void scan(__global float* input, 
                   __global float* output, 
                   __global float* bound_values, 
                   __local float* a, 
                   __local float* b)
{
  uint gid = get_group_id(0);
  uint lid = get_local_id(0);
  uint block_size = get_local_size(0);

  uint ix = lid + gid * block_size;

  a[lid] = b[lid] = input[ix];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint s = 1; s < block_size; s *= 2) {
    if (lid > (s - 1)) {
      b[lid] = a[lid] + a[lid - s];
    }
    else {
      b[lid] = a[lid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    SWAP(a, b);
  }

  output[ix] = a[lid];
  if (lid == block_size - 1) {
    bound_values[gid] = a[lid];
  }
}

__kernel 
void add_bounds(__global float* output, __global float* bound_values)
{
  uint lid = get_local_id(0);
  uint gid = get_group_id(0);
  uint block_size = get_local_size(0);
  
  if (gid > 0) {
    uint ix = lid + gid * block_size;
    output[ix] += bound_values[gid - 1];
  }
}
