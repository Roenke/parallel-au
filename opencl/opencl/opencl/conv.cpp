#include "conv.h"
#include <chrono>
#include <ostream>
#include <CL/cl.h>
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>

static constexpr const char* OPENCL_PROGRAM_FILE = "conv.cl";

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
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  std::vector<cl::Device> devices;
  for (cl::Platform platform : platforms) {
    std::vector<cl::Device> gpu_devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &gpu_devices);
    for (cl::Device gpu : gpu_devices) {
      devices.push_back(gpu);
    }
  }

  if (devices.size() == 0) {
    std::cerr << "no available devices" << std::endl;
    return error;
  }

  cl::Device dev = devices[0];

  cl::Context context(devices);

  cl::CommandQueue queue(context, devices.front(), CL_QUEUE_PROFILING_ENABLE);

  std::ifstream cl_program(OPENCL_PROGRAM_FILE);

  auto it = std::istreambuf_iterator<char>(cl_program);
  auto end = std::istreambuf_iterator<char>();
  std::string code(it, end);

  cl::Program::Sources sourceCode(1, std::make_pair(code.c_str(), code.length() + 1));
  cl::Program program(context, sourceCode);
  cl_int compication_result = program.build(devices);

  if (compication_result != 0) {
    std::string name = devices[0].getInfo<CL_DEVICE_NAME>();
    std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
    std::cerr << "Build log for " << name << ":" << std::endl
      << buildlog << std::endl;
    return error;
  }

  matrix c = allocate(a.size);

  size_t a_buf_size = sizeof(float) * a.size * a.size;
  size_t b_buf_size = sizeof(float) * b.size * b.size;
  size_t c_buf_size = sizeof(float) * c.size * c.size;

  cl::Buffer dev_in_a(context, CL_MEM_READ_ONLY, a_buf_size);
  cl::Buffer dev_in_b(context, CL_MEM_READ_ONLY, b_buf_size);
  cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, c_buf_size);

  queue.enqueueWriteBuffer(dev_in_a, CL_TRUE, 0, a_buf_size, &a.elems[0][0]);
  queue.enqueueWriteBuffer(dev_in_b, CL_TRUE, 0, b_buf_size, &b.elems[0][0]);
  queue.enqueueWriteBuffer(dev_output, CL_TRUE, 0, a_buf_size, &c.elems[0][0]);

  queue.finish();

  cl::Kernel kernel(program, "eval");
  kernel.setArg(0, dev_in_a);
  kernel.setArg(1, dev_in_b);
  kernel.setArg(2, dev_output);
  kernel.setArg(3, static_cast<int>(a.size));
  kernel.setArg(4, static_cast<int>(b.size));

  queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(a.size * a.size));

  cl_int evaluation_result = queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, c_buf_size, &c.elems[0][0]);
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start);
  if (evaluation_result != CL_SUCCESS) {
    std::cerr << "Something went wrong. Code = " << evaluation_result << std::endl;
    return error;
  }

  cl_program.close();
  return std::make_pair(c, elapsed.count());
}
