#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.h>
#include <chrono>
#include <CL/cl.hpp>
#include <algorithm>
#include <cassert>

static constexpr const char* INPUT_FILE = "input.txt";
static constexpr const char* OPENCL_PROGRAM_FILE = "scan.cl";
static constexpr const char* UUTPUT_FILE = "output.txt";
static constexpr size_t WORK_GROUP_SIZE = 256;

void read_numbers(const char* input_file_name, std::vector<float>& numbers) {
  std::ifstream input(input_file_name);
  size_t n;
  input >> n;
  numbers.clear();
  numbers.resize(n);
  for (int i = 0; i < n; ++i) {
    input >> numbers[i];
  }
}

std::pair<uint64_t, std::vector<float>> inclusive_scan_sequential(std::vector<float> const& input) {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> output(input.size());
  size_t n = input.size();

  if (n != 0) {
    output[0] = input[0];
  }

  for (int i = 1; i < n; ++i) {
    output[i] = output[i - 1] + input[i];
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start);
  return {elapsed.count(), output};
}

void print_vector(std::ostream& out, std::vector<float> const& data) {
  size_t n = data.size();
  for (int i = 0; i < n; ++i) {
    out << data[i] << " ";
  }

  out << std::endl;
}

// contract: fit_to_workgroup_size(size) >= size
size_t fit_to_workgroup_size(size_t size) {
  return size % WORK_GROUP_SIZE == 0 ? size : size + (WORK_GROUP_SIZE - size % WORK_GROUP_SIZE);
}

std::vector<float> copy_and_resize(std::vector<float> const& base, size_t size) {
  auto result = base;
  result.resize(size);
  return result;
}

std::pair<uint64_t, std::vector<float>> inclusive_scan_parallel(std::vector<float> const& input) {
  auto start = std::chrono::high_resolution_clock::now();
  static const auto error = make_pair(-1, std::vector<float>{});
  size_t old_size = input.size();
  size_t evaluation_size = fit_to_workgroup_size(old_size);
  std::vector<float> in = old_size == evaluation_size ? input : copy_and_resize(input, evaluation_size);

  std::vector<float> output(evaluation_size);

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
    for (cl::Device dev : devices) {
      std::string name = dev.getInfo<CL_DEVICE_NAME>();
      std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cerr << "Build log for " << name << ":" << std::endl
          << buildlog << std::endl;
    }

    cl_program.close();
    return error;
  }

  size_t bound_values_count = evaluation_size / WORK_GROUP_SIZE;
  cl::Buffer dev_input(context, CL_MEM_READ_ONLY, evaluation_size * sizeof(float));
  cl::Buffer dev_bound(context, CL_MEM_READ_ONLY, bound_values_count * sizeof(float));
  cl::Buffer dev_bound_sum(context, CL_MEM_READ_ONLY, bound_values_count * sizeof(float));

  cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, evaluation_size * sizeof(float));

  queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, evaluation_size * sizeof(float), &in[0]);

  queue.finish();

  size_t localSize = std::min(evaluation_size, WORK_GROUP_SIZE) * sizeof(float);

  cl::Kernel scan_kernel(program, "local_scan");
  scan_kernel.setArg(0, dev_input);
  scan_kernel.setArg(1, dev_output);
  scan_kernel.setArg(2, dev_bound);
  scan_kernel.setArg(3, cl::Local(localSize));
  scan_kernel.setArg(4, cl::Local(localSize));

  cl::Kernel add_bounds_kernel(program, "add_lefter_bounds");
  add_bounds_kernel.setArg(0, dev_output);
  add_bounds_kernel.setArg(1, dev_bound);

  auto res = queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(evaluation_size), cl::NDRange(WORK_GROUP_SIZE));
  assert(res == CL_SUCCESS);

  res = queue.enqueueNDRangeKernel(add_bounds_kernel, cl::NullRange, cl::NDRange(evaluation_size), cl::NDRange(WORK_GROUP_SIZE));
  assert(res == CL_SUCCESS);

  cl_int evaluation_result = queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * evaluation_size, &output[0]);

  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start);
  if (evaluation_result != CL_SUCCESS) {
    std::cerr << "Something went wrong. Code = " << evaluation_result << std::endl;
    return error;
  }

  cl_program.close();
  output.resize(old_size);
  return {elapsed.count(), output};
}

bool check_equals(std::vector<float> const& left, std::vector<float> const& right) {
  const static float DELTA = 10.;
  if (left.size() != left.size()) {
    return false;
  }

  auto size = left.size();
  for (size_t i = 0; i < size; ++i) {
    if (fabs(left[i] - right[i]) >= DELTA) {
      return false;
    }
  }

  return true;
}

std::vector<float> generate_test(size_t n) {
  std::vector<float> result(n);
  for (size_t i = 0; i < n; ++i) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    result[i] = r;
  }

  return result;
}

int main(void) {
  std::vector<float> in = generate_test(1000000);

  auto seq_result = inclusive_scan_sequential(in);
  auto parallel_result = inclusive_scan_parallel(in);

  std::cout << "Sequential time = " << seq_result.first << "ns" << std::endl;
  std::cout << "Parallel time = " << parallel_result.first << "ns" << std::endl;
  std::cout << "acceleration: " << seq_result.first / parallel_result.first << std::endl;

  if (seq_result.first == -1 || parallel_result.first == -1) {
    std::cerr << "Evaluation failed" << std::endl;
    return -1;
  }

  assert(check_equals(seq_result.second, parallel_result.second));
}
