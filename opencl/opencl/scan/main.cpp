#include <iostream>
#include <fstream>
#include <vector>
constexpr const char* INPUT_FILE = "input.txt";
constexpr const char* UUTPUT_FILE = "output.txt";

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

void inclusive_scan(std::vector<float> const& input, std::vector<float>& output) {
  output.resize(input.size());
  size_t n = input.size();
  if (n == 0) {
    return;
  }

  output[0] = input[0];
  for (int i = 1; i < n; ++i) {
    output[i] = output[i - 1] + input[i];
  }
}

void exclusive_scan(std::vector<float> const& input, std::vector<float>& output) {
  inclusive_scan(input, output);
  output.pop_back();
  output.insert(output.begin(), 0.f);
}

void print_vector(std::ostream& out, std::vector<float> const& data) {
  size_t n = data.size();
  for (int i = 0; i < n; ++i) {
    out << data[i] << " ";
  }

  out << std::endl;
}

int main(void) {
  std::vector<float> in;
  read_numbers(INPUT_FILE, in);
  std::vector<float> inclusive;
  std::vector<float> exclusive;

  inclusive_scan(in, inclusive);
  exclusive_scan(in, exclusive);

  print_vector(std::cout, inclusive);
  print_vector(std::cout, exclusive);
}
