#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ----------------------------------------------------------------------------
// random utils
// ----------------------------------------------------------------------------

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

inline std::vector<float> make_random_f32(size_t n, float lo = -1.0f,
                                          float hi = 1.0f) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

inline std::vector<int32_t> make_random_i32(size_t n, int lo = 1, int hi = 4) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(lo, hi);
  std::vector<int32_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

inline bool arg_is_flag(const char *arg) {
  return arg != nullptr && arg[0] == '-' && arg[1] == '-';
}

inline bool has_flag(int argc, char **argv, const char *flag) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == flag) {
      return true;
    }
  }
  return false;
}

inline std::vector<std::string> collect_positionals(int argc, char **argv) {
  std::vector<std::string> values;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--verify" || arg == "--bench") {
      continue;
    }
    if (arg == "--iters" || arg == "--cpu-iters") {
      if (i + 1 < argc) {
        ++i;
      }
      continue;
    }
    if (arg == "--") {
      for (++i; i < argc; ++i) {
        values.emplace_back(argv[i]);
      }
      break;
    }
    if (arg_is_flag(arg.c_str())) {
      continue;
    }
    values.emplace_back(std::move(arg));
  }
  return values;
}

template <typename T>
inline T parse_numeric(const std::string &text) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(std::stoll(text));
  } else {
    return static_cast<T>(std::stod(text));
  }
}

template <typename T>
inline T positional_arg(const std::vector<std::string> &args, size_t index,
                        T default_value) {
  if (index >= args.size()) {
    return default_value;
  }
  return parse_numeric<T>(args[index]);
}

inline int flag_value_or(int argc, char **argv, const char *flag, int default_value) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == flag && i + 1 < argc) {
      return std::stoi(argv[i + 1]);
    }
  }
  return default_value;
}

struct TestRunOptions {
  bool verify = true;
  bool bench = false;
  int iters = 100;
  int cpu_iters = 10;
};

inline TestRunOptions parse_run_options(int argc, char **argv) {
  TestRunOptions opts;
  opts.verify = !has_flag(argc, argv, "--bench") || has_flag(argc, argv, "--verify");
  opts.bench = has_flag(argc, argv, "--bench");
  opts.iters = std::max(1, flag_value_or(argc, argv, "--iters", opts.iters));
  opts.cpu_iters = std::max(1, flag_value_or(argc, argv, "--cpu-iters", opts.cpu_iters));
  return opts;
}

template <typename Fn>
inline double benchmark_host_ms(int repeats, Fn &&fn) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; ++i) {
    fn();
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

template <typename T>
inline std::string preview_vector(const std::vector<T> &values, size_t limit = 5) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(std::is_integral_v<T> ? 0 : 6);
  for (size_t i = 0; i < std::min(limit, values.size()); ++i) {
    if (i) {
      oss << ", ";
    }
    oss << values[i];
  }
  return oss.str();
}

template <typename T>
inline void validate_close(const std::vector<T> &actual,
                           const std::vector<T> &expected, const char *name,
                           double atol = 1e-4, double rtol = 1e-4) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(name) + ": size mismatch");
  }

  size_t worst_index = 0;
  double worst_diff = 0.0;
  double worst_limit = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double a = static_cast<double>(actual[i]);
    double b = static_cast<double>(expected[i]);
    double diff = std::fabs(a - b);
    double limit = std::is_integral_v<T> ? 0.0 : (atol + rtol * std::fabs(b));
    if constexpr (std::is_integral_v<T>) {
      if (actual[i] != expected[i]) {
        std::ostringstream oss;
        oss << name << " mismatch at " << i << ": got " << actual[i]
            << ", expected " << expected[i];
        throw std::runtime_error(oss.str());
      }
    } else {
      if (diff > limit) {
        std::ostringstream oss;
        oss << name << " mismatch at " << i << ": got " << actual[i]
            << ", expected " << expected[i] << ", diff=" << diff
            << ", limit=" << limit;
        throw std::runtime_error(oss.str());
      }
      if (diff > worst_diff) {
        worst_diff = diff;
        worst_limit = limit;
        worst_index = i;
      }
    }
  }

  if constexpr (!std::is_integral_v<T>) {
    std::cout << name << " max_abs_diff=" << worst_diff
              << " at " << worst_index << " limit=" << worst_limit << "\n";
  }
}

inline void relu_ref(const std::vector<float> &x, std::vector<float> &y) {
  y.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    y[i] = std::max(0.0f, x[i]);
  }
}

inline void gelu_ref(const std::vector<float> &x, std::vector<float> &y) {
  constexpr float kMinExp = -88.3762626647949f;
  constexpr float kMaxExp = 88.3762626647949f;
  constexpr float kScale = 0.7978845608028654f;
  y.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    float v = std::min(std::max(x[i], kMinExp), kMaxExp);
    y[i] = 0.5f * v * (1.0f + std::tanh(kScale * (v + 0.044715f * v * v * v)));
  }
}

inline void rmsnorm_ref(const std::vector<float> &x, std::vector<float> &y, int N,
                        int K) {
  y.resize(x.size());
  constexpr float kEps = 1e-5f;
  for (int row = 0; row < N; ++row) {
    const float *row_ptr = x.data() + static_cast<size_t>(row) * K;
    float sum_sq = 0.0f;
    for (int col = 0; col < K; ++col) {
      sum_sq += row_ptr[col] * row_ptr[col];
    }
    float scale = 1.0f / std::sqrt(sum_sq / K + kEps);
    float *out_ptr = y.data() + static_cast<size_t>(row) * K;
    for (int col = 0; col < K; ++col) {
      out_ptr[col] = row_ptr[col] * scale;
    }
  }
}

inline void sgemv_ref(const std::vector<float> &a, const std::vector<float> &x,
                      std::vector<float> &y, int M, int K) {
  y.assign(M, 0.0f);
  for (int row = 0; row < M; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < K; ++col) {
      sum += a[static_cast<size_t>(row) * K + col] * x[col];
    }
    y[row] = sum;
  }
}

inline void softmax_ref(const std::vector<float> &x, std::vector<float> &y, int S,
                        int H) {
  y.resize(x.size());
  for (int s = 0; s < S; ++s) {
    const float *row = x.data() + static_cast<size_t>(s) * H;
    float *out = y.data() + static_cast<size_t>(s) * H;
    float max_v = row[0];
    for (int i = 1; i < H; ++i) {
      max_v = std::max(max_v, row[i]);
    }
    float denom = 0.0f;
    for (int i = 0; i < H; ++i) {
      out[i] = std::exp(row[i] - max_v);
      denom += out[i];
    }
    for (int i = 0; i < H; ++i) {
      out[i] /= denom;
    }
  }
}

inline void cumsum_ref(const std::vector<int32_t> &input,
                       std::vector<int32_t> &output) {
  output.resize(input.size());
  int64_t acc = 0;
  for (size_t i = 0; i < input.size(); ++i) {
    acc += input[i];
    output[i] = static_cast<int32_t>(acc);
  }
}

inline void sgemm_ref(const std::vector<float> &a, const std::vector<float> &b,
                      std::vector<float> &c, int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f) {
  std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a[static_cast<size_t>(row) * K + k] * b[static_cast<size_t>(k) * N + col];
      }
      const float c_prev =
          (c.size() == out.size()) ? c[static_cast<size_t>(row) * N + col] : 0.0f;
      out[static_cast<size_t>(row) * N + col] =
          alpha * sum + beta * c_prev;
    }
  }
  c.swap(out);
}

inline void print_head(const char *tag, const std::vector<float> &v, int k = 4) {
  std::cout << tag << ": ";
  for (int i = 0; i < k && i < static_cast<int>(v.size()); ++i) {
    std::cout << v[i] << (i + 1 == k ? "" : ", ");
  }
  std::cout << "\n";
}

inline void print_head(const char *tag, const std::vector<int32_t> &v, int k = 8) {
  std::cout << tag << ": ";
  for (int i = 0; i < k && i < static_cast<int>(v.size()); ++i) {
    std::cout << v[i] << (i + 1 == k ? "" : ", ");
  }
  std::cout << "\n";
}

#endif // _TEST_UTILS_H_
