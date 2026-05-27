// file: utils.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cassert>
//#include <cstdint>
//#include <type_traits>
//#include <limits>
//#include <cmath>
//#include <stdexcept>
//#include <chrono>

#ifndef ITERATIONS
#define ITERATIONS 1000
#endif // default iterations for correctness test

#define Small_Arg_1D rng.randint(size_t(1), size_t(512))
#define Small_Arg_2D rng.randint(size_t(1), size_t(32))
#define Small_Arg_3D rng.randint(size_t(1), size_t(10))
#define Small_Arg_4D rng.randint(size_t(1), size_t(8))
#define Small_Arg_5D rng.randint(size_t(1), size_t(6))

#define Medium_Args_1D Arg(1<<6)->Arg(1<<7)->Arg(1<<8)->Arg(1<<9)->Arg(1<<10)->Arg(1<<11)->Arg(1<<12)
#define Medium_Args_3D Args({1<<2, 1<<2, 1<<2})->Args({1<<2, 1<<2, 1<<3})->Args({1<<1, 1<<5, 1<<1})->Args({1<<5, 1<<1, 1<<1})->Args({1, 1, 1<<5})->Args({1<<4, 1<<4, 1<<2})

#define Large_Args_1D Arg(1<<10)->Arg(1<<12)->Arg(1<<14)->Arg(1<<16)->Arg(1<<18)->Arg(1<<20)
#define Large_Args_2D Args({1<<5, 1<<5})->Args({1<<6, 1<<6})->Args({1<<4, 1<<10})->Args({1<<16, 1})->Args({1, 1<<18})->Args({1<<10, 1<<10})
#define Large_Args_3D Args({1<<2, 1<<4, 1<<4})->Args({1<<3, 1<<3, 1<<6})->Args({1<<1, 1<<12, 1<<1})->Args({1<<14, 1<<1, 1<<1})->Args({1, 1, 1<<18})->Args({1<<6, 1<<7, 1<<7})
#define Large_Args_4D \
    Args({1<<2, 1<<2, 1<<2, 1<<4})-> \
    Args({1<<1, 1<<1, 1<<9, 1<<1})-> \
    Args({1<<1, 1<<11, 1<<1, 1<<1})-> \
    Args({1<<13, 1<<1, 1<<1, 1<<1})-> \
    Args({1<<1, 1<<5, 1<<4, 1<<3})-> \
    Args({1<<4, 1<<4, 1<<4, 1<<4})

#define Large_Args_5D \
    Args({1<<1, 1<<2, 1<<2, 1<<2, 1<<1})-> \
    Args({1<<2, 1<<1, 1<<3, 1<<1, 1<<1})-> \
    Args({1<<1, 1<<1, 1<<1, 1<<4, 1<<1})-> \
    Args({1<<5, 1<<1, 1<<1, 1<<1, 1<<1})-> \
    Args({1<<1, 1<<1, 1<<1, 1<<1, 1<<6})-> \
    Args({1<<3, 1<<3, 1<<3, 1<<3, 1<<3})
    

#ifndef DEFAULT_SEED
#define DEFAULT_SEED 0xdeadbeef
#endif // default seed for random number generator

#ifndef ERROR_PRINT
#define ERROR_PRINT 0
#endif // default error print flag

class Random {
    private:
        std::mt19937 rng;
    
    public:
        Random() {
            // a true random value
            std::random_device rd;
            rng = std::mt19937(rd());
        }

        Random(uint32_t seed){
            rng = std::mt19937(seed); // given seed
        }

        Random(std::mt19937 _rng_){
            rng = _rng_;
        }
    
        // Returns a random floating-point N such that 0.0 <= N < 1.0
        // Equivalent to random.random()
        template<typename T>
        T random() {
            assert(std::is_floating_point<T>::value);
            std::uniform_real_distribution<T> dist(0.0, 1.0);
            return dist(rng);
        }

        // Returns a random boolean value
        bool random_bool(){
            std::uniform_int_distribution<> dist(0, 1);
            return dist(rng) == 1;
        }
    
        // Returns a random integer N such that a <= N <= b
        // work with any integral type (int8_t, int16_t, int32_t, uint8_t, uint16_t, \
           uint32_t, int64_t, uint64_t, etc.)
        template<typename T>
        T randint(T a = std::numeric_limits<T>::min(), T b = std::numeric_limits<T>::max()) {
            static_assert(std::is_integral<T>::value, "T must be an integral type.");
            if(a==b){ return a; }
            if(a>b){ std::swap(a, b); }
            std::uniform_int_distribution<T> dist(a, b);
            return dist(rng);
        }
    
        // Returns a random floating-point N such that a <= N <= b,
        // work with any floating-point type (float, double, long double)
        template<typename T>
        T uniform(T a = T(-100.0), T b = T(100.0)) {
            static_assert(std::is_floating_point<T>::value, "T must be a floating-point type.");
            if (a == b){ return a; }
            if (a > b){ std::swap(a, b); }
            if (b > std::numeric_limits<T>::max()) { b = std::numeric_limits<T>::max(); }
            if (a < - std::numeric_limits<T>::max()) { a = - std::numeric_limits<T>::max(); }
            std::uniform_real_distribution<T> dist(a, b);
            T result = dist(rng);
            if (std::isinf(result) || std::isnan(result)) { result = 0; }
            return result;
        }

        // normal distribution
        template<typename T>
        T normal_distribution(T mean = 0.0, T deviation = 1000.0) {
            std::normal_distribution<T> dist(mean, deviation);
            T result = dist(rng);
            // if (std::isinf(result) || std::isnan(result)) { result = 0; }
            return result;
        }


        // `if constexpr’ only available with ‘-std=c++17’ or ‘-std=gnu++17’
        // initializes a vector with random values
        // if bool_item is true, initializes with random boolean values
        template <typename T>
        void initialize_vector_with_random_values(std::vector<T>& vec, bool bool_item = false) {
            static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value || std::is_same<T, bool>::value, 
                "T must be an integral, floating-point, or bool type.");
            if (bool_item){
                for (size_t i = 0; i < vec.size(); ++i) {
                    vec[i] = random_bool();
                }
            }else if constexpr (std::is_same<T, bool>::value) {
                for (size_t i = 0; i < vec.size(); ++i) {
                    vec[i] = random_bool();
                }
            }else if constexpr (std::is_integral<T>::value) {
                for (auto& element : vec) {
                    element = randint<T>();
                }
            } else if constexpr (std::is_floating_point<T>::value) {
                for (auto& element : vec) {
                    element = uniform<T>();
                }
            } else {
                throw std::invalid_argument("Unsupported type for random initialization");
            }
        }

        // initialize_vector_with_permutation
        void initialize_vector_with_permutation(std::vector<int>& vec) {
            std::iota(vec.begin(), vec.end(), 1);
            std::shuffle(vec.begin(), vec.end(), rng);
        }
    
        // Returns a random element from the non-empty vector
        // Equivalent to random.choice
        template<typename T>
        T choice(const std::vector<T>& vec) {
            if(vec.empty()) {
                throw std::invalid_argument("Cannot choose from an empty vector");
            }
            // Use uniform_int_distribution to get a random index
            std::uniform_int_distribution<> dist(0, vec.size() - 1);
            return vec[dist(rng)];
        }
    
        // Shuffles the elements of the vector in place
        // Equivalent to random.shuffle
        template<typename T>
        void shuffle(std::vector<T>& vec) {
            std::shuffle(vec.begin(), vec.end(), rng);
        }

        // generate a random string from a given charset
        std::string random_string( const std::string& charset, size_t length ){
            std::uniform_int_distribution<> dist(0, charset.size() - 1);
                std::string result;
                result.reserve(length);
                for (size_t i = 0; i < length; ++i) { result += charset[dist(rng)]; }
                return result;
        }
};

const std::string CHARSET_LOWERCASE = "abcdefghijklmnopqrstuvwxyz";
const std::string CHARSET_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const std::string CHARSET_DIGITS = "0123456789";
const std::string CHARSET_ALPHA = CHARSET_LOWERCASE + CHARSET_UPPERCASE;
const std::string CHARSET_ALPHANUMERIC = CHARSET_ALPHA + CHARSET_DIGITS;
const std::string CHARSET_HEX = "0123456789ABCDEF";
const std::string CHARSET_PRINTABLE = 
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

// Returns True if two arrays are element-wise equal within a tolerance.
// The tolerance is defined as the maximum of the absolute difference and the relative difference.
// Note: equal_nan is defaulted to true, meaning that NaNs are considered equal.
template<typename T>
bool allclose(const std::vector<T>& a, const std::vector<T>& b, \
    double rtol = 1e-5, double atol = 1e-8, bool equal_nan = true) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        T x = a[i], y = b[i];
        if constexpr (std::is_floating_point<T>::value){
            // floating points
            if (std::isnan(x) || std::isnan(y)) {
                if (!equal_nan || !(std::isnan(x) && std::isnan(y))) return false;
                continue;
            }
            // Handle infinities
            if ((std::isinf(x) && !std::isinf(y)) || (!std::isinf(x) && std::isinf(y))) {
                return false; // Must match sign and magnitude
            }
            if (std::abs(x - y) > (atol + rtol * std::abs(y))) {
                return false;
            }
        } else if constexpr (std::is_integral<T>::value) {
            // integers and bools            
            if (x != y) return false;
        } else if constexpr (std::is_same<T, std::string>::value) {
            // strings
            if (x != y) return false;
        } else {
            static_assert(std::is_arithmetic<T>::value, "Unsupported type");
        }
    }
    return true;
}

// Overload for single values
template<typename T>
bool allclose(const T& a, const T& b, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = true) {
    T x = a, y = b;
    if constexpr (std::is_floating_point<T>::value){
        // floating points
        if (std::isnan(x) || std::isnan(y)) {
            if (!equal_nan || !(std::isnan(x) && std::isnan(y))) return false;
        }
        if (std::abs(x - y) > (atol + rtol * std::abs(y))) {
            return false;
        }
    }else if constexpr (std::is_integral<T>::value) {
        // integers and bools            
        return x == y;
    } else if constexpr (std::is_same<T, std::string>::value){
        // strings
        return x == y;
    } else {
        static_assert(std::is_arithmetic<T>::value, "Unsupported type");
    }
    return true;
}

// Deprecated: use google benchmark instead
// Benchmark function with multiple iterations (without Google Benchmark library)
// Reutrns the average time taken for the function to execute (in nanoseconds)
/*
template <typename Func, typename... Args>
long long benchmark_average_time(Func func, int iterations, Args&&... args) {
    iterations = std::max(100, iterations);  // Ensure at least 100 iteration
    std::vector<long long> times;
    // Warm-up phase (not timed)
    int warmup = iterations * 0.2;
    for (int i = 0; i < warmup; ++i) {
        std::forward<Func>(func)(std::forward<Args>(args)...);
    }
    for (int i = 0; i < iterations; ++i) {
        auto start_t = std::chrono::high_resolution_clock::now();
        std::forward<Func>(func)(std::forward<Args>(args)...);
        auto end_t = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count());
    }
    //long long total = std::accumulate(times.begin(), times.end(), 0LL);
    //return total / iterations;  // average in nanoseconds
    
    // Discard top and bottom 10%
    std::sort(times.begin(), times.end());
    int discard = iterations * 0.1;
    auto begin = times.begin() + discard;
    auto end = times.end() - discard;
    
    long long total = std::accumulate(begin, end, 0LL);
    return total / std::distance(begin, end);  // average in nanoseconds
}
*/

// for debugging
template<typename T>
void MyPrint(const std::vector<T>& vec) {
    for (const auto& element : vec) {
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value){
            std::cout << static_cast<int>(element) << " ";
        } else {
            std::cout << element << " ";
        }
    }
    std::cout << std::endl;
}

#endif // UTILS_HPP