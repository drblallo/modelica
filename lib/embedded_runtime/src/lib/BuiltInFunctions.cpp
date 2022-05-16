#include "../../include/marco/lib/StdFunctions.h"
#include "../../include/marco/lib/BuiltInFunctions.h"
/*
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
*/
//===----------------------------------------------------------------------===//
// abs
//===----------------------------------------------------------------------===//

template<typename T>
inline T abs(T value)
{
	if (value >= 0) {
    return value;
  }

	return -1 * value;
}

inline bool abs_i1(bool value)
{
	return value;
}

inline int32_t abs_i32(int32_t value)
{
  return abs(value);
}

inline int64_t abs_i64(int64_t value)
{
  return abs(value);
}

inline float abs_f32(float value)
{
  return abs(value);
}

inline double abs_f64(double value)
{
  return abs(value);
}

RUNTIME_FUNC_DEF(abs, bool, bool)
RUNTIME_FUNC_DEF(abs, int32_t, int32_t)
RUNTIME_FUNC_DEF(abs, int64_t, int64_t)
RUNTIME_FUNC_DEF(abs, float, float)
RUNTIME_FUNC_DEF(abs, double, double)

//===----------------------------------------------------------------------===//
// acos
//===----------------------------------------------------------------------===//

inline float acos_f32(float value)
{
  return stde::acos(value);
}

inline double acos_f64(double value)
{
  return stde::acos(value);
}

RUNTIME_FUNC_DEF(acos, float, float)
RUNTIME_FUNC_DEF(acos, double, double)

//===----------------------------------------------------------------------===//
// asin
//===----------------------------------------------------------------------===//

inline float asin_f32(float value)
{
  return stde::asin(value);
}

inline double asin_f64(double value)
{
  return stde::asin(value);
}

RUNTIME_FUNC_DEF(asin, float, float)
RUNTIME_FUNC_DEF(asin, double, double)

//===----------------------------------------------------------------------===//
// atan
//===----------------------------------------------------------------------===//

inline float atan_f32(float value)
{
  return stde::atan(value);
}

inline double atan_f64(double value)
{
  return stde::atan(value);
}

RUNTIME_FUNC_DEF(atan, float, float)
RUNTIME_FUNC_DEF(atan, double, double)

//===----------------------------------------------------------------------===//
// atan2
//===----------------------------------------------------------------------===//

inline float atan2_f32(float y, float x)
{
  return stde::atan2(y, x);
}

inline double atan2_f64(double y, double x)
{
  return stde::atan2(y, x);
}

RUNTIME_FUNC_DEF(atan2, float, float, float)
RUNTIME_FUNC_DEF(atan2, double, double, double)

//===----------------------------------------------------------------------===//
// cos
//===----------------------------------------------------------------------===//

inline float cos_f32(float value)
{
  return stde::cos(value);
}

inline double cos_f64(double value)
{
  return stde::cos(value);
}

RUNTIME_FUNC_DEF(cos, float, float)
RUNTIME_FUNC_DEF(cos, double, double)

//===----------------------------------------------------------------------===//
// cosh
//===----------------------------------------------------------------------===//

inline float cosh_f32(float value)
{
  return stde::cosh(value);
}

inline double cosh_f64(double value)
{
  return stde::cosh(value);
}

RUNTIME_FUNC_DEF(cosh, float, float)
RUNTIME_FUNC_DEF(cosh, double, double)

//===----------------------------------------------------------------------===//
// diagonal
//===----------------------------------------------------------------------===//

/// Place some values on the diagonal of a matrix, and set all the other
/// elements to zero.
///
/// @tparam T 					destination matrix type
/// @tparam U 					source values type
/// @param destination destination matrix
/// @param values 			source values
template<typename T, typename U>
inline void diagonal_void(UnsizedArrayDescriptor<T>* destination, UnsizedArrayDescriptor<U>* values)
{
	// Check that the array is square-like (all the dimensions have the same
	// size). Note that the implementation is generalized to n-D dimensions,
	// while the "identity" Modelica function is defined only for 2-D arrays.
	// Still, the implementation complexity would be the same.

	stde::assertt(destination->hasSameSizes());

	// Check that the sizes of the matrix dimensions match with the amount of
	// values to be set.

	stde::assertt(destination->getRank() > 0);
	stde::assertt(values->getRank() == 1);
	stde::assertt(destination->getDimensionSize(0) == values->getDimensionSize(0));

	// Directly use the iterators, as we need to determine the current indexes
	// so that we can place a 1 if the access is on the matrix diagonal.

	for (auto it = destination->begin(), end = destination->end(); it != end; ++it) {
		auto indexes = it.getCurrentIndexes();
		stde::assertt(!indexes.empty());

		bool isIdentityAccess = stde::all_of(indexes.begin(), indexes.end(), [&indexes](const auto& i) {
			return i == indexes[0];
		});

		*it = isIdentityAccess ? values->get(indexes[0]) : 0;
	}
}

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(diagonal, void, ARRAY(double), ARRAY(double))


//===----------------------------------------------------------------------===//
// exp
//===----------------------------------------------------------------------===//

inline float exp_f32(float value)
{
  return stde::exp(value);
}

inline double exp_f64(double value)
{
	return stde::exp(value);
}

RUNTIME_FUNC_DEF(exp, float, float)
RUNTIME_FUNC_DEF(exp, double, double)

//===----------------------------------------------------------------------===//
// fill
//===----------------------------------------------------------------------===//

/// Set all the elements of an array to a given value.
///
/// @tparam T 		 data type
/// @param array  array to be populated
/// @param value  value to be set
template<typename T>
inline void fill_void(UnsizedArrayDescriptor<T>* array, T value)
{
	for (auto& element : *array) {
    element = value;
  }
}

RUNTIME_FUNC_DEF(fill, void, ARRAY(bool), bool)
RUNTIME_FUNC_DEF(fill, void, ARRAY(int32_t), int32_t)
RUNTIME_FUNC_DEF(fill, void, ARRAY(int64_t), int64_t)
RUNTIME_FUNC_DEF(fill, void, ARRAY(float), float)
RUNTIME_FUNC_DEF(fill, void, ARRAY(double), double)

//===----------------------------------------------------------------------===//
// identity
//===----------------------------------------------------------------------===//

/// Set a multi-dimensional array to an identity like matrix.
///
/// @tparam T 	   data type
/// @param array  array to be populated
template<typename T>
inline void identity_void(UnsizedArrayDescriptor<T>* array)
{
	// Check that the array is square-like (all the dimensions have the same
	// size). Note that the implementation is generalized to n-D dimensions,
	// while the "identity" Modelica function is defined only for 2-D arrays.
	// Still, the implementation complexity would be the same.

	stde::assertt(array->hasSameSizes());

	// Directly use the iterators, as we need to determine the current indexes
	// so that we can place a 1 if the access is on the matrix diagonal.

	for (auto it = array->begin(), end = array->end(); it != end; ++it) {
		auto indexes = it.getCurrentIndexes();
		stde::assertt(!indexes.empty());

		bool isIdentityAccess = stde::all_of(indexes.begin(), indexes.end(), [&indexes](const auto& i) {
			return i == indexes[0];
		});

		*it = isIdentityAccess ? 1 : 0;
	}
}

RUNTIME_FUNC_DEF(identity, void, ARRAY(bool))
RUNTIME_FUNC_DEF(identity, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(identity, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(identity, void, ARRAY(float))
RUNTIME_FUNC_DEF(identity, void, ARRAY(double))

//===----------------------------------------------------------------------===//
// linspace
//===----------------------------------------------------------------------===//

/// Populate a 1-D array with equally spaced elements.
///
/// @tparam T 		 data type
/// @param array  array to be populated
/// @param start  start value
/// @param end 	 end value
template<typename T>
inline void linspace_void(UnsizedArrayDescriptor<T>* array, double start, double end)
{
	using dimension_t = typename UnsizedArrayDescriptor<T>::dimension_t;
	stde::assertt(array->getRank() == 1);

	auto n = array->getDimensionSize(0);
	double step = (end - start) / ((double) n - 1);

	for (dimension_t i = 0; i < n; ++i) {
    array->get(i) = start + static_cast<double>(i) * step;
  }
}

RUNTIME_FUNC_DEF(linspace, void, ARRAY(bool), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(bool), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int32_t), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int32_t), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int64_t), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(int64_t), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(float), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(float), double, double)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(double), float, float)
RUNTIME_FUNC_DEF(linspace, void, ARRAY(double), double, double)

//===----------------------------------------------------------------------===//
// ln
//===----------------------------------------------------------------------===//

inline float log_f32(float value)
{
	stde::assertt(value > 0);
	return stde::log(value);
}

inline double log_f64(double value)
{
  stde::assertt(value > 0);
  return stde::log(value);
}

RUNTIME_FUNC_DEF(log, float, float)
RUNTIME_FUNC_DEF(log, double, double)

//===----------------------------------------------------------------------===//
// log
//===----------------------------------------------------------------------===//

inline float log10_f32(float value)
{
  stde::assertt(value > 0);
  return stde::log10(value);
}

inline double log10_f64(double value)
{
	stde::assertt(value > 0);
	return stde::log10(value);
}

RUNTIME_FUNC_DEF(log10, float, float)
RUNTIME_FUNC_DEF(log10, double, double)

//===----------------------------------------------------------------------===//
// max
//===----------------------------------------------------------------------===//

template<typename T>
inline T max(UnsizedArrayDescriptor<T>* array)
{
	return *stde::max_element(array->begin(), array->end());
}

inline bool max_i1(UnsizedArrayDescriptor<bool>* array)
{
  return stde::any_of(array->begin(), array->end(), [](const bool& value) {
    return value;
  });
}

inline int32_t max_i32(UnsizedArrayDescriptor<int32_t>* array)
{
  return max(array);
}

inline int32_t max_i64(UnsizedArrayDescriptor<int64_t>* array)
{
  return max(array);
}

inline float max_f32(UnsizedArrayDescriptor<float>* array)
{
  return max(array);
}

inline double max_f64(UnsizedArrayDescriptor<double>* array)
{
  return max(array);
}

RUNTIME_FUNC_DEF(max, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(max, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(max, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(max, float, ARRAY(float))
RUNTIME_FUNC_DEF(max, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// max
//===----------------------------------------------------------------------===//

template<typename T>
inline T max(T x, T y)
{
	return stde::max(x, y);
}

inline bool max_i1(bool x, bool y)
{
  return x || y;
}

inline int32_t max_i32(int32_t x, int32_t y)
{
  return max(x, y);
}

inline int64_t max_i64(int64_t x, int64_t y)
{
  return max(x, y);
}

inline float max_f32(float x, float y)
{
  return max(x, y);
}

inline double max_f64(double x, double y)
{
  return max(x, y);
}

RUNTIME_FUNC_DEF(max, bool, bool, bool)
RUNTIME_FUNC_DEF(max, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(max, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(max, float, float, float)
RUNTIME_FUNC_DEF(max, double, double, double)

//===----------------------------------------------------------------------===//
// min
//===----------------------------------------------------------------------===//

template<typename T>
inline T min(UnsizedArrayDescriptor<T>* array)
{
	return *stde::min_element(array->begin(), array->end());
}

inline bool min_i1(UnsizedArrayDescriptor<bool>* array)
{
  return stde::all_of(array->begin(), array->end(), [](const bool& value) {
    return value;
  });
}

inline int32_t min_i32(UnsizedArrayDescriptor<int32_t>* array)
{
  return min(array);
}

inline int32_t min_i64(UnsizedArrayDescriptor<int64_t>* array)
{
  return min(array);
}

inline float min_f32(UnsizedArrayDescriptor<float>* array)
{
  return min(array);
}

inline double min_f64(UnsizedArrayDescriptor<double>* array)
{
  return min(array);
}

RUNTIME_FUNC_DEF(min, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(min, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(min, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(min, float, ARRAY(float))
RUNTIME_FUNC_DEF(min, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// min
//===----------------------------------------------------------------------===//

template<typename T>
inline T min(T x, T y)
{
	return stde::min(x, y);
}

inline bool min_i1(bool x, bool y)
{
  return x && y;
}

inline int32_t min_i32(int32_t x, int32_t y)
{
  return min(x, y);
}

inline int64_t min_i64(int64_t x, int64_t y)
{
  return min(x, y);
}

inline float min_f32(float x, float y)
{
  return min(x, y);
}

inline double min_f64(double x, double y)
{
  return min(x, y);
}

RUNTIME_FUNC_DEF(min, bool, bool, bool)
RUNTIME_FUNC_DEF(min, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(min, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(min, float, float, float)
RUNTIME_FUNC_DEF(min, double, double, double)

//===----------------------------------------------------------------------===//
// ones
//===----------------------------------------------------------------------===//

/// Set all the elements of an array to ones.
///
/// @tparam T data type
/// @param array   array to be populated
template<typename T>
inline void ones_void(UnsizedArrayDescriptor<T> *array)
{
	for (auto& element :* array) {
    element = 1;
  }
}

RUNTIME_FUNC_DEF(ones, void, ARRAY(bool))
RUNTIME_FUNC_DEF(ones, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(ones, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(ones, void, ARRAY(float))
RUNTIME_FUNC_DEF(ones, void, ARRAY(double))

//===----------------------------------------------------------------------===//
// product
//===----------------------------------------------------------------------===//

/// Multiply all the elements of an array.
///
/// @tparam T 		 data type
/// @param array  array
/// @return product of all the values
template<typename T>
inline T product(UnsizedArrayDescriptor<T>* array)
{
	return stde::accumulate(array->begin(), array->end(), 1, stde::multiplies<T>());
}

inline bool product_i1(UnsizedArrayDescriptor<bool> *array)
{
  return stde::all_of(array->begin(), array->end(), [](const bool& value) {
    return value;
  });
}

inline int32_t product_i32(UnsizedArrayDescriptor<int32_t> *array)
{
  return product(array);
}

inline int64_t product_i64(UnsizedArrayDescriptor<int64_t> *array)
{
  return product(array);
}

inline float product_f32(UnsizedArrayDescriptor<float> *array)
{
  return product(array);
}

inline double product_f64(UnsizedArrayDescriptor<double> *array)
{
  return product(array);
}

RUNTIME_FUNC_DEF(product, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(product, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(product, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(product, float, ARRAY(float))
RUNTIME_FUNC_DEF(product, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// sign
//===----------------------------------------------------------------------===//

/// Get the sign of a value.
///
/// @tparam T		 data type
/// @param value  value
/// @return 1 if value is > 0, -1 if < 0, 0 if = 0
template<typename T>
inline long sign(T value)
{
	if (value == 0) {
    return 0;
  }

	if (value > 0) {
    return 1;
  }

	return -1;
}

template<typename T>
inline int32_t sign_i32(T value)
{
  return sign(value);
}

template<>
inline int32_t sign_i32(bool value)
{
  return value ? 1 : 0;
}

template<typename T>
inline int64_t sign_i64(T value)
{
  return sign(value);
}

template<>
inline int64_t sign_i64(bool value)
{
  return value ? 1 : 0;
}

RUNTIME_FUNC_DEF(sign, int32_t, bool)
RUNTIME_FUNC_DEF(sign, int32_t, int32_t)
RUNTIME_FUNC_DEF(sign, int32_t, int64_t)
RUNTIME_FUNC_DEF(sign, int32_t, float)
RUNTIME_FUNC_DEF(sign, int32_t, double)

RUNTIME_FUNC_DEF(sign, int64_t, bool)
RUNTIME_FUNC_DEF(sign, int64_t, int32_t)
RUNTIME_FUNC_DEF(sign, int64_t, int64_t)
RUNTIME_FUNC_DEF(sign, int64_t, float)
RUNTIME_FUNC_DEF(sign, int64_t, double)

//===----------------------------------------------------------------------===//
// sin
//===----------------------------------------------------------------------===//

inline float sin_f32(float value)
{
	return stde::sin(value);
}

inline double sin_f64(double value)
{
  return stde::sin(value);
}

RUNTIME_FUNC_DEF(sin, float, float)
RUNTIME_FUNC_DEF(sin, double, double)

//===----------------------------------------------------------------------===//
// sinh
//===----------------------------------------------------------------------===//

inline float sinh_f32(float value)
{
  return stde::sinh(value);
}

inline double sinh_f64(double value)
{
  return stde::sinh(value);
}

RUNTIME_FUNC_DEF(sinh, float, float)
RUNTIME_FUNC_DEF(sinh, double, double)

//===----------------------------------------------------------------------===//
// sqrt
//===----------------------------------------------------------------------===//

inline float sqrt_f32(float value)
{
  stde::assertt(value >= 0);
  return stde::sqrt(value);
}

inline double sqrt_f64(double value)
{
	stde::assertt(value >= 0);
	return stde::sqrt(value);
}

RUNTIME_FUNC_DEF(sqrt, float, float)
RUNTIME_FUNC_DEF(sqrt, double, double)

//===----------------------------------------------------------------------===//
// sum
//===----------------------------------------------------------------------===//

/// Sum all the elements of an array.
///
/// @tparam T 		 data type
/// @param array  array
/// @return sum of all the values
template<typename T>
inline T sum(UnsizedArrayDescriptor<T>* array)
{
	return stde::accumulate(array->begin(), array->end(), 0, stde::plus<T>());
}

inline bool sum_i1(UnsizedArrayDescriptor<bool>* array)
{
  return stde::any_of(array->begin(), array->end(), [](const bool& value) {
    return value;
  });
}

inline int32_t sum_i32(UnsizedArrayDescriptor<int32_t>* array)
{
  return sum(array);
}

inline int64_t sum_i64(UnsizedArrayDescriptor<int64_t>* array)
{
  return sum(array);
}

inline float sum_f32(UnsizedArrayDescriptor<float>* array)
{
  return sum(array);
}

inline double sum_f64(UnsizedArrayDescriptor<double>* array)
{
  return sum(array);
}

RUNTIME_FUNC_DEF(sum, bool, ARRAY(bool))
RUNTIME_FUNC_DEF(sum, int32_t, ARRAY(int32_t))
RUNTIME_FUNC_DEF(sum, int64_t, ARRAY(int64_t))
RUNTIME_FUNC_DEF(sum, float, ARRAY(float))
RUNTIME_FUNC_DEF(sum, double, ARRAY(double))

//===----------------------------------------------------------------------===//
// symmetric
//===----------------------------------------------------------------------===//

/// Populate the destination matrix so that it becomes the symmetric version
/// of the source one, thus discarding the elements below the source diagonal.
///
/// @tparam Destination	destination type
/// @tparam Source				source type
/// @param destination		destination matrix
/// @param source				source matrix
template<typename Destination, typename Source>
void symmetric_void(UnsizedArrayDescriptor<Destination>* destination, UnsizedArrayDescriptor<Source>* source)
{
	using dimension_t = typename UnsizedArrayDescriptor<Destination>::dimension_t;

	// The two arrays must have exactly two dimensions
	stde::assertt(destination->getRank() == 2);
	stde::assertt(source->getRank() == 2);

	// The two matrixes must have the same dimensions
	stde::assertt(destination->getDimensionSize(0) == source->getDimensionSize(0));
	stde::assertt(destination->getDimensionSize(1) == source->getDimensionSize(1));

	auto size = destination->getDimensionSize(0);

	// Manually iterate on the dimensions, so that we can explore just half
	// of the source matrix.

	for (dimension_t i = 0; i < size; ++i) {
		for (dimension_t j = i; j < size; ++j) {
			destination->set({ i, j }, source->get({ i, j }));

			if (i != j) {
        destination->set({j, i}, source->get({ i, j }));
      }
		}
	}
}

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(symmetric, void, ARRAY(double), ARRAY(double))

//===----------------------------------------------------------------------===//
// tan
//===----------------------------------------------------------------------===//

inline float tan_f32(float value)
{
  return stde::tan(value);
}

inline double tan_f64(double value)
{
	return stde::tan(value);
}

RUNTIME_FUNC_DEF(tan, float, float)
RUNTIME_FUNC_DEF(tan, double, double)

//===----------------------------------------------------------------------===//
// tanh
//===----------------------------------------------------------------------===//

inline float tanh_f32(float value)
{
  return stde::tanh(value);
}

inline double tanh_f64(double value)
{
	return stde::tanh(value);
}

RUNTIME_FUNC_DEF(tanh, float, float)
RUNTIME_FUNC_DEF(tanh, double, double)

//===----------------------------------------------------------------------===//
// transpose
//===----------------------------------------------------------------------===//

/// Transpose a matrix.
///
/// @tparam Destination destination type
/// @tparam Source 		 source type
/// @param destination  destination matrix
/// @param source  		 source matrix
template<typename Destination, typename Source>
void transpose_void(UnsizedArrayDescriptor<Destination>* destination, UnsizedArrayDescriptor<Source>* source)
{
	using dimension_t = typename UnsizedArrayDescriptor<Source>::dimension_t;

	// The two arrays must have exactly two dimensions
	stde::assertt(destination->getRank() == 2);
	stde::assertt(source->getRank() == 2);

	// The two matrixes must have transposed dimensions
	stde::assertt(destination->getDimensionSize(0) == source->getDimensionSize(1));
	stde::assertt(destination->getDimensionSize(1) == source->getDimensionSize(0));

	// Directly use the iterators, as we need to determine the current
	// indexes and transpose them to access the other matrix.

	for (auto it = source->begin(), end = source->end(); it != end; ++it) {
		auto indexes = it.getCurrentIndexes();
		stde::assertt(indexes.size() == 2);

		stde::Vector<dimension_t> transposedIndexes;

		//for (auto revIt = indexes.rbegin(), revEnd = indexes.rend(); revIt != revEnd; ++revIt) {
		for (auto revIt = indexes.end(), revEnd = indexes.begin(); revIt != revEnd; --revIt) {
      		transposedIndexes.push_back(*revIt);
    	}

		destination->set(transposedIndexes, *it);
	}
}

RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(transpose, void, ARRAY(double), ARRAY(double))

//===----------------------------------------------------------------------===//
// zeros
//===----------------------------------------------------------------------===//

/// Set all the elements of an array to zero.
///
/// @tparam T data type
/// @param array   array to be populated
template<typename T>
inline void zeros_void(UnsizedArrayDescriptor<T>* array)
{
	for (auto& element : *array) {
    element = 0;
  }
}

RUNTIME_FUNC_DEF(zeros, void, ARRAY(bool))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(float))
RUNTIME_FUNC_DEF(zeros, void, ARRAY(double))
