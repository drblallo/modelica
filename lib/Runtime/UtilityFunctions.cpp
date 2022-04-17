#include "marco/Runtime/UtilityFunctions.h"
#include <cassert>
#include <cstring>
#include <iostream>

//===----------------------------------------------------------------------===//
// clone
//===----------------------------------------------------------------------===//

namespace
{
  /// Clone an array into another one.
  ///
  /// @tparam T 					destination array type
  /// @tparam U 					source array type
  /// @param destination  destination array
  /// @param values 			source values
  template<typename T, typename U>
  void clone_void(UnsizedArrayDescriptor<T> destination, UnsizedArrayDescriptor<U> source)
  {
    assert(source.getNumElements() == destination.getNumElements());

    auto sourceIt = source.begin();
    auto destinationIt = destination.begin();

    for (size_t i = 0, e = source.getNumElements(); i < e; ++i) {
      *destinationIt = *sourceIt;

      ++sourceIt;
      ++destinationIt;
    }
  }

  // Optimization for arrays with the same type
  template<typename T>
  void clone_void(UnsizedArrayDescriptor<T> destination, UnsizedArrayDescriptor<T> source)
  {
    auto destinationSize = destination.getNumElements();
    assert(source.getNumElements() == destinationSize);
    memcpy(destination.getData(), source.getData(), destinationSize * sizeof(T));
  }
}

RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(double))
