// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 1.000000e+00
// CHECK-NEXT: 8.660254e-01
// CHECK-NEXT: 7.071068e-01

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" double __modelica_ciface_foo(double x);

using namespace std;

int main() {
	array<double, 3> x = { 0, M_PI / 6, M_PI / 4 };
	cout << "results" << endl;

	for (const auto& value : x)
		cout << scientific << __modelica_ciface_foo(value) << endl;

	return 0;
}
