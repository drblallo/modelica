// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mod
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 10
// CHECK-NEXT: 20

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 1>* x);

using namespace std;

int main() {
	array<long, 2> x = { 0, 0 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	__modelica_ciface_foo(&xDescriptor);

	cout << "results" << endl;
	cout << xDescriptor[0] << endl;
	cout << xDescriptor[1] << endl;

	return 0;
}
