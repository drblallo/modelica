// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[0, 1], [2, 3], [4, 5]]

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" long __modelica_ciface_foo(
		ArrayDescriptor<long, 1>* x, ArrayDescriptor<long, 1>* y, ArrayDescriptor<long, 1>* z, ArrayDescriptor<long, 2>* t);

using namespace std;

int main() {
	array<long, 2> x = { 0, 1 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	array<long, 2> y = { 2, 3 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	array<long, 2> z = { 4, 5 };
	ArrayDescriptor<long, 1> zDescriptor(z);

	array<long, 6> t = { 0, 0, 0, 0, 0, 0 };
	ArrayDescriptor<long, 2> tDescriptor(t.data(), { 3, 2 });

	cout << "results" << endl;
	__modelica_ciface_foo(&xDescriptor, &yDescriptor, &zDescriptor, &tDescriptor);
	cout << tDescriptor << endl;

	return 0;
}
