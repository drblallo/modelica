// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --rel-tol=1e-10 --abs-tol=1e-10 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -o %t
// RUN: %t | FileCheck %s

// x[1:3] = time
// y[1:3] = 3 * time + time ^ 2 / 2

// CHECK: time,x[1],x[2],x[3],y[1],y[2],y[3]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000
// CHECK-NEXT: 5.000000000000,5.000000000000,5.000000000000,5.000000000000,27.500000000000,27.500000000000,27.500000000000
// CHECK-NEXT: 10.000000000000,10.000000000000,10.000000000000,10.000000000000,80.000000000000,80.000000000000,80.000000000000
