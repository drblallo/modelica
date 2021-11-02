// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -o %t
// RUN: %t | FileCheck %s

// der(x[1:3, 1:4]) = 2 * der(x[2, 2]) - 4

// CHECK: time,x[1,1],x[1,2],x[1,3],x[1,4],x[2,1],x[2,2],x[2,3],x[2,4],x[3,1],x[3,2],x[3,3],x[3,4]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,
// CHECK-SAME: 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000

// CHECK-NEXT: 5.000000000000,20.000000000000,20.000000000000,20.000000000000,20.000000000000,20.000000000000,
// CHECK-SAME: 20.000000000000,20.000000000000,20.000000000000,20.000000000000,20.000000000000,20.000000000000,20.000000000000

// CHECK-NEXT: 10.000000000000,40.000000000000,40.000000000000,40.000000000000,40.000000000000,40.000000000000,40.000000000000,
// CHECK-SAME: 40.000000000000,40.000000000000,40.000000000000,40.000000000000,40.000000000000,40.000000000000
