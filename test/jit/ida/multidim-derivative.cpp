// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// x[1:2, 1] = time
// x[1:2, 2:3] = 2 * time

// CHECK: time,x[1,1],x[1,2],x[1,3],x[2,1],x[2,2],x[2,3]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000
// CHECK-NEXT: 5.000000000000,5.000000000000,10.000000000000,10.000000000000,5.000000000000,10.000000000000,10.000000000000
// CHECK-NEXT: 10.000000000000,10.000000000000,20.000000000000,20.000000000000,10.000000000000,20.000000000000,20.000000000000
