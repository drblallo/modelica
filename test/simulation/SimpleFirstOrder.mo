// RUN: marco --omc-bypass --model=SimpleFirstOrder --end-time=1 -o simulation %s
// RUN: ./simulation | FileCheck %s

// CHECK: time;x
// CHECK-NEXT: 0.000000000000;0.000000000000
// CHECK-NEXT: 0.100000000000;0.000000000000
// CHECK-NEXT: 0.200000000000;0.100000000000
// CHECK-NEXT: 0.300000000000;0.190000000000
// CHECK-NEXT: 0.400000000000;0.271000000000
// CHECK-NEXT: 0.500000000000;0.343900000000
// CHECK-NEXT: 0.600000000000;0.409510000000
// CHECK-NEXT: 0.700000000000;0.468559000000
// CHECK-NEXT: 0.800000000000;0.521703100000
// CHECK-NEXT: 0.900000000000;0.569532790000
// CHECK-NEXT: 1.000000000000;0.612579511000

model SimpleFirstOrder
    Real x;
equation
    der(x) = 1 - x;
end SimpleFirstOrder;
