// RUN: marco --omc-bypass --model=MultidimensionalArray --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x[1,1]","x[1,2]","x[1,3]","x[2,1]","x[2,2]","x[2,3]"
// CHECK-NEXT: 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
// CHECK-NEXT: 0.100000,0.100000,0.200000,0.200000,0.100000,0.200000,0.200000
// CHECK-NEXT: 0.200000,0.200000,0.400000,0.400000,0.200000,0.400000,0.400000
// CHECK-NEXT: 0.300000,0.300000,0.600000,0.600000,0.300000,0.600000,0.600000
// CHECK-NEXT: 0.400000,0.400000,0.800000,0.800000,0.400000,0.800000,0.800000
// CHECK-NEXT: 0.500000,0.500000,1.000000,1.000000,0.500000,1.000000,1.000000
// CHECK-NEXT: 0.600000,0.600000,1.200000,1.200000,0.600000,1.200000,1.200000
// CHECK-NEXT: 0.700000,0.700000,1.400000,1.400000,0.700000,1.400000,1.400000
// CHECK-NEXT: 0.800000,0.800000,1.600000,1.600000,0.800000,1.600000,1.600000
// CHECK-NEXT: 0.900000,0.900000,1.800000,1.800000,0.900000,1.800000,1.800000
// CHECK-NEXT: 1.000000,1.000000,2.000000,2.000000,1.000000,2.000000,2.000000

model MultidimensionalArray
	Real[2, 3] x(each start = 0, fixed = true);
equation
	for i in 1:2 loop
		der(x[i, 1]) = 1.0;

		for j in 2:3 loop
			der(x[i, j]) = 2.0;
		end for;
	end for;
end MultidimensionalArray;
