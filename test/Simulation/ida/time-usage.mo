// RUN: marco %s --omc-bypass --model=TimeUsage --variable-filter="time;x;der(x)" --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x"
// CHECK: 0.0000,0.0000,0.0000
// CHECK: 1.0000,0.5000,1.0000

model TimeUsage
	Real x(start = 0, fixed = true);
equation
	der(x) = time;
end TimeUsage;
