// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:       func.func @icModelEnd() {
// CHECK-NEXT:      call @foo() : () -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @foo() {
    func.return
}

simulation.ic_model_end {
    func.call @foo() : () -> ()
}

// -----

// CHECK:       func.func @icModelEnd() {
// CHECK-NEXT:      call @foo() : () -> ()
// CHECK-NEXT:      call @bar() : () -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @foo() {
    func.return
}

func.func @bar() {
    func.return
}

simulation.ic_model_end {
    func.call @foo() : () -> ()
}

simulation.ic_model_end {
    func.call @bar() : () -> ()
}
