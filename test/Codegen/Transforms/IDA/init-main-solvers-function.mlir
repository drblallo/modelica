// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       simulation.function @initMainSolvers() {
// CHECK:           ida.create @ida_main
// CHECK:           simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}

// -----

// Check variables.

// CHECK: ida.instance @ida_main
// CHECK:       simulation.function @initMainSolvers() {
// CHECK:           %[[state_ida:.*]] = ida.add_state_variable @ida_main {derivativeGetter = @{{.*}}, derivativeSetter = @{{.*}}, dimensions = [1], stateGetter = @{{.*}}, stateSetter = @{{.*}}}
// CHECK:           simulation.return
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<!modelica.real>
    modelica.simulation_variable @der_x : !modelica.variable<!modelica.real>

    modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
        modelica.variable @x : !modelica.variable<!modelica.real>
        modelica.variable @der_x : !modelica.variable<!modelica.real>
    }
}
