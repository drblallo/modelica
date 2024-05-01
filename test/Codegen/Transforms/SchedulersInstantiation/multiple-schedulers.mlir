// RUN: modelica-opt %s --split-input-file --schedulers-instantiation | FileCheck %s

// CHECK-DAG:   #[[index_set:.*]] = #modeling<index_set {[0,9]}>
// CHECK-DAG:   #[[range:.*]] = #modeling<multidim_range [0,9]>

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.main_model {
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block {
// CHECK-NEXT:                  runtime.scheduler_run @[[scheduler_0:.*]]
// CHECK-NEXT:              } {readVariables = [], writtenVariables = [#bmodelica.var<@x, #[[index_set]]>]}
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block {
// CHECK-NEXT:                  runtime.scheduler_run @[[scheduler_1:.*]]
// CHECK-NEXT:              } {readVariables = [], writtenVariables = [#bmodelica.var<@y, #[[index_set]]>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       runtime.scheduler @[[scheduler_0]]

// CHECK:       runtime.dynamic_model_begin {
// CHECK-NEXT:      runtime.scheduler_create @[[scheduler_0]]
// CHECK-NEXT:      runtime.scheduler_add_equation @[[scheduler_0]] {function = @[[equation_0_wrapper:.*]], ranges = #[[range]]}
// CHECK-NEXT:  }

// CHECK:       runtime.equation_function @[[equation_0_wrapper]](%[[i0_lb:.*]]: index, %[[i0_ub:.*]]: index) {
// CHECK-NEXT:      bmodelica.call @equation_0(%[[i0_lb]], %[[i0_ub]])
// CHECK-NEXT:      runtime.return
// CHECK-NEXT:  }

// CHECK:       runtime.dynamic_model_end {
// CHECK-NEXT:      runtime.scheduler_destroy @[[scheduler_0]]
// CHECK-NEXT:  }

// CHECK:       runtime.scheduler @[[scheduler_1]]

// CHECK:       runtime.dynamic_model_begin {
// CHECK-NEXT:      runtime.scheduler_create @[[scheduler_1]]
// CHECK-NEXT:      runtime.scheduler_add_equation @[[scheduler_1]] {function = @[[equation_1_wrapper:.*]], ranges = #[[range]]}
// CHECK-NEXT:  }

// CHECK:       runtime.equation_function @[[equation_1_wrapper]](%[[i0_lb:.*]]: index, %[[i0_ub:.*]]: index) {
// CHECK-NEXT:      bmodelica.call @equation_1(%[[i0_lb]], %[[i0_ub]])
// CHECK-NEXT:      runtime.return
// CHECK-NEXT:  }

// CHECK:       runtime.dynamic_model_end {
// CHECK-NEXT:      runtime.scheduler_destroy @[[scheduler_1]]
// CHECK-NEXT:  }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

        bmodelica.schedule @schedule {
            bmodelica.main_model {
                bmodelica.parallel_schedule_blocks {
                    bmodelica.schedule_block {
                        bmodelica.equation_call @equation_0 {indices = #modeling<multidim_range [0,9]>}
                    } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@x, #modeling<index_set {[0,9]}>>]}
                }
                bmodelica.parallel_schedule_blocks {
                    bmodelica.schedule_block {
                        bmodelica.equation_call @equation_1 {indices = #modeling<multidim_range [0,9]>}
                    } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@y, #modeling<index_set {[0,9]}>>]}
                }
            }
        }
    }

    bmodelica.equation_function @equation_0(%arg0: index, %arg1: index) {
        bmodelica.yield
    }

    bmodelica.equation_function @equation_1(%arg0: index, %arg1: index) {
        bmodelica.yield
    }
}