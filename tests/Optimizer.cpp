/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <vector>
#include <cn24.h>

std::vector<Conv::JSON> test_optimizers{
    Conv::JSON::object({
                           {"optimization_method","gd"},
                           {"gd_momentum", 0.8},
                           {"learning_rate", 0.1}
                       }),
    Conv::JSON::object({
                           {"optimization_method","adam"},
                           {"ad_step_size", 0.1},
//                           {"ad_beta1", 0.0},
//                           {"ad_beta2", 0.0},
                           {"ad_epsilon", 0.00000001}
                       })
};

int main() {
  Conv::System::Init();

  for(Conv::JSON& optimizer_json : test_optimizers) {
    LOGINFO << "Testing optimizer: " << optimizer_json.dump();
    Conv::Optimizer* optimizer = Conv::JSONOptimizerFactory::ConstructOptimizer(optimizer_json);
    if(optimizer != nullptr) {
      Conv::CombinedTensor ctensor(3);
      ctensor.data.Clear();
      ctensor.data(0) = (Conv::datum)-3921.392;
      ctensor.data(0) = (Conv::datum)21.392;
      ctensor.data(0) = (Conv::datum)0.2139;
      ctensor.delta.Clear();

      Conv::CombinedTensor unrelated_tensor(15);
      unrelated_tensor.data.Clear();
      unrelated_tensor.delta.Clear();

      bool reached_minimum = false;
      for(unsigned int i = 0; i < 10000; i++) {
        // Evaluate gradient
        ctensor.delta[0] = (Conv::datum)2.0 * (ctensor.data(0) - (Conv::datum)7.0);
        ctensor.delta[1] = (Conv::datum)0.4 * ctensor.data(1) + (Conv::datum)0.4;
        ctensor.delta[2] = (Conv::datum)6.0 * ctensor.data(2) - (Conv::datum)69.0;

        // Do gradient step
        optimizer->Step({&ctensor, &unrelated_tensor}, i);

        // Evaluate test function
        Conv::datum function_value =
            (ctensor.data(0) - (Conv::datum)14.0) * ctensor.data(0)
            + (Conv::datum)0.2 * (ctensor.data(1) + (Conv::datum)2.0) * ctensor.data(1)
            + (Conv::datum)3.0 * ctensor.data(2) * (ctensor.data(2) - (Conv::datum)23.0);

        // Minimum is at (7,-1,23/2)
        Conv::datum distance_to_minimum =
            (ctensor.data(0) - (Conv::datum)7) * (ctensor.data(0) - (Conv::datum)7)
            + (ctensor.data(1) + (Conv::datum)1) * (ctensor.data(1) + (Conv::datum)1)
            + (ctensor.data(2) - (Conv::datum)(23.0/2.0)) * (ctensor.data(2) - (Conv::datum)(23.0/2.0));

        if(distance_to_minimum < (Conv::datum)0.01 && (function_value - (Conv::datum)(8919.0/2.0)) < (Conv::datum)0.01) {
          reached_minimum = true;
          LOGINFO << "Reached minimum after " << i << " iterations.";
          break;
        }
      }

      if(!reached_minimum) {
        FATAL("  ...failed to reach minimum! Stopped at (" << ctensor.data(0) << "," << ctensor.data(1) << "," << ctensor.data(2) << ")");
      } else {
        for(unsigned int e = 0; e < unrelated_tensor.data.elements(); e++) {
          if(unrelated_tensor.data(e) != (Conv::datum)0) {
            FATAL("  ...modified zero-gradient data!");
          }
        }
      }
    } else {
      FATAL("  ...would not construct!");
    }
  }
  LOGEND;
  return 0;
}
