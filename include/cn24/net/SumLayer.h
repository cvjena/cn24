/**
 * @file SumLayer.h
 * @class SumLayer
 * @brief Concatenates the inputs (used to add non-convolvable information).
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_SUMLAYER_H
#define CONV_SUMLAYER_H

#include <string>

#include "Layer.h"

namespace Conv {

class SumLayer: public Layer {
public:
  SumLayer();

  // Layer implementations
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs,
                const NetStatus* status );
  void FeedForward();
  void BackPropagate();

  std::string GetLayerDescription() { return "Sum Layer"; }
  void CreateBufferDescriptors(std::vector< NetGraphBuffer >& buffers) {
    NetGraphBuffer buffer;
    buffer.description = "Output";
    buffers.push_back(buffer);
  };
private:
  CombinedTensor* input_a_ = nullptr;
  CombinedTensor* input_b_ = nullptr;
  CombinedTensor* output_ = nullptr;
  
  unsigned int maps_ = 0;
  unsigned int samples_ = 0;
};

}
#endif
