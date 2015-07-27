/**
 * @file ConcatenationLayer.h
 * @class ConcatenationLayer
 * @brief Concatenates the inputs (used to add non-convolvable information).
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CONCATENATIONLAYER_H
#define CONV_CONCATENATIONLAYER_H

#include <string>

#include "Layer.h"

namespace Conv {

class ConcatenationLayer: public Layer {
public:
  ConcatenationLayer();

  // Layer implementations
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs,
                const NetStatus* status );
  void FeedForward();
  void BackPropagate();

  std::string GetLayerDescription() { return "Concatenation Layer"; }
  void CreateBufferDescriptors(std::vector< NetGraphBuffer >& buffers) {
    NetGraphBuffer buffer;
    buffer.description = "Output";
    buffers.push_back(buffer);
  };
private:
  CombinedTensor* input_a_ = nullptr;
  CombinedTensor* input_b_ = nullptr;
  CombinedTensor* output_ = nullptr;
  
  unsigned int maps_a_ = 0;
  unsigned int maps_b_ = 0;
  unsigned int samples_ = 0;
};

}
#endif
