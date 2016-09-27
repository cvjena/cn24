/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_DUMMYERRORLAYER_H
#define CONV_DUMMYERRORLAYER_H

#include "../util/CombinedTensor.h"
#include "Layer.h"
#include "LossFunctionLayer.h"
#include "NetStatus.h"

namespace Conv {

class DummyErrorLayer : public Layer, public LossFunctionLayer {
public:
  DummyErrorLayer() : Layer(JSON::object()) {};

  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs) {
		UNREFERENCED_PARAMETER(inputs);
		UNREFERENCED_PARAMETER(outputs);
		return true; }
  bool Connect (const std::vector< CombinedTensor* >& inputs, const std::vector< CombinedTensor* >& outputs, const NetStatus* net ) {
		UNREFERENCED_PARAMETER(outputs);
		UNREFERENCED_PARAMETER(net);
		first_ = inputs[0];
		second_ = inputs[1];
		third_ = inputs[2];
		return true;
	}
  void FeedForward() { first_->delta.Clear(); }
  void BackPropagate() { }

  // Implementations for LossFunctionLayer
  datum CalculateLossFunction() { return (datum)0.0; }

	std::string GetLayerDescription() {
		std::ostringstream ss;
		ss << "Dummy Error Layer";
		return ss.str();
	}
private:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;
};

}

#endif
