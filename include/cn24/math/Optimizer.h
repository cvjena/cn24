/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_OPTIMIZER_H
#define CONV_OPTIMIZER_H

#include "../util/Config.h"
#include "../util/Tensor.h"
#include "../util/CombinedTensor.h"
#include "../util/JSONParsing.h"

namespace Conv {
class Optimizer {
protected:
  Optimizer(JSON configuration);

  // Number of buffers that match the parameter space in size
  // e.g. buffer for momentum
  virtual unsigned int GetRequiredBufferCount() const = 0;

  // Perform an optimization step
  virtual void Step(std::vector<Tensor>& buffers, CombinedTensor* parameters, unsigned int iteration) = 0;

  // Reset additional state
  virtual void ResetInner();

public:
  virtual ~Optimizer();

  // Reset/initialize state
  void Reset();

  // Perform an optimization step (public method)
  void Step(const std::vector<CombinedTensor*>& parameters, unsigned int iteration = 0);

  // Get a string that describes the inner state of the optimizer (e.g. current learning rate)
  virtual std::string GetStatusDescription(unsigned int iteration) { UNREFERENCED_PARAMETER(iteration); return ""; }

  // Returns true if optimizer is OpenCL aware
  virtual bool IsGPUMemoryAware() { return false; }
private:
  JSON configuration_;
  std::vector<std::vector<Tensor>> buffers_;
protected:
  unsigned int iterations_since_reset_ = 0;
};
}

#endif
