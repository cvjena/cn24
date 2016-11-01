/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_SGDOPTIMIZER_H
#define CONV_SGDOPTIMIZER_H

#include "Optimizer.h"

namespace Conv {
class SGDOptimizer : public Optimizer {
public:
  SGDOptimizer(JSON descriptor);
  void Step(std::vector<Tensor> &buffers, CombinedTensor* parameters, unsigned int iteration);
  unsigned int GetRequiredBufferCount() const { return 1; }
  std::string GetStatusDescription(unsigned int iteration);

private:
  datum learning_rate_;
  datum momentum_;
  datum learning_rate_exponent_;
  datum learning_rate_gamma_;
};
}
#endif
