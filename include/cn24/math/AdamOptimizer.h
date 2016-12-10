/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_ADAMOPTIMIZER_H
#define CONV_ADAMOPTIMIZER_H

#include "Optimizer.h"

namespace Conv {
class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(JSON descriptor);
  void Step(std::vector<Tensor> &buffers, CombinedTensor* parameters, unsigned int iteration);
  unsigned int GetRequiredBufferCount() const { return 2; }
  std::string GetStatusDescription(unsigned int iteration);

private:
  datum step_size_;
  datum beta1_;
  datum beta2_;
  datum epsilon_;
  bool sqrt_ss_;
};
}
#endif
