/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>
#include <random>

int main() {
  Conv::System::Init();

  Conv::Tensor test_tensor(1, 16, 16, 1);
  std::mt19937 test_rand;
  std::uniform_real_distribution<Conv::datum> dist((Conv::datum)-2, (Conv::datum)2);

  for(unsigned int y = 0; y < 16; y++) {
    for(unsigned int x = 0; x < 16; x++) {
      *(test_tensor.data_ptr(x, y)) = dist(test_rand);
    }
  }

  for(Conv::datum y = 0; y <= 15.0; y += 0.037) {
    for(Conv::datum x = 0; x <= 15.0; x += 0.0821) {
      unsigned int left_x = std::floor(x);
      unsigned int right_x = std::ceil(x);
      unsigned int left_y = std::floor(y);
      unsigned int right_y = std::ceil(y);
      const Conv::datum Q11 = *(test_tensor.data_ptr(left_x, left_y));
      const Conv::datum Q21 = *(test_tensor.data_ptr(right_x, left_y));
      const Conv::datum Q12 = *(test_tensor.data_ptr(left_x, right_y));
      const Conv::datum Q22 = *(test_tensor.data_ptr(right_x, right_y));

      const Conv::datum max_value = std::max(std::max(Q11,Q21),std::max(Q12,Q22));
      const Conv::datum min_value = std::min(std::min(Q11,Q21),std::min(Q12,Q22));

      const Conv::datum smooth_value = test_tensor.GetSmoothData(x, y, 0, 0);
      Conv::AssertGreaterEqual(min_value, smooth_value, "Smooth Value");
      Conv::AssertLessEqual(max_value, smooth_value, "Smooth Value");
    }
  }

  LOGEND;
  return 0;
}