/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
__kernel void DOWN ( __global float* X,
                   __global float* Y,
                   uint target_width,
                   uint target_height,
                   uint source_width,
                   uint source_height,
                   uint region_width,
                   uint region_height,
                   float target_factor)
{
  uint target_x = get_global_id(0);
  uint target_y = get_global_id(1);
  uint target_skid = get_global_id(2);
  
  uint source_x = target_x * region_width;
  uint source_y = target_y * region_height;
  
  uint X_sk = source_width * source_height * target_skid;
  
  float sum = 0.0;
  for(uint ry = 0; ry < region_height; ry++) {
    const uint X_line = X_sk + (source_width * (source_y + ry));
    for(uint rx = 0; rx < region_width; rx++) {
      const uint X_idx = X_line + source_x + rx;
      const float X_val = X[X_idx];
      sum += X_val;
    }
  }
  
  uint Y_sk = target_width * target_height * target_skid;
  uint Y_line = Y_sk + (target_width * target_y);
  uint Y_idx = Y_line + target_x;
  Y[Y_idx] = sum * target_factor;
}


__kernel void UP ( __global float* X,
                   __global float* Y,
                   uint target_width,
                   uint target_height,
                   uint source_width,
                   uint source_height,
                   uint region_width,
                   uint region_height,
                   float target_factor)
{
  uint target_x = get_global_id(0);
  uint target_y = get_global_id(1);
  uint target_skid = get_global_id(2);
  
  uint source_x = target_x / region_width;
  uint source_y = target_y / region_height;
  
  uint X_sk = source_width * source_height * target_skid;
  const uint X_line = X_sk + (source_width * source_y);
  const uint X_idx = X_line + source_x;
  const float X_val = X[X_idx];
  
  uint Y_sk = target_width * target_height * target_skid;
  uint Y_line = Y_sk + (target_width * target_y);
  uint Y_idx = Y_line + target_x;
  Y[Y_idx] = X_val * target_factor;
}