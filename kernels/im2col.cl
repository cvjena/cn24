/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
__kernel void IM2COL ( __global float* X, __global float* Y,
                       int source_width, int source_height,
                       int maps, int samples,
                       int target_width, int target_height, int target_maps,
                       int kernel_width, int kernel_height,
                       int stride_width, int stride_height,
                       int pad_width, int pad_height)
{
  const int target_oelement = get_global_id ( 0 );
  const int target_map = get_global_id ( 1 );
  const int sample = get_global_id ( 2 );
  
  const int ox = target_oelement % target_width;
  const int oy = target_oelement / target_width;
  
  const int kx = target_map % kernel_width;
  const int ky = (target_map / kernel_width) % kernel_height;
  const int imap = target_map / (kernel_width * kernel_height);

  const int iy = oy * stride_height - pad_height + ky;
  const int ix = ox * stride_width - pad_width + kx;
  
  if(iy >= 0 && iy < source_height && ix >= 0 && ix < source_width && imap < maps && sample < samples) {
    Y[(target_map * samples * target_width * target_height)
        + (sample * target_height + oy) * target_width + ox] =
      X[(sample * maps * source_width * source_height)
        + (imap * source_height + iy) * source_width + ix];
  } else {
    Y[(target_map * samples * target_width * target_height)
        + (sample * target_height + oy) * target_width + ox] = 0.0;
  }
}

__kernel void COL2IM ( __global float* X, __global float* Y,
                       int source_width, int source_height,
                       int maps, int samples,
                       int target_width, int target_height, int target_maps,
                       int kernel_width, int kernel_height,
                       int stride_width, int stride_height,
                       int pad_width, int pad_height)
{
  const int source_oelement = get_global_id ( 0 );
  const int source_map = get_global_id ( 1 );
  const int sample = get_global_id ( 2 );
  
  const int iy = source_oelement / source_width;
  const int ix = source_oelement % source_width;

  float sum = 0;
  for(int ky = 0; ky < kernel_height; ky++) {
    const int oy = (pad_height + iy - ky) / stride_height;
    if(oy >= 0 && oy < target_height && ((pad_height + iy - ky) % stride_height == 0)) {
      for(int kx = 0; kx < kernel_width; kx++) {
        const int ox = (pad_width + ix - kx) / stride_width;
        if(ox >= 0 && ox < target_width && ((pad_width + ix - kx) % stride_width == 0)) {
          const int target_map = (kernel_width * kernel_height) * source_map +
            (kernel_width * ky) + kx;
          
          sum += Y[(target_map * samples * target_width * target_height)
            + (sample * target_height + oy) * target_width + ox];
        }
      }
    }
  }
  
  X[(sample * maps * source_width * source_height) +
    (source_map * source_width * source_height) + source_oelement ] = sum;
}