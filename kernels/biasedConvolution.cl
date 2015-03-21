//  This file is part of the CN24 Semantic Segmentation Software
//  Copyright (C) 2014 Clemens-Alexander Brust (ikosa dot de at gmail dot com)

//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

/*
 * NDRange for forward convolution:
 *  1: Output X coordinate
 *  2: Output Y coordinate
 *  3: Kernel number + sample number * total kernels
 */
__kernel void BIASED_CONVOLUTION ( __global __read_only float* X,
                                   __global __read_only float* W,
                                   __global __read_only float* b,
                                   __global __write_only float* Y,
                                   __global __read_only float* D,
                                   uint input_width,
                                   uint input_height,
                                   uint input_maps,
                                   uint kernel_width,
                                   uint kernel_height,
                                   uint output_width,
                                   uint output_height,
                                   uint output_maps,
                                   float weight_factor
                                 )
{
    uint output_x = get_global_id ( 0 );
    uint output_y = get_global_id ( 1 );
    uint sk_id = get_global_id ( 2 );
    uint kernel_id = sk_id % output_maps;
    uint sample_id = sk_id / output_maps;

    float sum = 0;
    float dropout_mask_item = D[sk_id];

    if(dropout_mask_item != 0.0) {
      uint X_begin_sample = input_width * input_height * input_maps * sample_id;
      uint W_begin_kid = kernel_width * kernel_height * input_maps * kernel_id;

      for ( uint imap = 0; imap < input_maps; imap++ ) {
          const uint X_begin_imap = X_begin_sample + ( imap * input_width * input_height );
          const uint W_begin_imap = W_begin_kid + ( imap * kernel_width * kernel_height );
          for ( uint ky = 0; ky < kernel_height; ky++ ) {
              const uint X_begin_kyx = X_begin_imap + ( input_width * ( ky + output_y ) ) + output_x;
              const uint W_begin_ky = W_begin_imap + ( kernel_width * ky );
              if ( kernel_width > 3 ) {
                  const uint vector_fetch_end = ( kernel_width - 4 ) & ~ ( 0x3 );
                  for ( uint kx = 0; kx <= vector_fetch_end; kx+=4 ) {
                      const float4 X_value = vload4(0, X + X_begin_kyx + kx);
                      const float4 W_value = vload4(0, W + W_begin_ky + kx);
                      sum += dot ( X_value, W_value );
                  }
                  for ( uint kx = vector_fetch_end + 4; kx < kernel_width; kx++ ) {
                      const float X_value = X[X_begin_kyx + kx];
                      const float W_value = W[W_begin_ky + kx];
                      sum += ( X_value * W_value );
                  }
              } else {
                  for ( uint kx = 0; kx < kernel_width; kx++ ) {
                      const float X_value = X[X_begin_kyx + kx];
                      const float W_value = W[W_begin_ky + kx];
                      sum += ( X_value * W_value );
                  }
              }
          }
      }

      sum += b[kernel_id];
      sum *= weight_factor;
    }

    const uint Y_begin_sample = output_width * output_height * sk_id;
    const uint Y_begin_line = Y_begin_sample + ( output_width * output_y );
    Y[Y_begin_line + output_x] = sum;

}
