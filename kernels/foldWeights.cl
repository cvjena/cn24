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
 * NDRange for FOLD_WEIGHTS
 *  1: Kernel X coordinate
 *  2: Kernel Y coordinate
 *  3: Kernel input map + (input maps * output map)
 */

__kernel void FOLD_WEIGHTS ( __read_only __global float* dWps,
                                  __write_only __global float* dW,
                                  uint input_maps,
                                  uint output_maps,
                                  uint kernel_width,
                                  uint kernel_height,
                                  uint samples )
{

    const uint dW_x = get_global_id ( 0 );
    const uint dW_y = get_global_id ( 1 );

    const uint dW_imap = get_global_id ( 2 ) % input_maps;
    const uint dW_omap = get_global_id ( 2 ) / input_maps;

    float sum = 0;
    for(uint sample = 0; sample < samples; sample++) {
      const uint dwps_idx_sample = kernel_width * kernel_height * input_maps * output_maps * sample;
      const uint dwps_idx_omap = dwps_idx_sample + (kernel_width * kernel_height * input_maps * dW_omap);
      const uint dwps_idx_imap = dwps_idx_omap + (kernel_width * kernel_height * dW_imap);
      const uint dwps_idx_line = dwps_idx_imap + (kernel_width * dW_y);
      sum += dWps[dwps_idx_line + dW_x];
    }

    const uint target_idx_omap = kernel_width * kernel_height * input_maps * dW_omap;
    const uint target_idx_imap = target_idx_omap + ( kernel_width * kernel_height * dW_imap );
    const uint target_idx_line = target_idx_imap + ( kernel_width * dW_y );

    dW[target_idx_line + dW_x] = sum;
}
