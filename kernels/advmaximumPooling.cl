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
 * NDRange for maximum pooling forward
 *  1: Output X coordinate
 *  2: Output Y coordinate
 *  3: sample number * map number
 */
__kernel void AMAXIMUM_POOLING_FWD (  __global float* X,
                                 __global uint* M,
                                 __global float* Y,
                                 uint input_width,
                                 uint input_height,
                                 uint maps,
                                 uint output_width,
                                 uint output_height,
				 uint region_width,
				 uint region_height,
         uint stride_width,
         uint stride_height
                               )
{
    const uint output_x = get_global_id ( 0 );
    const uint output_y = get_global_id ( 1 );
    const uint skid = get_global_id ( 2 );
    
    const uint input_x = output_x * stride_width;
    const uint input_y = output_y * stride_height;
    
    const uint M_idx_sk = input_width * input_height * skid;

    // DO NOT USE FLT_MIN, IT WILL _NOT_ WORK WITH SOME OPENCL COMPILERS
    float maximum = -FLT_MAX;
    uint max_x = 0;
    uint max_y = 0;
    for(uint y = 0; y < region_height; y++) {
      const uint M_idx_line = M_idx_sk + (input_width * (input_y + y));
      for(uint x = 0; x < region_width; x++) {
        const uint M_idx = M_idx_line + input_x + x;
        const float X_val = X[M_idx];
        if(X_val > maximum) {
          max_x = x;
          max_y = y;
          maximum = X_val;
        }
      }
    }
    
    // Save result
    const uint Y_idx_sk = output_width * output_height * skid;
    const uint Y_idx_line = Y_idx_sk + (output_width * output_y);
    Y[Y_idx_line + output_x] = maximum;
    
    M[Y_idx_line + output_x] = (input_y + max_y) * input_width + (input_x + max_x);
}

/*
 * NDRange for maximum pooling backward
 *  1: Input X coordinate
 *  2: Input Y coordinate
 *  3: sample number * map number
 */
__kernel void AMAXIMUM_POOLING_BWD (  
				  __global float* X,
				 __global uint* M,
                                 __global float* Y,
                                 uint input_width,
                                 uint input_height,
                                 uint maps,
                                 uint output_width,
                                 uint output_height,
				 uint region_width,
				 uint region_height,
         uint stride_width,
         uint stride_height
                               )
{
    const uint input_x = get_global_id ( 0 ); // OK
    const uint input_y = get_global_id ( 1 ); // OK
    const uint skid = get_global_id ( 2 ); // OK
    const uint mask_index = input_y * input_width + input_x;
    
    const uint Y_idx_sk = output_width * output_height * skid; // OK
    
    const uint oxstart = (input_x < region_width) ? 
      0 : (input_x - region_width) / stride_width + 1;
    const uint oxend = min(input_x / stride_width + 1, output_width);
    
    const uint oystart = (input_y < region_height) ? 
      0 : (input_y - region_height) / stride_height + 1;
    const uint oyend = min(input_y / stride_height + 1, output_height);
    
    //printf("ox: %d-%d, oy: %d-%d, ix:%d, iy:%d\n", oxstart, oxend, oystart, oyend, input_x, input_y);
    
    float sum = 0.0;
    uint runs=0;
    for(uint output_y = oystart; output_y < oyend; output_y++) {
      const uint Y_idx_line = Y_idx_sk + (output_width * output_y); // OK
      for(uint output_x = oxstart; output_x < oxend; output_x++) {
        runs++;
        const uint Y_idx = Y_idx_line + output_x; // OK
        //printf("current mask index: %d vs. read %d, yval: %f", mask_index, M[Y_idx], Y[Y_idx]);
        if(M[Y_idx] == mask_index)
          sum += Y[Y_idx];
      }
    }
    
    //printf("runs: %d, sum: %f\n", runs, sum);
    
    const uint M_idx_sk = input_width * input_height * skid; // OK
    const uint M_idx_line = M_idx_sk + (input_width * input_y); // OK
    const uint M_idx = M_idx_line + input_x; // OK
    
    X[M_idx] = sum; // OK
}