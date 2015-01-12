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
 * NDRange for biased matrix vector
 *  1: Output neuron
 *  2: sample
 */
__kernel void BIASED_MATRIX_VECTOR_FWD ( __global float* X,
                                 __global float* W,
                                 __global float* b,
                                 __global float* Y,
                                 uint input_units,
				 uint neurons,
				 float weight_factor
                               )
{
  
    uint output_neuron = get_global_id ( 0 );
    uint sample = get_global_id ( 1 ); 
    
    float sum = 0;
    
    for(uint i = 0; i < input_units; i++) {
      const float W_val = W[i * neurons + output_neuron];
      const float X_val = X[i + sample * input_units];
      sum += W_val * X_val;
    }
    
    sum *= weight_factor;
    sum += b[output_neuron];
    
    Y[sample * neurons + output_neuron] = sum;
}

/*
 * NDRange for biased matrix vector backwards
 *  1: Input neuron
 *  2: sample
 */
__kernel void BIASED_MATRIX_VECTOR_BWD ( __global float* dX,
                                 __global float* W,
                                 __global float* dY,
                                 uint input_units,
				 uint neurons
                               )
{
  
    uint input_neuron = get_global_id ( 0 );
    uint sample = get_global_id ( 1 ); 
    
    float sum = 0;
    
    for(uint o = 0; o < neurons; o++) {
      const float W_val = W[input_neuron * neurons + o];
      const float dY_val = dY[sample * neurons + o];
      sum += W_val * dY_val;
    }
   
    dX[input_neuron + sample * input_units] = sum;
}

/*
 * NDRange for biased matrix vector for bias gradient
 *  1: Output neuron
 */
__kernel void BIASED_MATRIX_VECTOR_GRAD ( __global float* db,
                                 __global float* dY,
				 uint neurons,
				 uint samples,
				 float local_lr
                               )
{
  
    uint output_neuron = get_global_id ( 0 );
    
    float sum = 0;
    for(uint sample = 0; sample < samples; sample++) {
      const float dY_val = dY[output_neuron + sample * neurons];
      sum += dY_val; 
    }
    
    sum *= local_lr;
    
    db[output_neuron] = sum;
}
