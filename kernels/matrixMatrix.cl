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
 * NDRange for matrix matrix
 *  1: Output neuron
 *  2: Input neuron
 */
__kernel void MATRIX_MATRIX ( __global float* X,
                                 __global float* dW,
                                 __global float* dY,
                                 uint input_units,
				 uint neurons,
			         uint samples,
				 float local_lr
                               )
{
  
    uint output_neuron = get_global_id ( 0 );
    uint input_neuron = get_global_id ( 1 );
    
    float sum = 0;
    
    for(uint sample = 0; sample < samples; sample++)  {
      const float X_val = X[input_neuron + sample * input_units];
      const float dY_val = dY[output_neuron + sample * neurons];
      sum += X_val * dY_val;
    }     
      
    sum *= local_lr;
    
    dW[input_neuron * neurons + output_neuron] = sum;
}
