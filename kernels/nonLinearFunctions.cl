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
 * NDRange for nonlinearity
 *  1: Element
 */
__kernel void NL_TANH_FWD ( __global float* X,
                        __global float* Y
                      )
{
    uint element = get_global_id ( 0 );
    Y[element] = tanh ( X[element] );
}
__kernel void NL_TANH_BWD ( __global float* dX,
			    __global float* Y,
                        __global float* dY
                      )
{
    uint element = get_global_id ( 0 );
    const float Y_value = Y[element];
    const float dY_value = dY[element];
    dX[element] = dY_value * (1.0 - (Y_value * Y_value));
}

__kernel void NL_SIGM_FWD ( __global float* X,
                        __global float* Y
                      )
{
    uint element = get_global_id ( 0 );
    Y[element] =  1.0f / ( 1.0f + exp( - X[element] ));
}

__kernel void NL_SIGM_BWD ( __global float* dX,
                        __global float* Y,
			__global float* dY
                      )
{
    uint element = get_global_id ( 0 );
    const float Y_value = Y[element];
    const float dY_value = dY[element];
    dX[element] = dY_value * Y_value * (1.0 - Y_value);
}

__kernel void NL_LEAKY_FWD ( __global float* X,
                        __global float* Y
                      )
{
    uint element = get_global_id ( 0 );
    if(X[element] > 0) {
        Y[element] =  X[element];
    } else {
        Y[element] =  0.1 * X[element];
    }
}


__kernel void NL_LEAKY_BWD ( __global float* dX,
                        __global float* X,
			__global float* dY
                      )
{
    uint element = get_global_id ( 0 );
    const float X_value = X[element];
    const float dY_value = dY[element];
    if(X_value > 0) {
        dX[element] = dY_value;
    } else {
        dX[element] = dY_value * 0.1f;
    }
}
