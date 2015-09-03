/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
__kernel void SET_VALUE ( __global float* X,
                        float value, uint offset )
{
    uint element = get_global_id ( 0 );
    X[element + offset] = value;
}