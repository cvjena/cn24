/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
__kernel void SMS ( __global float* X, __global float* Y,
                        uint width, uint height, uint maps, uint samples )
{
    uint target_element = get_global_id ( 0 );
    uint target_skid = target_element / (width * height);
    
    uint target_sample = target_skid / maps;
    uint target_map = target_skid % maps;
    
    uint source_element = target_element % (width * height);
    
    Y[target_element] = X[(target_map * width * height * samples) + (target_sample * width * height) + source_element];
}