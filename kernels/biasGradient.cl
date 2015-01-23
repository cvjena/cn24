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
 * NDRange for bias gradient part 1:
 *  1: Output map
 *  2: Sample
 */
__kernel void BIAS_GRADIENT_PART1 ( __global float* dY,
                                    __global float* bias_buffer,
                                    uint output_width,
                                    uint output_height,
                                    uint output_maps
                                  )
{
    const uint output_map = get_global_id ( 0 );
    const uint sample = get_global_id ( 1 );

    const uint dY_idx_sample = output_width * output_height * output_maps * sample;
    const uint dY_idx_omap = dY_idx_sample + output_width * output_height * output_map;

    float sum = 0;

    for ( uint y = 0; y < output_height; y++ ) {
        const uint dY_idx_line = dY_idx_omap + output_width * y;
        if ( output_width > 3 ) {
            const uint vector_fetch_end = ( output_width - 4 ) & ~ ( 0x3 );
            for ( uint x = 0; (x << 2) <= vector_fetch_end; x++ ) {
		const float4 dY_val = vload4(x, dY + dY_idx_line);
		sum += dot(dY_val, 1.0);
            }
            for ( uint x = vector_fetch_end + 4 ; x < output_width; x++ ) {
                const float dY_val = dY[dY_idx_line + x];
                sum += dY_val;
            }
        } else {
            for ( uint x = 0; x < output_width; x++ ) {
                const float dY_val = dY[dY_idx_line + x];
                sum += dY_val;
            }
        }
    }

    bias_buffer[sample * output_maps + output_map] = sum;
}

/*
 * NDRange for bias gradient part 2
 *  1: Output map
 */
__kernel void BIAS_GRADIENT_PART2 ( __global float* bias_buffer,
                                    __global float* db,
                                    uint output_maps,
                                    uint samples,
                                    float local_lr )
{
    const uint output_map = get_global_id ( 0 );
    float sum = 0;
    for ( uint sample = 0; sample < samples; sample++ ) {
        const float bias_buffer_val = bias_buffer[sample * output_maps + output_map];
        sum += bias_buffer_val;
    }

    sum *= ( local_lr );

    db[output_map] = sum;
}
