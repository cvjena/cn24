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
 * NDRange for backward convolution:
 *  1: Input X coordinate
 *  2: Input Y coordinate
 *  3: Kernel input map number + sample number * total input maps
 */
__kernel void FULL_CONVOLUTION ( __read_only __global float* dY,
                                 __read_only __global float* W,
                                 __write_only __global float* dX,
                                 uint input_width,
                                 uint input_height,
                                 uint input_maps,
                                 uint kernel_width,
                                 uint kernel_height,
                                 uint output_width,
                                 uint output_height,
                                 uint output_maps
                               )
{

    const uint input_x = get_global_id ( 0 );
    const uint input_y = get_global_id ( 1 );

    const uint sample = get_global_id ( 2 ) / input_maps;
    const uint imap = get_global_id ( 2 ) % input_maps;

    float sum = 0.0;
    const uint dY_x0 = ( uint ) max ( 0, ( int ) input_x - ( ( int ) kernel_width - 1 ) );
    const uint dY_y0 = ( uint ) max ( 0, ( int ) input_y - ( ( int ) kernel_height - 1 ) );
    const uint dY_xmax = ( uint ) min ( ( int ) input_x, ( int ) output_width - 1 );
    const uint dY_ymax = ( uint ) min ( ( int ) input_y, ( int ) output_height - 1 );
    const uint ky_0 = ( int ) dY_y0 - ( ( int ) input_y - ( ( int ) kernel_height - 1 ) );
    const uint kx_0 = ( int ) dY_x0 - ( ( int ) input_x - ( ( int ) kernel_width - 1 ) );

    const uint output_idx_sample = output_width * output_height * output_maps * sample;
    for ( uint omap = 0; omap < output_maps; omap++ ) {
        const uint output_idx_omap = output_idx_sample + ( output_width * output_height * omap );
        const uint kernel_idx_omap = kernel_width * kernel_height * input_maps * omap;
        const uint kernel_idx_imap = kernel_idx_omap + ( kernel_width * kernel_height * imap );

        for ( uint oy = dY_y0, ky = ky_0; oy <= dY_ymax; oy++, ky++ ) {
            const uint output_idx_line = output_idx_omap + ( output_width * oy );
//      const uint kernel_idx_line = kernel_idx_imap + (kernel_width * ky);
            const uint kernel_idx_line = kernel_idx_imap + ( kernel_width * ( kernel_height - ( ky + 1 ) ) );
            for ( uint ox = dY_x0, kx = kx_0; ox <= dY_xmax; ox++, kx++ ) {
//	const float W_value = W[kernel_idx_line + kx];
                const float W_value = W[kernel_idx_line + ( kernel_width - ( kx + 1 ) )];
                const float dY_value = dY[output_idx_line + ox];
                sum += W_value * dY_value;
            }
        }
    }

    const uint input_idx_sample = input_width * input_height * input_maps * sample;
    const uint input_idx_imap = input_idx_sample + ( input_width * input_height * imap );

    const uint input_idx_line = input_idx_imap + ( input_width * input_y );

    dX[input_idx_line + input_x] = sum;

}

