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
 * NDRange for CROSS_CORRELATION:
 *  1: Output X coordinate
 *  2: Output Y coordinate
 *  3: Kernel input map + (input maps * output map) + (input maps * output maps * sample)
 */

__kernel void CROSS_CORRELATION ( __read_only __global float* X,
                                  __read_only __global float* dY,
                                  __write_only __global float* dWps,
                                  uint input_height,
                                  uint input_width,
                                  uint input_maps,
                                  uint output_width,
                                  uint output_height,
                                  uint output_maps,
                                  uint kernel_width,
                                  uint kernel_height,
                                  uint samples,
                                  float local_lr )
{

    const uint dW_x = get_global_id ( 0 );
    const uint dW_y = get_global_id ( 1 );

    const uint dW_imap = get_global_id ( 2 ) % input_maps;
    const uint sample = get_global_id ( 2 ) / ( input_maps * output_maps );
    const uint dW_omap = ( get_global_id ( 2 ) % ( input_maps * output_maps ) ) / input_maps;

    float sum = 0;

    const uint input_idx_sample = input_width * input_height * input_maps * sample;
    const uint input_idx_imap = input_idx_sample + ( input_width * input_height * dW_imap );

    const uint output_idx_sample = output_width * output_height * output_maps * sample;
    const uint output_idx_omap = output_idx_sample + ( output_width * output_height * dW_omap );

    for ( uint oy = 0; oy < output_height; oy++ ) {
        const uint output_idx_line = output_idx_omap + ( output_width * oy );
        const uint input_idx_linex = input_idx_imap + ( input_width * ( oy + dW_y ) ) + dW_x;

        if ( output_width > 3 ) {
            const uint vector_fetch_end = ( output_width - 4 ) & ~ ( 0x3 );
            for ( uint ox = 0; ox <= vector_fetch_end; ox+=4 ) {
                const float4 dY_value = vload4(0, dY + output_idx_line + ox);
                const float4 X_value = vload4(0, X + input_idx_linex + ox);
		sum += dot(dY_value, X_value);
	    }
            for ( uint ox = vector_fetch_end + 4; ox < output_width; ox++ ) {
                const float dY_value = dY[output_idx_line + ox];
                const float X_value = X[input_idx_linex + ox];
                sum += ( dY_value * X_value );
            }
        } else {
            for ( uint ox = 0; ox < output_width; ox++ ) {
                const float dY_value = dY[output_idx_line + ox];
                const float X_value = X[input_idx_linex + ox];
                sum += ( dY_value * X_value );
            }
        }
    }

    const uint target_idx_sample = kernel_width * kernel_height * input_maps * output_maps * sample;
    const uint target_idx_omap = target_idx_sample + ( kernel_width * kernel_height * input_maps * dW_omap );
    const uint target_idx_imap = target_idx_omap + ( kernel_width * kernel_height * dW_imap );
    const uint target_idx_line = target_idx_imap + ( kernel_width * dW_y );

    dWps[target_idx_line + dW_x] = sum * local_lr;
}
