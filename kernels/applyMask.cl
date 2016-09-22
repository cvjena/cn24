/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

__kernel void APPLY_MASK (__global float* X,
                       __global float* Y,
                       __global float* Z) {
 uint i = get_global_id (0);
 Z[i] = X[i] * Y[i];
}
