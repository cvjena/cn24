/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <cstring>
#include <cmath>

#include "Segmentation.h"

#include <cstdlib>

namespace Conv {

void Segmentation::ExtractPatches (const int patchsize_x,
                                   const int patchsize_y,
                                   Tensor& target,
                                   Tensor& helper,
                                   const Tensor& source,
                                   const int source_sample,
                                   bool substract_mean) {
  int image_width = source.width();
  int image_height = source.height();
  unsigned int image_maps = source.maps();

  // new version
  unsigned int npatches = image_width * image_height;
  int offsetx = (patchsize_x / 2);
  int offsety = (patchsize_y / 2);


  target.Resize (npatches, patchsize_x, patchsize_y, image_maps);
  helper.Resize (npatches, 2);

  for (int px = 0; px < image_width; px++) {
    for (int py = 0; py < image_height; py++) {
      unsigned int element =  image_width * py + px;
      for (int ipy = 0; ipy < patchsize_y; ipy++) {
        int image_y = py + ipy - offsety;
        if (image_y < 0)
          image_y = -image_y;
        if (image_y >= image_height)
          image_y = -1 + image_height + image_height - image_y;

        for (int ipx = 0; ipx < patchsize_x; ipx++) {
          int image_x = px + ipx - offsetx;
          if (image_x < 0)
            image_x = -image_x;
          if (image_x >= image_width)
            image_x = -1 + image_width + image_width - image_x;

          // Copy pixel
          for (unsigned int map = 0; map < image_maps; map++) {
            const datum pixel =
              *source.data_ptr_const (image_x, image_y, map, source_sample);

            *target.data_ptr (ipx, ipy, map, element) = pixel;
          }
        }
      }

      // Copy helper data
      helper[2 * element] =
        fmax (0.0, fmin (image_height - (patchsize_y - 1), py - offsety)) / image_height;
      helper[2 * element + 1] =
        fmax (0.0, fmin (image_width - (patchsize_x - 1), px - offsetx)) / image_width;
    }
  }

  // FIXME make this configurable: KITTI needs it, LMF does not
  /*  for (unsigned int e = 0; e < target.elements(); e++) {
      target[e] -= 0.5;
      target[e] *= 2.0;
    }*/

  if (substract_mean) {
    const unsigned int elements_per_sample = patchsize_x * patchsize_y * image_maps;

    // substract mean
    for (unsigned int sample = 0; sample < target.samples(); sample++) {
      // Add up elements
      datum sum = 0;
      for (unsigned int e = 0; e < elements_per_sample; e++) {
        sum += target[sample * elements_per_sample + e];
      }

      // Calculate mean
      const datum mean = sum / (datum) elements_per_sample;

      // Substract mean
      for (unsigned int e = 0; e < elements_per_sample; e++) {
        target[sample * elements_per_sample + e] -= mean;
      }
    }
  }
}

void Segmentation::ExtractLabels (const int patchsize_x,
                                  const int patchsize_y,
                                  Tensor& labels, Tensor& weight,
                                  const Tensor& source,
                                  const int source_sample,
                                  const int ignore_class) {

  int image_width = (int)source.width();
  int image_height = (int)source.height();

  // new version
  int npatches = image_width * image_height;
  int offsetx = (patchsize_x / 2);
  int offsety = (patchsize_y / 2);

  labels.Resize ((const size_t)npatches, 1, 1, 1);
  weight.Resize ((const size_t)npatches);

  for (int px = 0; px < image_width; px++) {
    for (int py = 0; py < image_height; py++) {
      unsigned int element =  (unsigned int)(image_width * py + px);
      const int ipy = (patchsize_y / 2) + 1;
      int image_y = py + ipy - offsety;
      if (image_y < 0)
        image_y = -image_y;
      if (image_y >= image_height)
        image_y = -1 + image_height + image_height - image_y;

      int ipx = (patchsize_x / 2) + 1;
      int image_x = px + ipx - offsetx;
      if (image_x < 0)
        image_x = -image_x;
      if (image_x >= image_width)
        image_x = -1 + image_width + image_width - image_x;

      // Copy pixel
      const duint nlabel = * ( (duint*) source.data_ptr_const ((const size_t)image_x, (const size_t)image_y, 0,
                                                               (const size_t)source_sample));
      *labels.data_ptr (0, 0, 0, element) =
        *source.data_ptr_const ((const size_t)image_x, (const size_t)image_y, 0, (const size_t)source_sample);

      // Assign weight
      if (nlabel != (duint)ignore_class)
        *weight.data_ptr (0, 0, 0, element) = 1.0;
      else
        *weight.data_ptr (0, 0, 0, element) = 0.0;
    }
  }
}

}