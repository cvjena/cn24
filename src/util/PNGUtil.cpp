/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <iostream>

#ifdef BUILD_PNG
#include <png.h>
#endif

#include "Config.h"
#include "Log.h"
#include "Tensor.h"
#include "PNGUtil.h"

namespace Conv {

bool PNGUtil::LoadFromStream (std::istream& stream, Tensor& tensor) {
#ifndef BUILD_PNG
  LOGERROR << "PNG is not supported by this build!";
  return false;
#else
  // Check the header for a valid signature
  if (!CheckSignature (stream)) {
    LOGERROR << "PNG signature invalid!";
    return false;
  }

  // Create a file handle with libpng
  png_struct* png_handle = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL,
                           NULL, NULL);

  // Check the pointer
  if (!png_handle) {
    LOGERROR << "libpng did not create a read structure";
    return false;
  }

  // Create a pointer to an info structure
  png_info* png_info_handle = png_create_info_struct (png_handle);

  // Check this pointer too
  if (!png_info_handle) {
    LOGERROR << "libpng did not create an info structure";

    // Destroy the handle
    png_destroy_read_struct (&png_handle, 0, 0);
    return false;
  }

  // libpng cannot read streams, so we have to provide our own function
  png_set_read_fn (png_handle, (png_voidp) &stream, ReadFromStream);

  // Read image information
  png_read_info (png_handle, png_info_handle);

  png_uint_32 image_width = png_get_image_width (png_handle, png_info_handle);
  png_uint_32 image_height = png_get_image_height (png_handle, png_info_handle);
  png_uint_32 image_depth = png_get_bit_depth (png_handle, png_info_handle);
  png_uint_32 image_channels = png_get_channels (png_handle, png_info_handle);
  png_uint_32 image_colors = png_get_color_type (png_handle, png_info_handle);

  // Check header data for correct format
  if (image_depth != 8) {
    LOGERROR << "Only 8 bits per channel are supported! This image has "
      << image_depth;
    png_destroy_read_struct (&png_handle, &png_info_handle, 0);
    return false;
  }

  if (image_colors & PNG_COLOR_MASK_PALETTE) {
    LOGERROR << "Unsupported color type: " << image_colors;
    return false;
  }


  // Allocate memory for row pointers
  png_bytep* row_pointers = new png_bytep[image_height];

  // Allocate memory for image
  png_byte* image_data = new png_byte[image_height * image_width *
                                      image_channels];

  // Set row pointers
  const png_uint_32 row_stride = image_width * image_channels;
  for (png_uint_32 row = 0; row < image_height; row++) {
    row_pointers[row] = &image_data[row * row_stride];
  }

  png_read_image (png_handle, row_pointers);

  delete[] row_pointers;
  png_destroy_read_struct (&png_handle, &png_info_handle, 0);

  // Read image into Tensor
  tensor.Resize (1, image_width, image_height, image_channels);
  datum* target = tensor.data_ptr();

  // We need to realign the color data because our tensor channels are separate
  // Also we need to convert from unsigned char to our custom datum type
  for (std::size_t channel = 0; channel < image_channels; channel++) {
    for (std::size_t y = 0; y < image_height; y++) {
      std::size_t rc_offset = tensor.Offset (0, y, channel, 0);
      for (std::size_t x = 0; x < image_width; x++) {
        png_byte pixel = image_data[ (image_width * image_channels * y) +
                                     (image_channels * x) + channel];
        const datum v = DATUM_FROM_UCHAR (pixel);
        target[rc_offset + x] = v;
      }
    }
  }
  
  // Free image data
  delete[] image_data;

  return true;
#endif
}

bool PNGUtil::WriteToStream (std::ostream& stream, Tensor& tensor) {
  if(tensor.samples() != 1) {
    LOGERROR << "Cannot write PNGs with more than 1 sample!";
    return false;
  }
  if(tensor.maps() != 3) {
    LOGERROR << "Cannot write PNGs with channels != 3";
    return false;
  }
  
  return false;
}

#ifdef BUILD_PNG

bool PNGUtil::CheckSignature (std::istream& stream) {
  // Allocate 8 bytes for the PNG signature
  png_byte signature[8];

  // Read the first 8 bytes from the stream
  stream.read ( (char*) signature, 8);

  // Check if the read worked
  if (!stream.good()) {
    LOGERROR << "Could not read from the stream!";
    return false;
  }

  // Rewind the stream
  stream.seekg (0);

  // Check the signature
  int comparison_result = png_sig_cmp (signature, 0, 8);

  return comparison_result == 0;
}

void PNGUtil::ReadFromStream (png_structp png_handle, png_bytep data,
                                png_size_t length) {
  // We need our stream back
  png_voidp stream_ptr = png_get_io_ptr (png_handle);

  // Error checking is useless here :(
  ( (std::istream*) stream_ptr) -> read ( (char*) data, length);
}

#endif

}
