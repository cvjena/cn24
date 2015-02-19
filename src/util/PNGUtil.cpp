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

/*
 * PNG-specific declarations should not be in public headers, so they are here
 */

/** 
 * @brief Check if the provided stream contains a valid PNG file.
 *
 * @param stream Input stream to read from
 * @returns True if the file is a valid PNG file, false otherwise
 */
bool CheckSignature (std::istream& stream);

/**
 * @brief This function is needed because libPNG doesn't support streams.
 */
#ifdef BUILD_PNG
void PNGReadFromStream (png_structp png_handle, png_bytep data,
                            png_size_t length);
#endif

/**
 * @brief This function is needed because libPNG doesn't support streams.
 */
#ifdef BUILD_PNG
void PNGWriteToStream (png_structp png_handle, png_bytep data, png_size_t length);
#endif

bool PNGUtil::LoadFromStream ( std::istream& stream, Tensor& tensor ) {
#ifndef BUILD_PNG
  LOGERROR << "PNG is not supported by this build!";
  return false;
#else

  // Check the header for a valid signature
  if ( !CheckSignature ( stream ) ) {
    LOGERROR << "PNG signature invalid!";
    return false;
  }

  // Create a file handle with libpng
  png_struct* png_handle = png_create_read_struct ( PNG_LIBPNG_VER_STRING, NULL,
                           NULL, NULL );

  // Check the pointer
  if ( !png_handle ) {
    LOGERROR << "libpng did not create a read structure";
    return false;
  }

  // Create a pointer to an info structure
  png_info* png_info_handle = png_create_info_struct ( png_handle );

  // Check this pointer too
  if ( !png_info_handle ) {
    LOGERROR << "libpng did not create an info structure";

    // Destroy the handle
    png_destroy_read_struct ( &png_handle, 0, 0 );
    return false;
  }

  // libpng cannot read streams, so we have to provide our own function
  png_set_read_fn ( png_handle, ( png_voidp ) &stream, PNGReadFromStream );

  // Read image information
  png_read_info ( png_handle, png_info_handle );

  png_uint_32 image_width = png_get_image_width ( png_handle, png_info_handle );
  png_uint_32 image_height = png_get_image_height ( png_handle, png_info_handle );
  png_uint_32 image_depth = png_get_bit_depth ( png_handle, png_info_handle );
  png_uint_32 image_channels = png_get_channels ( png_handle, png_info_handle );
  png_uint_32 image_colors = png_get_color_type ( png_handle, png_info_handle );

  // Check header data for correct format
  if ( image_depth != 8 && image_depth != 16 ) {
    LOGERROR << "Only 8/16 bits per channel are supported! This image has "
             << image_depth;
    png_destroy_read_struct ( &png_handle, &png_info_handle, 0 );
    return false;
  }

  if ( image_colors & PNG_COLOR_MASK_PALETTE ) {
    LOGERROR << "Unsupported color type: " << image_colors;
    return false;
  }


  if(image_depth == 8) {
    // Allocate memory for row pointers
    png_bytep* row_pointers = new png_bytep[image_height];

    // Allocate memory for image
    png_byte* image_data = new png_byte[image_height * image_width *
                                        image_channels];

    // Set row pointers
    const png_uint_32 row_stride = image_width * image_channels;

    for ( png_uint_32 row = 0; row < image_height; row++ ) {
      row_pointers[row] = &image_data[row * row_stride];
    }

    png_read_image ( png_handle, row_pointers );

    delete[] row_pointers;
    png_destroy_read_struct ( &png_handle, &png_info_handle, 0 );

    // Read image into Tensor
    tensor.Resize ( 1, image_width, image_height, image_channels );
    datum* target = tensor.data_ptr();

    // We need to realign the color data because our tensor channels are separate
    // Also we need to convert from unsigned char to our custom datum type
    for ( std::size_t channel = 0; channel < image_channels; channel++ ) {
      for ( std::size_t y = 0; y < image_height; y++ ) {
        std::size_t rc_offset = tensor.Offset ( 0, y, channel, 0 );

        for ( std::size_t x = 0; x < image_width; x++ ) {
          png_byte pixel = image_data[ ( image_width * image_channels * y ) +
                                       ( image_channels * x ) + channel];
          const datum v = DATUM_FROM_UCHAR ( pixel );
          target[rc_offset + x] = v;
        }
      }
    }

    // Free image data
    delete[] image_data;

  } else if (image_depth == 16) {
    // Allocate memory for row pointers
    png_uint_16pp row_pointers = new png_uint_16*[image_height];

    // Allocate memory for image
    png_uint_16* image_data = new png_uint_16[image_height * image_width *
                                        image_channels];

    // Set row pointers
    const png_uint_32 row_stride = image_width * image_channels;

    for ( png_uint_32 row = 0; row < image_height; row++ ) {
      row_pointers[row] = &image_data[row * row_stride];
    }

    png_read_image ( png_handle, (png_bytepp)row_pointers );

    delete[] row_pointers;
    png_destroy_read_struct ( &png_handle, &png_info_handle, 0 );

    // Read image into Tensor
    tensor.Resize ( 1, image_width, image_height, image_channels );
    datum* target = tensor.data_ptr();

    // We need to realign the color data because our tensor channels are separate
    // Also we need to convert from unsigned char to our custom datum type
    for ( std::size_t channel = 0; channel < image_channels; channel++ ) {
      for ( std::size_t y = 0; y < image_height; y++ ) {
        std::size_t rc_offset = tensor.Offset ( 0, y, channel, 0 );

        for ( std::size_t x = 0; x < image_width; x++ ) {
          png_uint_16 pixel = image_data[ ( image_width * image_channels * y ) +
                                       ( image_channels * x ) + channel];
          pixel = (pixel >> 8) | (pixel << 8);
          const datum v = DATUM_FROM_USHORT ( pixel );
          target[rc_offset + x] = v;
        }
      }
    }

    // Free image data
    delete[] image_data;

  }

  return true;
#endif
}

bool PNGUtil::WriteToStream ( std::ostream& stream, Tensor& tensor ) {
#ifndef BUILD_PNG
  LOGERROR << "PNG is not supported by this build!";
  return false;
#else
  if ( tensor.samples() != 1 ) {
    LOGERROR << "Cannot write PNGs with more than 1 sample!";
    return false;
  }

  if ( tensor.maps() != 3 ) {
    LOGERROR << "Cannot write PNGs with channels != 3";
    return false;
  }

  png_structp png_handle = png_create_write_struct ( PNG_LIBPNG_VER_STRING, NULL,
                           NULL, NULL );

  // Check the pointer
  if ( !png_handle ) {
    LOGERROR << "libpng did not create a write structure";
    return false;
  }

  png_infop png_info_handle = png_create_info_struct ( png_handle );

  // Check this pointer too
  if ( !png_info_handle ) {
    LOGERROR << "libpng did not create an info structure";

    // Destroy the handle
    png_destroy_write_struct ( &png_handle, 0 );
    return false;
  }

  // libpng cannot write streams, so we have to provide our own function
  png_set_write_fn ( png_handle, ( png_voidp ) &stream, PNGWriteToStream, NULL );

  // Set and write information
  png_set_IHDR (
    png_handle,
    png_info_handle,
    tensor.width(), tensor.height(), 8,
    PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT );
  png_write_info ( png_handle, png_info_handle );

  // Make rows
  png_byte* pixels = new png_byte[3 * tensor.width() * tensor.height()];

  //...
  for(unsigned int y = 0; y < tensor.height(); y++) {
    for(unsigned int x = 0; x < tensor.width(); x++) {
      pixels[3 * tensor.width() * y + 3 * x + 0] = UCHAR_FROM_DATUM(
	  *tensor.data_ptr_const(x,y,0,0) );
      pixels[3 * tensor.width() * y + 3 * x + 1] = UCHAR_FROM_DATUM(
	  *tensor.data_ptr_const(x,y,1,0) );
      pixels[3 * tensor.width() * y + 3 * x + 2] = UCHAR_FROM_DATUM(
	  *tensor.data_ptr_const(x,y,2,0) );
    }
  }

  // Make row pointers
  png_byte** row_pointers = new png_byte*[tensor.height()];

  for ( unsigned int y = 0; y < tensor.height(); y++ )
    row_pointers[y] = &pixels[3*tensor.width() * y];
  
  png_write_image(png_handle, row_pointers);

  png_write_end ( png_handle, NULL );
  return true;
#endif
}

#ifdef BUILD_PNG

bool CheckSignature ( std::istream& stream ) {
  // Allocate 8 bytes for the PNG signature
  png_byte signature[8];

  // Read the first 8 bytes from the stream
  stream.read ( ( char* ) signature, 8 );

  // Check if the read worked
  if ( !stream.good() ) {
    LOGERROR << "Could not read from the stream!";
    return false;
  }

  // Rewind the stream
  stream.seekg ( 0 );

  // Check the signature
  int comparison_result = png_sig_cmp ( signature, 0, 8 );

  return comparison_result == 0;
}

void PNGReadFromStream ( png_structp png_handle, png_bytep data,
                                  png_size_t length ) {
  // We need our stream back
  png_voidp stream_ptr = png_get_io_ptr ( png_handle );

  // Error checking is useless here :(
  ( ( std::istream* ) stream_ptr ) -> read ( ( char* ) data, length );
}

void PNGWriteToStream ( png_structp png_handle, png_bytep data,
                                 png_size_t length ) {
  // We need our stream back
  png_voidp stream_ptr = png_get_io_ptr ( png_handle );

  // Error checking is useless here :(
  ( ( std::ostream* ) stream_ptr ) -> write ( ( char* ) data, length );
}


#endif

}
