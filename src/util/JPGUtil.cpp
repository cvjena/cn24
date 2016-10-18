/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <iostream>
#include <string>
#include <cstdio>

#ifdef BUILD_JPG
#include <jpeglib.h>
#endif

#include "Config.h"
#include "Log.h"
#include "Tensor.h"
#include "JPGUtil.h"

namespace Conv {
#ifdef BUILD_JPG
void dont_do_anything(j_common_ptr p) {
  UNREFERENCED_PARAMETER(p);
}
#endif

bool JPGUtil::LoadFromFile (const std::string& file, Tensor& tensor) {
#ifndef BUILD_JPG
  LOGERROR << "JPG is not supported by this build!";
  return false;
#else
  
  // Create decompression object
  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.err->error_exit = dont_do_anything;
  jpeg_create_decompress(&cinfo);
  
  FILE* in_file = fopen(file.c_str(), "rb");
  if(in_file == NULL) {
    LOGERROR << "Cannot open " << file;
    return false;
  }
  
  jpeg_stdio_src(&cinfo, in_file);
  int ret_val = jpeg_read_header(&cinfo, true);
  if(ret_val != JPEG_HEADER_OK) {
    LOGERROR << "Not a JPEG file: " << file;
    return false;
  }
  
  jpeg_start_decompress(&cinfo);

  unsigned int image_width = cinfo.output_width;
  unsigned int image_height = cinfo.output_height;
  unsigned int image_channels = cinfo.output_components;

  JSAMPARRAY samples = (cinfo.mem->alloc_sarray)
  ((j_common_ptr)&cinfo, JPOOL_IMAGE, image_width * image_channels, 1);
  
  tensor.Resize(1, image_width, image_height, image_channels);
  
  unsigned int current_line = 0;
  while(cinfo.output_scanline < cinfo.output_height) {
    const unsigned int addition = jpeg_read_scanlines(&cinfo, samples, 1);
    for(unsigned int c = 0; c < image_channels; c++) {
      for(unsigned int x = 0; x < image_width; x++) {
        datum* target = tensor.data_ptr(x, current_line, c);
        JSAMPLE samp = samples[0][(x * image_channels) + c];
        const datum val = DATUM_FROM_UCHAR(samp);
        *target = val;
      }
    }
    current_line += addition;
  }
  
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(in_file);
 
  return true;
#endif
}

bool JPGUtil::WriteToFile ( const std::string& file, Tensor& tensor ) {
#ifndef BUILD_JPG
  LOGERROR << "JPG is not supported by this build!";
  return false;
#else
  FILE* out_file = fopen(file.c_str(), "wb");
  if(out_file == NULL) {
    LOGERROR << "Cannot open " << file;
    return false;
  }
  
  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;
  
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, out_file);
  
  cinfo.image_width = tensor.width();
  cinfo.image_height = tensor.height();
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, true);
  jpeg_start_compress(&cinfo, true);
  
  
  JSAMPROW row_pointer = new JSAMPLE[3*tensor.width()];
  // Write scanlines
  while(cinfo.next_scanline < tensor.height()) {
    // Copy scanline
    for(unsigned int x = 0; x < tensor.width(); x++) {
      row_pointer[3 * x + 0] = UCHAR_FROM_DATUM(*tensor.data_ptr(x,cinfo.next_scanline,0,0));
      row_pointer[3 * x + 1] = UCHAR_FROM_DATUM(*tensor.data_ptr(x,cinfo.next_scanline,1,0));
      row_pointer[3 * x + 2] = UCHAR_FROM_DATUM(*tensor.data_ptr(x,cinfo.next_scanline,2,0));
    }
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }
  
  jpeg_finish_compress(&cinfo);
  fclose(out_file);
  return true;
#endif
}


bool JPGUtil::CheckSignature (std::istream& stream) {
  UNREFERENCED_PARAMETER(stream);
  return true;
}

}
