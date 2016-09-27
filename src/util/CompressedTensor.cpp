/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <string>


#ifdef BUILD_POSIX
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#endif

#include "PNGUtil.h"
#include "JPGUtil.h"

#ifdef BLAS_MKL
#include <mkl_service.h>
#endif

#include "Config.h"
#include "Log.h"
#include "CompressedTensor.h"
#include "CLHelper.h"

namespace Conv {
  
const unsigned int chars_per_datum = sizeof(Conv::datum)/sizeof(char);

CompressedTensor::CompressedTensor() {

}

CompressedTensor::~CompressedTensor() {
  DeleteIfPossible();
}

void CompressedTensor::Compress(Tensor& tensor)
{
  std::size_t compressed_length = 0;
  std::size_t uncompressed_elements = tensor.elements();
#ifdef BUILD_OPENCL
  tensor.MoveToCPU();
#endif
  
  void* compressed_buffer = new char[2 * tensor.elements() * chars_per_datum + 2];
  CompressedTensor::CompressData((void*)tensor.data_ptr(), uncompressed_elements, compressed_buffer, compressed_length);
  
  Resize(tensor.samples(), tensor.width(), tensor.height(), tensor.maps(), compressed_length, (char*)compressed_buffer, false);
}

void CompressedTensor::Decompress(Tensor& tensor, datum* preallocated_buffer)
{
  std::size_t compressed_length = compressed_length_;
  std::size_t uncompressed_elements = 0;
  datum* uncompressed_buffer = preallocated_buffer;
  if(uncompressed_buffer == nullptr)
    uncompressed_buffer = new datum[elements_];
  
  CompressedTensor::DecompressData(uncompressed_buffer, uncompressed_elements, compressed_data_ptr_, compressed_length);
  
  if(uncompressed_elements != elements_) {
    FATAL("Decompressed size mismatch!");
  }
    
  tensor.Resize(samples_, width_, height_, maps_, uncompressed_buffer, false);
}


void CompressedTensor::Resize ( const std::size_t samples, const std::size_t width,
                      const std::size_t height, const std::size_t maps, const std::size_t compressed_length, char* const preallocated_memory, bool mmapped) {
  // Delete the old allocation
  DeleteIfPossible();

  // Don't need to allocate zero memory
  if ( compressed_length == 0 )
    return;

  if(preallocated_memory != nullptr) {
    compressed_data_ptr_ = preallocated_memory;
    mmapped_ = mmapped;
  } else {
    // Allocate
    compressed_data_ptr_ = new char[compressed_length];
  }

  // Save configuration
  samples_ = samples;
  width_ = width;
  height_ = height;
  maps_ = maps;
  elements_ = samples * width * height * maps;
  compressed_length_ = compressed_length;
}

void CompressedTensor::Serialize ( std::ostream& output ) {
  uint64_t samples = samples_;
  uint64_t width = width_;
  uint64_t height = height_;
  uint64_t maps = maps_;
  uint64_t compressed_length = compressed_length_;

  output.write ( ( const char* ) &samples, sizeof ( uint64_t ) / sizeof ( char ) );
  output.write ( ( const char* ) &width, sizeof ( uint64_t ) / sizeof ( char ) );
  output.write ( ( const char* ) &height, sizeof ( uint64_t ) / sizeof ( char ) );
  output.write ( ( const char* ) &maps, sizeof ( uint64_t ) / sizeof ( char ) );
  output.write ( ( const char* ) &compressed_length, sizeof ( uint64_t ) / sizeof ( char ) );

  if ( elements_ > 0 )
    output.write ( ( const char* ) compressed_data_ptr_, compressed_length_);
}

void CompressedTensor::Deserialize ( std::istream& input , bool head_only, bool try_mmap, int fd) {
  uint64_t samples = 0;
  uint64_t width = 0;
  uint64_t height = 0;
  uint64_t maps = 0;
  uint64_t compressed_length = 0;

  if ( !input.good() )
    LOGERROR << "Cannot deserialize from this stream!";

  input.read ( ( char* ) &samples, sizeof ( uint64_t ) / sizeof ( char ) );
  input.read ( ( char* ) &width, sizeof ( uint64_t ) / sizeof ( char ) );
  input.read ( ( char* ) &height, sizeof ( uint64_t ) / sizeof ( char ) );
  input.read ( ( char* ) &maps, sizeof ( uint64_t ) / sizeof ( char ) );
  input.read ( ( char* ) &compressed_length, sizeof ( uint64_t ) / sizeof ( char ) );

#ifdef BUILD_POSIX
  if(!try_mmap || fd == 0)
#endif
    Resize ( samples, width, height, maps, compressed_length );
    
  if ( compressed_length > 0 && !head_only ) {
#ifdef BUILD_POSIX
    if(try_mmap && fd != 0) {
      // Get page size
      long int page_size = sysconf(_SC_PAGESIZE);
      long int current_position = input.tellg();
      long int offset_in_page = current_position % page_size;
#ifdef BUILD_LINUX
      void* target_mmap = mmap64(NULL, compressed_length + offset_in_page, PROT_READ, MAP_PRIVATE, fd, current_position - offset_in_page);
#elif defined(BUILD_OSX)
      // OS X is 64-bit by default
      void* target_mmap = mmap(NULL, compressed_length + offset_in_page, PROT_READ, MAP_PRIVATE, fd, current_position - offset_in_page);
#endif
      if(target_mmap == MAP_FAILED) {
        LOGERROR << "Memory map failed: " << errno;
      }
      original_mmap_ = target_mmap;
      
      target_mmap = (void*)(((long)target_mmap) + offset_in_page);
      Resize(samples, width, height, maps, compressed_length, (char*)target_mmap, true);
      input.seekg(compressed_length, std::ios::cur);
    } else
#endif
      input.read ( ( char* ) compressed_data_ptr_, compressed_length);
  }
  else if(head_only)
    input.seekg(compressed_length, std::ios::cur);
}

void CompressedTensor::DeleteIfPossible() {
  if ( compressed_data_ptr_ != nullptr ) {
#ifdef BUILD_POSIX
    if(mmapped_) {
      munmap((void*)original_mmap_, compressed_length_);
      original_mmap_ = nullptr;
      mmapped_ = false;
    } else {
#endif
      delete[] compressed_data_ptr_;
#ifdef BUILD_POSIX
    }
#endif

    compressed_data_ptr_ = nullptr;
  }

  samples_ = 0;
  width_ = 0;
  height_ = 0;
  maps_ = 0;
  elements_ = samples_ * width_ * height_ * maps_;
  compressed_length_ = 0;
}

void CompressedTensor::PrintStats() {

}

std::ostream& operator<< ( std::ostream& output, const CompressedTensor& tensor ) {
  return output << "C(" << tensor.samples() << "s@" << tensor.width() <<
         "x" << tensor.height() << "x" << tensor.maps() << "m)";
}

/*
 * This is the compression part. Don't change this or you will break the file format.
 */
const unsigned char rl_marker = 'X';
const unsigned char rl_doublemarker = 'X';
const unsigned char rl_rle = 'Y';
const unsigned int rl_bytes = 1;
const unsigned int rl_max = (unsigned int)((1L << (8L * (unsigned long)rl_bytes)) - 3L);
const unsigned int rl_min = 1 + (5 + rl_bytes) / chars_per_datum;

void CompressedTensor::CompressData(void* uncompressed, const std::size_t& uncompressed_elements, void* compressed, std::size_t& compressed_length)
{
  std::size_t bytes_out = 0;
  
  Conv::datum last_symbol = 0;
  std::size_t running_length = 0;
  
  unsigned char* output_ptr = (unsigned char*)compressed;
  
  const datum* data_ptr_const = (const datum*) uncompressed;
  
  for(std::size_t pos = 0; pos <= uncompressed_elements; pos++) {
    Conv::datum current_symbol = 0;
    if(pos < uncompressed_elements) {
      current_symbol = data_ptr_const[pos];
      if(current_symbol == last_symbol) {
        // Increase running length
        running_length++;
      }
    } else {
      // Force emission of last symbol
    }
    
    
    if(
    // EOF reached
    (pos == uncompressed_elements) ||
    // Different symbol
    (current_symbol != last_symbol) ||
    // Maxmimum run length reached
    (running_length == rl_max)) {
        
      // Emit...
      if(running_length > 0 && running_length < rl_min) {
        // Emit single symbol(s)
        for(std::size_t r = 0; r < running_length; r++) {
          for(std::size_t b = 0; b < chars_per_datum; b++) {
            char char_to_emit = ((char*)&last_symbol)[b];
            if(char_to_emit == rl_marker) {
              // Emit escaped
              *output_ptr = rl_marker;
              output_ptr++; bytes_out++;
              *output_ptr = rl_doublemarker;
              output_ptr++; bytes_out++;
            } else {
              // Emit directly
              *output_ptr = char_to_emit;
              output_ptr++; bytes_out++;
            }
          }
        }
      } else if(running_length >= rl_min) {
        // Emit encoded
        *output_ptr = rl_marker;
        output_ptr++; bytes_out++;
        *output_ptr = rl_rle;
        output_ptr++; bytes_out++;
        
        // Running length output
        for(std::size_t b = 0; b < rl_bytes; b++) {
          *output_ptr = (running_length >> ((rl_bytes - (b+1)) * 8)) & 0xFF;
          output_ptr++; bytes_out++;
        }
        
        for(std::size_t b = 0; b < chars_per_datum; b++) {
          unsigned char char_to_emit = ((char*)&last_symbol)[b];
          *output_ptr = char_to_emit;
          output_ptr++; bytes_out++;
        }
      }
        
      // ...and reset
      if(running_length == rl_max)
        running_length = 0;
      else
        running_length = 1;
    }
      
    last_symbol = current_symbol;
  }
  compressed_length = bytes_out;
}

void CompressedTensor::DecompressData(void* uncompressed, std::size_t& uncompressed_elements, void* compressed, const std::size_t& compressed_length)
{
  unsigned int bytes_out = 0;
  unsigned char* output_ptr = (unsigned char*)uncompressed;
  const unsigned char* input_ptr = (const unsigned char*)compressed;
  
  for(unsigned int pos = 0; pos < compressed_length; pos++) {
    unsigned char current_symbol = input_ptr[pos];
    if(current_symbol == rl_marker) {
      pos++; current_symbol = input_ptr[pos];
      if(current_symbol == rl_doublemarker) {
        // Emit single marker
        *output_ptr = rl_marker;
        output_ptr++; bytes_out++;
      } else if(current_symbol == rl_rle) {
        unsigned int running_length = 0;
        
        // Running length input
        for(unsigned int b = 0; b < rl_bytes; b++) {
          pos++; current_symbol = input_ptr[pos];
          running_length += current_symbol;
          if((b+1) != rl_bytes)
            running_length <<= 8;
        }
        
        for(unsigned int r = 0; r < running_length; r++) {
          for(unsigned int b = 0; b < chars_per_datum; b++) {
            pos++; current_symbol = input_ptr[pos];
            *output_ptr = current_symbol;
            output_ptr++; bytes_out++;
          }
          pos -= chars_per_datum;
        }
        pos += chars_per_datum;
      } else {
        FATAL("Incorrect encoding!");
      }
    } else {
      // Emit directly
      *output_ptr = current_symbol;
      output_ptr++; bytes_out++;
    }
  }
  if(bytes_out % chars_per_datum != 0) {
    FATAL("Compressed length wrong!");
  }
  uncompressed_elements = bytes_out / chars_per_datum;
}


}
