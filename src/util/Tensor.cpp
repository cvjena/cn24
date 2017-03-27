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
#include <sstream>
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
#include "Tensor.h"
#include "TensorRegistry.h"
#include "CLHelper.h"

#include "base64_default_rfc4648.hpp"

namespace Conv {

Tensor::Tensor(bool ignore, const std::string& owner) : owner(owner) {
  UNREFERENCED_PARAMETER(ignore);
  if(System::registry != nullptr) {
    System::registry->RegisterTensor(this);
  }
  construction = "Default";
}

Tensor::Tensor ( const std::string& filename, const std::string& owner ) : owner(owner) {
  if(System::registry != nullptr) {
    System::registry->RegisterTensor(this);
  }
  LoadFromFile ( filename );
  construction = "File";
}


Tensor::Tensor ( const Tensor& tensor, bool intentional, const std::string& owner ) : owner(owner) {
  if(System::registry != nullptr) {
    System::registry->RegisterTensor(this);
  }
  if(intentional) {
    construction = "Copy";
  } else {
    construction = "Unint. copy";
  }
  
  // Match size of source Tensor
  Resize ( tensor );

  // Get pointers
  const datum* source_data = tensor.data_ptr_const();
  datum* target_data = data_ptr();

  // Count copy size
  std::size_t bytes_to_copy = tensor.elements() * sizeof ( datum );

  // Copy
  std::memcpy ( target_data, source_data, bytes_to_copy );

  if ( !intentional ) {
    LOGDEBUG << "Tensor copied! Is this intentional?";
  }
}

Tensor::Tensor ( Tensor && tensor, const std::string& owner ) : owner(owner) {
  if(System::registry != nullptr) {
    System::registry->RegisterTensor(this);
  }
  construction = "Move";
#ifdef BUILD_OPENCL
  tensor.MoveToCPU();
#endif
  
  data_ptr_ = tensor.data_ptr_;
  samples_ = tensor.samples_;
  maps_ = tensor.maps_;
  width_ = tensor.width_;
  height_ = tensor.height_;
  elements_ = tensor.elements_;

  tensor.data_ptr_ = nullptr;
  tensor.DeleteIfPossible();
}

Tensor::Tensor ( const std::size_t samples, const std::size_t width,
                 const std::size_t height, const std::size_t maps, const std::string& owner ) : owner(owner) {
  if(System::registry != nullptr) {
    System::registry->RegisterTensor(this);
  }
  construction = "Size";
  Resize ( samples, width, height, maps );
}


Tensor::~Tensor() {
  DeleteIfPossible();
  if(System::registry != nullptr) {
    System::registry->DeregisterTensor(this);
  }
}

void Tensor::Clear ( const datum value, const int sample ) {
#ifdef BUILD_OPENCL
  if ( sample == -1 ) {
    MoveToCPU(true);
  } else {
    MoveToCPU();
  }
#endif
  if ( sample == -1 ) {
    for ( std::size_t element = 0; element < elements_; element++ ) {
      data_ptr_[element] = value;
    }
  } else {
    for ( std::size_t element = ( width_ * height_ * maps_ * sample );
          element < ( width_* height_ * maps_ * ( sample+1 ) ); element++ ) {
      data_ptr_[element] = value;
    }
  }
}

void Tensor::Shadow ( Tensor& tensor ) {
  DeleteIfPossible();

  data_ptr_ = tensor.data_ptr_;
  samples_ = tensor.samples_;
  maps_ = tensor.maps_;
  width_ = tensor.width_;
  height_ = tensor.height_;
  elements_ = tensor.elements_;

  is_shadow_ = true;
  shadow_target_ = &tensor;

#ifdef BUILD_OPENCL
  shadow_target_->MoveToGPU();
  shadow_target_->MoveToCPU();
  cl_data_ptr_ = shadow_target_->cl_data_ptr_;
#endif
}


void Tensor::Resize ( const std::size_t samples, const std::size_t width,
                      const std::size_t height, const std::size_t maps, datum* const preallocated_memory, bool mmapped, bool dont_delete) {
  // Check if reshaping works
  if (preallocated_memory == nullptr && Reshape ( samples, width, height, maps ) )
    return;

  // Delete the old allocation if it is different from the new one
  if(preallocated_memory != data_ptr_ && !dont_delete)
    DeleteIfPossible();

  // Calculate memory requirement
  std::size_t elements = samples * maps * width * height;

  // Don't need to allocate zero memory
  if ( elements == 0 )
    return;

  if(preallocated_memory != nullptr) {
    data_ptr_ = preallocated_memory;
    mmapped_ = mmapped;
  } else {
    // Allocate
#ifdef BLAS_MKL
    data_ptr_ = ( datum* ) MKL_malloc ( elements * sizeof ( datum ) / sizeof ( char ), 32 );
#else
    data_ptr_ = new datum[elements];
#endif
  }

  // Save configuration
  samples_ = samples;
  width_ = width;
  height_ = height;
  maps_ = maps;
  elements_ = elements;
}

void Tensor::Extend(const std::size_t samples) {
  if(samples <= samples_) {
    LOGWARN << "Tried to shrink Tensor!";
    return;
  }

#ifdef BUILD_OPENCL
  MoveToCPU(true);
#endif

  // Make backup
  Tensor temp_tensor;
  temp_tensor.Resize(*this);
  if(!Tensor::Copy(*this, temp_tensor)) {
    FATAL("Failed to backup Tensor before extension!")
  }

  Resize(samples, width_, height_, maps_);
  Clear();

  // Restore data
  for(unsigned int s = 0; s < temp_tensor.samples(); s++) {
    if(!Tensor::CopySample(temp_tensor, s, *this, s)) {
      FATAL("Failed to restore Tensor after extension!");
    }
  }
}
void Tensor::Resize ( const Tensor& tensor ) {
  Resize ( tensor.samples(), tensor.width(), tensor.height(), tensor.maps() );
}


bool Tensor::Reshape ( const std::size_t samples, const std::size_t width,
                       const std::size_t height, const std::size_t maps ) {
  // Check for null pointer
  if ( data_ptr_ == nullptr )
    return false;

  // Check if element count matches
  std::size_t proposed_elements = samples * maps * width * height;

  if ( elements_ != proposed_elements )
    return false;

  // Reshape the Tensor
  samples_ = samples;
  width_ = width;
  height_ = height;
  maps_ = maps;
  elements_ = proposed_elements;

  return true;
}


void Tensor::Transpose() {
  if ( data_ptr_ == nullptr )
    return;

  // This copy _is_ intentional
  Tensor tmp ( *this, true );

  if ( !Reshape ( samples_, height_, width_, maps_ ) )
    FATAL ( "Didn't reshape!" );

  // This could be optimized
  for ( std::size_t s = 0; s < samples_; s++ ) {
    for ( std::size_t m = 0; m < maps_; m++ ) {
      for ( std::size_t x = 0; x < width_; x++ ) {
        for ( std::size_t y = 0; y < height_; y++ ) {
          *data_ptr ( x, y, m, s ) = *tmp.data_ptr_const ( y, x, m, s );
        }
      }
    }
  }
}


void Tensor::Serialize ( std::ostream& output, bool convert ) {
#ifdef BUILD_OPENCL
  MoveToCPU();
#endif

  if ( convert ) {
    if ( maps_ == 3 ) {
      std::size_t ime_count = width_ * height_;
      std::size_t is_count = ime_count * maps_;

      for ( std::size_t sample = 0; sample < samples_; sample++ ) {
        for ( std::size_t ime = 0; ime < ime_count; ime++ ) {
          unsigned char r = MCHAR_FROM_DATUM (
                              data_ptr_[is_count * sample + ime] );
          unsigned char g = MCHAR_FROM_DATUM (
                              data_ptr_[is_count * sample + ime_count * 1 + ime] );
          unsigned char b = MCHAR_FROM_DATUM (
                              data_ptr_[is_count * sample + ime_count * 2 + ime] );
          output.write ( ( const char* ) &r, 1 );
          output.write ( ( const char* ) &g, 1 );
          output.write ( ( const char* ) &b, 1 );
        }
      }
    } else {
      for ( std::size_t e = 0; e < elements_; e++ ) {
        unsigned char x = MCHAR_FROM_DATUM ( data_ptr_[e] );
        output.write ( ( const char* ) &x, 1 );
      }
    }

  } else {
    uint64_t samples = samples_;
    uint64_t width = width_;
    uint64_t height = height_;
    uint64_t maps = maps_;

    output.write ( ( const char* ) &samples, sizeof ( uint64_t ) / sizeof ( char ) );
    output.write ( ( const char* ) &width, sizeof ( uint64_t ) / sizeof ( char ) );
    output.write ( ( const char* ) &height, sizeof ( uint64_t ) / sizeof ( char ) );
    output.write ( ( const char* ) &maps, sizeof ( uint64_t ) / sizeof ( char ) );

    if ( elements_ > 0 )
      output.write ( ( const char* ) data_ptr_, ( elements_ * sizeof ( datum ) )
                     / sizeof ( char ) );
  }
}

void Tensor::Deserialize ( std::istream& input , bool head_only, bool try_mmap, int fd) {
#ifdef BUILD_OPENCL
  MoveToCPU ( true );
#endif
  uint64_t samples = 0;
  uint64_t width = 0;
  uint64_t height = 0;
  uint64_t maps = 0;

  if ( !input.good() )
    LOGERROR << "Cannot deserialize from this stream!";

  input.read ( ( char* ) &samples, sizeof ( uint64_t ) / sizeof ( char ) );
  input.read ( ( char* ) &width, sizeof ( uint64_t ) / sizeof ( char ) );
  input.read ( ( char* ) &height, sizeof ( uint64_t ) / sizeof ( char ) );
  input.read ( ( char* ) &maps, sizeof ( uint64_t ) / sizeof ( char ) );

#ifdef BUILD_POSIX
  if(!try_mmap || fd == 0)
#endif
    Resize ( samples, width, height, maps );
  
  std::size_t elements = samples * maps * width * height;

  if ( elements > 0 && !head_only ) {
#ifdef BUILD_POSIX
    if(try_mmap && fd != 0) {
      // Get page size
      long int page_size = sysconf(_SC_PAGESIZE);
      long int current_position = input.tellg();
      long int offset_in_page = current_position % page_size;
#ifdef BUILD_LINUX
      void* target_mmap = mmap64(NULL,((elements* sizeof(datum)) / sizeof(char)) + offset_in_page, PROT_READ, MAP_PRIVATE, fd, current_position - offset_in_page);
#elif defined(BUILD_OSX)
      // OS X is 64-bit by default
      void* target_mmap = mmap(NULL,((elements* sizeof(datum)) / sizeof(char)) + offset_in_page, PROT_READ, MAP_PRIVATE, fd, current_position - offset_in_page);
#endif
      if(target_mmap == MAP_FAILED) {
        LOGERROR << "Memory map failed: " << errno;
      }
      original_mmap_ = target_mmap;
      
      target_mmap = (void*)(((long)target_mmap) + offset_in_page);
      Resize(samples, width, height, maps, (datum*)target_mmap, true);
      input.seekg(( elements * sizeof ( datum ) ) / sizeof ( char ) , std::ios::cur);
    } else
#endif
      input.read ( ( char* ) data_ptr_, ( elements * sizeof ( datum ) )
                  / sizeof ( char ) );
  }
  else if(head_only)
    input.seekg(( elements * sizeof ( datum ) ) / sizeof ( char ) , std::ios::cur);
      
}

bool Tensor::Copy (const Tensor& source, Tensor& target) {
  if(source.elements() != target.elements()) {
    return false;
  }
#ifdef BUILD_OPENCL
  if(source.cl_gpu_ || target.cl_gpu_) {
    ((Tensor&)source).MoveToGPU();
    target.MoveToGPU(true);
    cl_int error_ret = clEnqueueCopyBuffer(CLHelper::queue, (cl_mem)source.cl_data_ptr_, (cl_mem)target.cl_data_ptr_, 0, 0, source.elements() * sizeof(datum), 0, NULL, NULL);
    if ( error_ret != CL_SUCCESS ) {
      FATAL ( "Error moving to GPU: " << error_ret );
    }
#ifdef BRUTAL_FINISH
    error_ret = clFinish (CLHelper::queue);
    if (error_ret != CL_SUCCESS) {
      FATAL("Error finishing command queue: " << (signed int) error_ret);
    }
#endif
    return true;
  } else {
#endif
    std::memcpy((void*)target.data_ptr(), (const void*)source.data_ptr_const(), source.elements() * sizeof(datum));
    return true; // std::memcpy does not return an error, maybe use try/catch?
#ifdef BUILD_OPENCL
  }
#endif
}

bool Tensor::CopySample ( const Tensor& source, const std::size_t source_sample,
                          Tensor& target, const std::size_t target_sample, const bool allow_oversize, const bool scale) {
  // Check if both Tensors have the same amount of feature maps/channels
  if ( source.maps() != target.maps() ) {
    if( source.maps() != 1 || target.maps() != 3) {
      return false;
    }
  }

  if ( source.width() != target.width() || source.height() != target.height() ) {
    if ( (target.width() < source.width() || target.height() < source.height()) && !allow_oversize && !scale)
      return false;
  }

  bool result = true;

  if(source.maps() == 1 && target.maps() == 3) {
      result &= CopyMap ( source, source_sample, 0,
                          target, target_sample, 0, allow_oversize, scale );
      result &= CopyMap ( source, source_sample, 0,
                          target, target_sample, 1, allow_oversize, scale );
      result &= CopyMap ( source, source_sample, 0,
                          target, target_sample, 2, allow_oversize, scale );
  } else {
    for ( std::size_t map = 0; map < source.maps(); map++ ) {
      result &= CopyMap ( source, source_sample, map,
                          target, target_sample, map, allow_oversize, scale );
    }
  }

  return result;
}

bool Tensor::CopyMap ( const Tensor& source, const std::size_t source_sample,
                       const std::size_t source_map, Tensor& target,
                       const std::size_t target_sample,
                       const std::size_t target_map, const bool allow_oversize, const bool scale) {
  // Check sample bounds
  if ( source_sample >= source.samples() || target_sample >= target.samples() )
    return false;

  // Check if image dimensions match
  if ( source.width() != target.width() || source.height() != target.height() ) {
    if ( (target.width() < source.width() || target.height() < source.height() ) && !allow_oversize && !scale)
      return false;

    if(!scale) {
      // Source image is smaller, okay..
      for (unsigned int y = 0; (y < source.height() && y < target.height()); y++) {
        for (unsigned int x = 0; (x < source.width() && y < target.width()); x++) {
          *target.data_ptr(x, y, target_map, target_sample) =
              *source.data_ptr(x, y, source_map, source_sample);
        }

        for (unsigned int x = source.width(); x < target.width(); x++) {
          *target.data_ptr(x, y, target_map, target_sample) = 0;
        }
      }

      for (unsigned int y = source.height(); y < target.height(); y++) {
        for (unsigned int x = 0; x < target.width(); x++) {
          *target.data_ptr(x, y, target_map, target_sample) = 0;
        }
      }
    } else {
      // Interpolate
#pragma omp parallel for default(shared)
      for(unsigned int y = 0; y < target.height(); y++) {
        const datum normalized_y = ((datum)y)/(datum)(target.height() - 1);
        const datum source_y = normalized_y * ((datum)source.height() - 1);
        for(unsigned int x = 0; x < target.width(); x++) {
          const datum normalized_x = ((datum)x)/(datum)(target.width() - 1);
          const datum source_x = normalized_x * ((datum)source.width() - 1);

          *target.data_ptr(x, y, target_map, target_sample) =
            source.GetSmoothData(source_x, source_y, source_map, source_sample);

        }
      }
    }

    return true;
  } else {

    // Okay, good to go...

    // Get offsets
    const datum* source_map_data = source.data_ptr_const ( 0, 0, source_map, source_sample );
    datum* target_map_data = target.data_ptr ( 0, 0, target_map, target_sample );

    // Count the number of elements to copy
    std::size_t elements_to_copy = source.width() * source.height();

    // Copy the data
    std::memcpy ( target_map_data, source_map_data,
                  sizeof ( datum ) * elements_to_copy / sizeof ( char ) );

    return true;
  }
}

void Tensor::DeleteIfPossible() {
  if ( data_ptr_ != nullptr ) {
    if ( !is_shadow_ ) {
#ifdef BUILD_POSIX
      if(mmapped_) {
        munmap((void*)original_mmap_, (elements_ * sizeof(datum)) / sizeof(char));
        original_mmap_ = nullptr;
        mmapped_ = false;
      } else {
#endif
#ifdef BLAS_MKL
        mkl_free ( data_ptr_ );
#else
        delete[] data_ptr_;
#endif
#ifdef BUILD_POSIX
      }
#endif
#ifdef BUILD_OPENCL
      if ( cl_data_ptr_ != 0 ) {
        clReleaseMemObject ( (cl_mem)cl_data_ptr_ );
        cl_data_ptr_ = 0;
      }
      
      cl_gpu_ = false;
#endif
    }

    data_ptr_ = nullptr;
  }

  samples_ = 0;
  width_ = 0;
  height_ = 0;
  maps_ = 0;
  elements_ = 0;
  is_shadow_ = false;
  shadow_target_ = nullptr;
}
#ifdef BUILD_OPENCL
void Tensor::MoveToGPU ( bool no_copy ) {
  if ( is_shadow_ ) {
    shadow_target_->MoveToGPU ( no_copy );
    return;
  }

  if ( !cl_gpu_ ) {
    if ( cl_data_ptr_ == 0 ) {
      // Error variable
      cl_int error_ret = 0;

      cl_data_ptr_ = clCreateBuffer ( CLHelper::context, CL_MEM_READ_WRITE,
                                      elements_ * sizeof ( datum ), NULL, &error_ret );

      if ( cl_data_ptr_ == NULL || error_ret != CL_SUCCESS ) {
        FATAL ( "Error creating GPU buffer: " << error_ret );
      }
    }


    if ( !no_copy ) {

      // Write to GPU
      cl_int error_ret = 0;
#ifdef BRUTAL_FINISH
      error_ret = clFinish ( CLHelper::queue );

      if ( error_ret != CL_SUCCESS ) {
        FATAL ( "Error finishing command queue (1): " << error_ret );
      }

#endif

      error_ret = clEnqueueWriteBuffer ( CLHelper::queue, (cl_mem)cl_data_ptr_, CL_TRUE,
                                         0, elements_ * sizeof ( datum ), data_ptr_, 0, NULL, NULL );

      if ( error_ret != CL_SUCCESS ) {
        FATAL ( "Error moving to GPU: " << error_ret );
      }
      
      CLHelper::bytes_up += elements_ * sizeof(datum);

#ifdef BRUTAL_FINISH
      error_ret = clFinish ( CLHelper::queue );

      if ( error_ret != CL_SUCCESS ) {
        FATAL ( "Error finishing command queue (2): " << error_ret );
      }

#endif
    }

    cl_gpu_ = true;
  }
}

void Tensor::MoveToCPU ( bool no_copy ) {
  if ( is_shadow_ ) {
    shadow_target_->MoveToCPU ( no_copy );
    return;
  }

  if ( cl_gpu_ ) {
    if ( !no_copy ) {
      if ( cl_data_ptr_ == 0 ) {
        FATAL ( "Move to CPU requested for memory that was not on the GPU" );
      }

      cl_int error_ret = 0;
      error_ret = clFinish ( CLHelper::queue );

      if ( error_ret != CL_SUCCESS ) {
        FATAL ( "Error finishing command queue (1): " << error_ret );
      }

      // Read from GPU
      error_ret = clEnqueueReadBuffer ( CLHelper::queue, (cl_mem)cl_data_ptr_, CL_TRUE, 0,
                                        elements_ * sizeof ( datum ), data_ptr_, 0, NULL, NULL );

      if ( error_ret != CL_SUCCESS ) {
        FATAL ( "Error moving to CPU: " << error_ret );
      }
      
      CLHelper::bytes_down += elements_ * sizeof(datum);

      error_ret = clFinish ( CLHelper::queue );

      if ( error_ret != CL_SUCCESS ) {
        FATAL ( "Error finishing command queue (2): " << error_ret );
      }
    }

    cl_gpu_ = false;
  }
}

#else
void Tensor::MoveToGPU ( bool no_copy ) {
}
void Tensor::MoveToCPU ( bool no_copy ) {
}
#endif

std::size_t Tensor::Maximum ( std::size_t sample ) {
  datum max_y = std::numeric_limits<datum>::lowest();
  std::size_t max_x = 0;

  for ( std::size_t x = 0; x < width_ * height_ * maps_; x++ ) {
    if ( data_ptr_[sample * ( width_ * height_ * maps_ ) + x] > max_y ) {
      max_x = x;
      max_y = data_ptr_[sample * ( width_ * height_ * maps_ ) + x];
    }
  }

  return max_x;
}

std::size_t Tensor::AbsMaximum () {
  datum max_y = 0;
  std::size_t max_x = 0;

  for ( std::size_t x = 0; x < elements_; x++ ) {
    const datum d = data_ptr_[x];
    const datum a = fabs ( d );

    if ( a > max_y ) {
      max_x = x;
      max_y = a;
    }
  }

  return max_x;
}

std::size_t Tensor::PixelMaximum ( std::size_t x, std::size_t y, std::size_t sample ) {
  unsigned int maxclass = 0;
  datum maxvalue = std::numeric_limits<datum>::lowest();

  for ( unsigned int c = 0; c < maps_; c++ ) {
    const datum value = *data_ptr_const ( x,y,c,sample );

    if ( value > maxvalue ) {
      maxvalue = value;
      maxclass = c;
    }
  }
  
  return maxclass;
}


void Tensor::LoadFromFile ( const std::string& filename ) {
#ifdef BUILD_PNG

  if ( ( filename.compare ( filename.length() - 3, 3, "png" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "PNG" ) == 0 )
     ) {
    std::ifstream input_image_file ( filename, std::ios::in | std::ios::binary );

    if ( !input_image_file.good() )
      FATAL ( "Cannot load " << filename );

    Conv::PNGUtil::LoadFromStream ( input_image_file, *this );
    return;
  }

#endif
#ifdef BUILD_JPG

  if ( ( filename.compare ( filename.length() - 3, 3, "jpg" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "jpeg" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "JPG" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "JPEG" ) == 0 )
     ) {
    Conv::JPGUtil::LoadFromFile ( filename, *this );
    return;
  }

#endif

  if ( filename.compare ( filename.length() - 6, 6, "Tensor" ) == 0 ) {
    std::ifstream input_image_file ( filename, std::ios::in | std::ios::binary );

    if ( !input_image_file.good() )
      FATAL ( "Cannot load " << filename );

    Deserialize ( input_image_file );
    return;
  }

  FATAL ( "File format not supported!" );
}

void Tensor::WriteToFile ( const std::string& filename ) {
#ifdef BUILD_PNG

  if ( ( filename.compare ( filename.length() - 3, 3, "png" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "PNG" ) == 0 )
     ) {
    std::ofstream output_image_file ( filename, std::ios::out | std::ios::binary );

    if ( !output_image_file.good() )
      FATAL ( "Cannot write " << filename );

    // Write png
    Conv::PNGUtil::WriteToStream ( output_image_file, *this );
    return;
  }

#endif
#ifdef BUILD_JPG

  if ( ( filename.compare ( filename.length() - 3, 3, "jpg" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "jpeg" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "JPG" ) == 0 )
       || ( filename.compare ( filename.length() - 3, 3, "JPEG" ) == 0 )
     ) {
    Conv::JPGUtil::WriteToFile ( filename, *this );
    return;
  }

#endif

  if ( filename.compare ( filename.length() - 6, 6, "Tensor" ) == 0 ) {
    std::ofstream output_image_file ( filename, std::ios::out | std::ios::binary );

    if ( !output_image_file.good() )
      FATAL ( "Cannot write " << filename );

    Serialize ( output_image_file );
    return;
  }

  FATAL ( "File format not supported!" );
}


void Tensor::PrintStats() {
#ifdef BUILD_OPENCL
	MoveToCPU();
#endif
	datum min = std::numeric_limits<datum>::max();
	datum max = std::numeric_limits<datum>::lowest();
	datum sum = 0;
	datum sum_of_sqares = 0;
	datum variance = 0;

	for (unsigned int e = 0; e < elements_; e++) {
		sum_of_sqares += data_ptr_[e] * data_ptr_[e];
		sum += data_ptr_[e];
		if (data_ptr_[e] > max)
			max = data_ptr_[e];
		if (data_ptr_[e] < min)
			min = data_ptr_[e];
	}

	datum avg = sum / (datum)elements_;
	datum l2 = sqrt(sum_of_sqares);

	for (unsigned int e = 0; e < elements_; e++) {
		variance += (data_ptr_[e] - avg) * (data_ptr_[e] - avg);
	}

	variance /= (datum)elements_;

	LOGINFO << "Minimum : " << min;
	LOGINFO << "Maximum : " << max;
	LOGINFO << "Average : " << avg;
	LOGINFO << "L2 Norm : " << l2;
	LOGINFO << "Variance: " << variance;
}

datum Tensor::GetSmoothData(datum x, datum y, std::size_t map, std::size_t sample) const {
  unsigned int left_x = (unsigned int)std::floor(x);
  unsigned int right_x = (unsigned int)std::ceil(x);
  unsigned int left_y = (unsigned int)std::floor(y);
  unsigned int right_y = (unsigned int)std::ceil(y);

  const Conv::datum Q11 = *(data_ptr(left_x, left_y, map, sample));
  const Conv::datum Q21 = *(data_ptr(right_x, left_y, map, sample));
  const Conv::datum Q12 = *(data_ptr(left_x, right_y, map, sample));
  const Conv::datum Q22 = *(data_ptr(right_x, right_y, map, sample));

  const Conv::datum L1 = (left_x == right_x) ? Q11 : ((right_x - x)/(right_x - left_x)) * Q11 + ((x - left_x)/(right_x - left_x)) * Q21;
  const Conv::datum L2 = (left_x == right_x) ? Q12 : ((right_x - x)/(right_x - left_x)) * Q12 + ((x - left_x)/(right_x - left_x)) * Q22;

  const Conv::datum smooth = (left_y == right_y) ? L1 : ((right_y - y)/(right_y - left_y)) * L1 + ((y - left_y)/(right_y - left_y)) * L2;
  return smooth;
}

std::ostream& operator<< ( std::ostream& output, const Tensor& tensor ) {
  std::stringstream ss;
  ss << "(" << tensor.samples() << "s@" << tensor.width() <<
         "x" << tensor.height() << "x" << tensor.maps() << "m)";
  return output << ss.str();
}

bool Tensor::operator==(const Tensor& rhs) const
{
  if(elements() == rhs.elements()) {
    // Check each element
    for(std::size_t e = 0; e < elements(); e++) {
      // This should not be used for authentication purposes :D
      if ((*this)(e) != rhs(e))
        return false;
    }
    return true;
  } else {
    // Can not possibly be equal
    return false;
  }
}


std::string Tensor::ToBase64(const int sample)
{
  MoveToCPU();
  if(sample >= 0) {
    std::size_t elements_per_sample = elements_ / samples_;
    return base64::encode((const uint8_t*)data_ptr_const(0,0,0,sample), elements_per_sample * sizeof(datum));
  } else {
    return base64::encode((const uint8_t*)data_ptr_, elements_ * sizeof(datum));
  }
}

bool Tensor::FromBase64(const std::string& base64_, const int sample)
{
  MoveToCPU();
    std::size_t max_elements = base64::decoded_max_size(base64_.length());
  if(sample >= 0) {
    std::size_t elements_per_sample = elements_ / samples_;
    if((elements_per_sample * sizeof(datum)) + 4 > max_elements) { 
      std::size_t decoded = base64::decode((uint8_t*)data_ptr(0,0,0,sample), max_elements,
        base64_);
      if(decoded == elements_per_sample * sizeof(datum))
        return true;
      else return false;
    } else
    return false;
  } else {
    if((elements_ * sizeof(datum)) + 4 > max_elements) { 
      std::size_t decoded = base64::decode((uint8_t*)data_ptr_, max_elements,
        base64_);
      if(decoded == elements_ * sizeof(datum))
        return true;
      else return false;
    } else
    return false;
  }
}


}
