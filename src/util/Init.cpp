/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#ifdef BUILD_OPENCL
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#endif

#include "Init.h"
#include "Config.h"
#include "Log.h"

#include <locale.h>

#ifdef BUILD_GUI
#include <gtk/gtk.h>
#endif

namespace Conv {

#ifdef BUILD_OPENCL
cl_context System::context = 0;
cl_command_queue System::queue = 0;
cl_device_id System::device = 0;

cl_kernel System::k_crossCorrelation = 0;
cl_kernel System::k_fullConvolution = 0;
cl_kernel System::k_biasedConvolution = 0;
cl_kernel System::k_biasedMatrixVector = 0;
cl_kernel System::k_biasedMatrixVectorGrad = 0;
cl_kernel System::k_biasedMatrixVectorBackward = 0;
cl_kernel System::k_biasGradientPart1 = 0;
cl_kernel System::k_biasGradientPart2 = 0;
cl_kernel System::k_matrixMatrix = 0;
cl_kernel System::k_foldWeights = 0;
cl_kernel System::k_maximumForward = 0;
cl_kernel System::k_maximumBackward = 0;
cl_kernel System::k_nlTanh = 0;
cl_kernel System::k_nlTanhBackward = 0;
cl_kernel System::k_nlSigm = 0;
cl_kernel System::k_nlSigmBackward = 0;
#endif

TensorViewer* System::viewer = nullptr;

void System::Init() {
  LOGINFO << "CN24 built on " << BUILD_DATE;
  LOGINFO << "Copyright (C) 2015 Clemens-Alexander Brust";
  LOGINFO << "For licensing information, see the LICENSE"
          << " file included with this project.";

#ifdef BUILD_OPENCL

  // TODO make this configurable
  unsigned int platform_number = 0;
  unsigned int device_number = 0;

  cl_uint platform_count = 0;
  clGetPlatformIDs ( 0, 0, &platform_count );

  if ( platform_count == 0 ) {
    FATAL ( "No OpenCL platforms detected!" );
  }

  cl_platform_id* platform_ids = new cl_platform_id[platform_count];
  clGetPlatformIDs ( platform_count, platform_ids, NULL );

  cl_uint device_count = 0;
  clGetDeviceIDs ( platform_ids[platform_number], CL_DEVICE_TYPE_ALL, 0,
                   NULL, &device_count );

  if ( device_count == 0 ) {
    FATAL ( "No OpenCL devices detected!" );
  }

  cl_device_id* device_ids = new cl_device_id[device_count];
  clGetDeviceIDs ( platform_ids[platform_number], CL_DEVICE_TYPE_ALL,
                   device_count, device_ids, NULL );

  char device_name_buffer[256];
  clGetDeviceInfo ( device_ids[device_number], CL_DEVICE_NAME, 256,
                    device_name_buffer, 0 );

  uint32_t support_buffer;
  clGetDeviceInfo ( device_ids[device_number], CL_DEVICE_IMAGE_SUPPORT, 4,
                    &support_buffer, 0 );

  uint32_t max_wg_size;
  clGetDeviceInfo ( device_ids[device_number], CL_DEVICE_MAX_WORK_GROUP_SIZE, 4,
                    &max_wg_size, 0 );

  LOGINFO << "Using OpenCL device: " << device_name_buffer;
  LOGINFO << "Image support: " << ( support_buffer ? "Yes" : "No" );
  LOGINFO << "Max work group size: " << max_wg_size;

  device = device_ids[device_number];

  // Create context
  const cl_context_properties context_properties [] = {
    CL_CONTEXT_PLATFORM,
    reinterpret_cast<cl_context_properties> ( platform_ids [platform_number] ),
    0, 0
  };

  LOGINFO << "Creating OpenCL context...";

  cl_int error = 0;
  context = clCreateContext ( context_properties, device_count,
                              device_ids, 0, 0, &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating OpenCL context: " << error );
  }

  // Create command queue
  LOGINFO << "Creating OpenCL command queue...";
  queue = clCreateCommandQueue ( context, device_ids[device_number], 0, &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating OpenCL command queue: " << error );
  }

  delete device_ids;
  delete platform_ids;

  // Compile kernels
  cl_program p_crossCorrelation = CreateProgram ( "kernels/crossCorrelation.cl" );
  cl_program p_biasedConvolution = CreateProgram ( "kernels/biasedConvolution.cl" );
  cl_program p_fullConvolution = CreateProgram ( "kernels/fullConvolution.cl" );
  cl_program p_foldWeights = CreateProgram ( "kernels/foldWeights.cl" );
  cl_program p_biasedMatrixVector = CreateProgram ( "kernels/biasedMatrixVector.cl" );
  cl_program p_biasGradient = CreateProgram ( "kernels/biasGradient.cl" );
  cl_program p_matrixMatrix = CreateProgram ( "kernels/matrixMatrix.cl" );
  cl_program p_maximum = CreateProgram ( "kernels/maximumPooling.cl" );
  cl_program p_nonLinearFunctions = CreateProgram ( "kernels/nonLinearFunctions.cl" );

  k_crossCorrelation = clCreateKernel ( p_crossCorrelation, "CROSS_CORRELATION", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_biasedConvolution = clCreateKernel ( p_biasedConvolution, "BIASED_CONVOLUTION", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_fullConvolution = clCreateKernel ( p_fullConvolution, "FULL_CONVOLUTION", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_foldWeights = clCreateKernel ( p_foldWeights, "FOLD_WEIGHTS", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_biasedMatrixVector = clCreateKernel ( p_biasedMatrixVector, "BIASED_MATRIX_VECTOR_FWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_biasedMatrixVectorGrad = clCreateKernel ( p_biasedMatrixVector, "BIASED_MATRIX_VECTOR_GRAD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_biasedMatrixVectorBackward = clCreateKernel ( p_biasedMatrixVector, "BIASED_MATRIX_VECTOR_BWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_biasGradientPart1 = clCreateKernel ( p_biasGradient, "BIAS_GRADIENT_PART1", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_biasGradientPart2 = clCreateKernel ( p_biasGradient, "BIAS_GRADIENT_PART2", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_matrixMatrix = clCreateKernel ( p_matrixMatrix, "MATRIX_MATRIX", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_maximumForward = clCreateKernel ( p_maximum, "MAXIMUM_POOLING_FWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_maximumBackward = clCreateKernel ( p_maximum, "MAXIMUM_POOLING_BWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_nlTanh = clCreateKernel ( p_nonLinearFunctions, "NL_TANH_FWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_nlTanhBackward = clCreateKernel ( p_nonLinearFunctions, "NL_TANH_BWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_nlSigm = clCreateKernel ( p_nonLinearFunctions, "NL_SIGM_FWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_nlSigmBackward = clCreateKernel ( p_nonLinearFunctions, "NL_SIGM_BWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

#endif

#ifdef BUILD_GUI
  gtk_init ( nullptr, nullptr );
#endif
  viewer = new TensorViewer();
}

#ifdef BUILD_OPENCL
cl_program System::CreateProgram ( const char* file_name ) {
  cl_int error = 0;
  cl_program program = 0;

  LOGDEBUG << "Compiling " << file_name;
#ifdef _MSC_VER
  std::ifstream kernel_file ( "../" + std::string ( file_name ), std::ios::in );
#else
  std::ifstream kernel_file ( file_name, std::ios::in );
#endif

  if ( !kernel_file.good() ) {
    FATAL ( "Cannot open kernel: " << file_name );
  }

  std::ostringstream oss;
  oss << kernel_file.rdbuf();

  std::string kernel_content = oss.str();
  const char* kernel_content_char = kernel_content.c_str();

  program = clCreateProgramWithSource ( context, 1, ( const char** ) &kernel_content_char, NULL, NULL );

  if ( program == NULL ) {
    FATAL ( "Cannot create kernel: " << file_name );
  }

  error = clBuildProgram ( program, 1, &device, NULL, NULL, NULL );

  if ( error != CL_SUCCESS ) {
    char build_log[16384];
    clGetProgramBuildInfo ( program, device, CL_PROGRAM_BUILD_LOG, 16384, build_log, NULL );
    LOGERROR << "Error compiling kernel " << file_name << ":\n" << std::string ( build_log );
    FATAL ( "Compilation failed, exiting..." );
  }

  return program;
}

#endif

}
