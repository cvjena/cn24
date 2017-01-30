/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Init.h"
#include "CLHelper.h"
#include "Config.h"
#include "ConfigParsing.h"
#include "Log.h"
#include "PathFinder.h"

#include <locale.h>

#ifdef BUILD_GUI_GTK
#include <gtk/gtk.h>
#endif

#ifdef BUILD_WIN32
#include <Windows.h>
#else
#ifdef BUILD_OSX 
#include <mach-o/dyld.h>
#else
#ifdef BUILD_LINUX
#include <unistd.h>
#endif
#endif
#endif

#include "JSONParsing.h"
#include "TensorViewer.h"
#include "StatAggregator.h"

namespace Conv {

#ifdef BUILD_OPENCL
long CLHelper::bytes_up = 0;
long CLHelper::bytes_down = 0;
cl_context CLHelper::context = 0;
cl_command_queue CLHelper::queue = 0;
cl_device_id CLHelper::device = 0;

cl_kernel CLHelper::k_crossCorrelation = 0;
cl_kernel CLHelper::k_fullConvolution = 0;
cl_kernel CLHelper::k_biasedConvolution = 0;
cl_kernel CLHelper::k_biasedMatrixVector = 0;
cl_kernel CLHelper::k_biasedMatrixVectorGrad = 0;
cl_kernel CLHelper::k_biasedMatrixVectorBackward = 0;
cl_kernel CLHelper::k_biasGradientPart1 = 0;
cl_kernel CLHelper::k_biasGradientPart2 = 0;
cl_kernel CLHelper::k_matrixMatrix = 0;
cl_kernel CLHelper::k_foldWeights = 0;
cl_kernel CLHelper::k_maximumForward = 0;
cl_kernel CLHelper::k_maximumBackward = 0;
cl_kernel CLHelper::k_amaximumForward = 0;
cl_kernel CLHelper::k_amaximumBackward = 0;
cl_kernel CLHelper::k_nlTanh = 0;
cl_kernel CLHelper::k_nlTanhBackward = 0;
cl_kernel CLHelper::k_nlSigm = 0;
cl_kernel CLHelper::k_nlSigmBackward = 0;
cl_kernel CLHelper::k_nlLeaky = 0;
cl_kernel CLHelper::k_nlLeakyBackward = 0;
cl_kernel CLHelper::k_setValue = 0;
cl_kernel CLHelper::k_sms = 0;
cl_kernel CLHelper::k_im2col = 0;
cl_kernel CLHelper::k_col2im = 0;
cl_kernel CLHelper::k_up = 0;
cl_kernel CLHelper::k_down = 0;
cl_kernel CLHelper::k_applyMask = 0;
#endif

TensorViewer* System::viewer = nullptr;
StatAggregator* System::stat_aggregator = nullptr;
int System::log_level = 0;

void System::Init(int requested_log_level) {
  if(requested_log_level == -1) {
#ifdef BUILD_VERBOSE
    log_level = 3;
#else
    log_level = 2;
#endif
  } else
    log_level = requested_log_level;
  
  LOGINFO << "CN24 version 3.0.0-SNAPSHOT";
  LOGINFO << "Copyright (C) 2016 Clemens-Alexander Brust";
  LOGINFO << "For licensing information, see the LICENSE"
          << " file included with this project.";
          
  std::string binary_path;
  GetExecutablePath(binary_path);
  LOGDEBUG << "Executable path: " << binary_path;
  
  unsigned int platform_number = 0;
  unsigned int device_number = 0;
  
  // Look for configuration file
  std::string config_path = PathFinder::FindPath("config.json", binary_path);
  if(config_path.length() == 0) {
    config_path = PathFinder::FindPath("config.json", binary_path + "../");
  }

  // Load and parse config file
  std::ifstream config_file(config_path, std::ios::in);
  if(config_path.length() > 0 && config_file.good()) {
    LOGINFO << "Loading config file: " << config_path;
    JSON config_json = JSON::parse(config_file);

    if(config_json.count("opencl_platform") == 1 && config_json["opencl_platform"].is_number())
      platform_number = config_json["opencl_platform"];
    if(config_json.count("opencl_device") == 1 && config_json["opencl_device"].is_number())
      device_number = config_json["opencl_device"];
  } else {
#ifdef BUILD_OPENCL
    LOGINFO << "Could not find a config file, using default OpenCL settings.";
#endif
  }

  CLHelper::Init(platform_number, device_number);
#ifdef BUILD_GUI_GTK
  if(!gtk_init_check ( nullptr, nullptr )) {
    LOGWARN << "Could not initialize GTK!";
  }
#endif

  // Initialize global TensorViewer
  viewer = new TensorViewer();
  
  // Initialize global StatAggregator
  stat_aggregator = new StatAggregator();
}

void System::GetExecutablePath(std::string& binary_path) {
#ifdef BUILD_WIN32
  binary_path = "";
  TCHAR path[16384];
  DWORD return_value = GetModuleFileName(NULL, path, 16384);
  if (return_value > 0 && return_value < 16384) {
    DWORD last_error = GetLastError();
    if (last_error != ERROR_SUCCESS) {
      LOGWARN << "Could not get executable path, may be unable to locate kernels!";
    }
    binary_path = std::string(path);
    std::size_t last_slash = binary_path.rfind("\\");
    // last_slash should never be npos because this is supposed to be a path
    binary_path = binary_path.substr(0, last_slash + 1);
  }
  else {
    LOGWARN << "Could not get executable path, may be unable to locate kernels!";
  }
#else
#ifdef BUILD_OSX 
  binary_path = "";
  char path[16384];
  uint32_t size = sizeof(path);
  if (_NSGetExecutablePath(path, &size) == 0) {
    binary_path = std::string(path);
    std::size_t last_slash = binary_path.rfind("/");
    // last_slash should never be npos because this is supposed to be a path
    binary_path = binary_path.substr(0, last_slash+1);
  }
  else {
    LOGWARN << "Could not get executable path, may be unable to locate kernels!";
  }
#else
#ifdef BUILD_LINUX
  char buffer[16384];
  ssize_t path_length = ::readlink("/proc/self/exe", buffer, sizeof(buffer)-1);
  binary_path = "";
  if(path_length != -1) {
    buffer[path_length] = '\0';
    binary_path = std::string(buffer);
    std::size_t last_slash = binary_path.rfind("/");
    // last_slash should never be npos because this is supposed to be a path
    binary_path = binary_path.substr(0, last_slash+1);
  } else {
    LOGWARN << "Could not get executable path, may be unable to locate kernels!";
  }
#else
  binary_path = "";
#endif
#endif
#endif
}
  
void System::Shutdown() {
  delete stat_aggregator;
  delete viewer;
  LOGEND;
}

void CLHelper::Init(unsigned int platform_number, unsigned int device_number) {
#ifdef BUILD_OPENCL
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
    FATAL ( "No OpenCL devices detected for platform " << platform_number << "!" );
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

  LOGINFO << "Using OpenCL device: " << device_name_buffer;
  LOGDEBUG << "Image support: " << ( support_buffer ? "Yes" : "No" );

  device = device_ids[device_number];

  // Create context
  const cl_context_properties context_properties [] = {
    CL_CONTEXT_PLATFORM,
    reinterpret_cast<cl_context_properties> ( platform_ids [platform_number] ),
    0, 0
  };

  LOGDEBUG << "Creating OpenCL context...";

  cl_int error = 0;
  context = clCreateContext ( context_properties, 1,
                              &device_ids[device_number], 0, 0, &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating OpenCL context: " << error );
  }

  // Create command queue
  LOGDEBUG << "Creating OpenCL command queue...";
  queue = clCreateCommandQueue ( context, device_ids[device_number], 0, &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating OpenCL command queue: " << error );
  }

  delete[] device_ids;
  delete[] platform_ids;

  // Compile kernels
  cl_program p_crossCorrelation = CreateProgram ( "kernels/crossCorrelation.cl" );
  cl_program p_biasedConvolution = CreateProgram ( "kernels/biasedConvolution.cl" );
  cl_program p_fullConvolution = CreateProgram ( "kernels/fullConvolution.cl" );
  cl_program p_foldWeights = CreateProgram ( "kernels/foldWeights.cl" );
  cl_program p_biasedMatrixVector = CreateProgram ( "kernels/biasedMatrixVector.cl" );
  cl_program p_biasGradient = CreateProgram ( "kernels/biasGradient.cl" );
  cl_program p_matrixMatrix = CreateProgram ( "kernels/matrixMatrix.cl" );
  cl_program p_maximum = CreateProgram ( "kernels/maximumPooling.cl" );
  cl_program p_amaximum = CreateProgram ( "kernels/advmaximumPooling.cl" );
  cl_program p_nonLinearFunctions = CreateProgram ( "kernels/nonLinearFunctions.cl" );
  cl_program p_scaling = CreateProgram ( "kernels/scaling.cl" );
  cl_program p_setValue = CreateProgram ( "kernels/setValue.cl" );
  cl_program p_sms = CreateProgram ( "kernels/sms.cl" );
  cl_program p_im2col = CreateProgram ( "kernels/im2col.cl" );
  cl_program p_applyMask = CreateProgram ( "kernels/applyMask.cl" );

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
  
  k_amaximumForward = clCreateKernel ( p_amaximum, "AMAXIMUM_POOLING_FWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_amaximumBackward = clCreateKernel ( p_amaximum, "AMAXIMUM_POOLING_BWD", &error );

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

  k_nlLeaky = clCreateKernel ( p_nonLinearFunctions, "NL_LEAKY_FWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_nlLeakyBackward = clCreateKernel ( p_nonLinearFunctions, "NL_LEAKY_BWD", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_setValue = clCreateKernel ( p_setValue, "SET_VALUE", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }
  
  k_sms = clCreateKernel ( p_sms, "SMS", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }
  
  k_im2col = clCreateKernel ( p_im2col, "IM2COL", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }
  
  k_col2im = clCreateKernel ( p_im2col, "COL2IM", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }
  
  k_up = clCreateKernel ( p_scaling, "UP", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }
  
  k_down = clCreateKernel ( p_scaling, "DOWN", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }

  k_applyMask = clCreateKernel ( p_applyMask, "APPLY_MASK", &error );

  if ( error != CL_SUCCESS ) {
    FATAL ( "Error creating kernel: " << ( signed int ) error );
  }
#ifdef BUILD_CLBLAS
  cl_int err = clblasSetup();
  if (err!=CL_SUCCESS)
    FATAL("Call to clblasSetup failed. Error: " << err);
#endif

#else
  UNREFERENCED_PARAMETER(platform_number);
  UNREFERENCED_PARAMETER(device_number);
#endif

}

#ifdef BUILD_OPENCL
cl_program CLHelper::CreateProgram ( const char* file_name ) {
  cl_int error = 0;
  cl_program program = 0;

  LOGDEBUG << "Compiling " << file_name;

  std::string binary_path;
  System::GetExecutablePath(binary_path);

  // Search in binary path first
  std::string full_path = PathFinder::FindPath(std::string(file_name), binary_path);
  
  // If kernel cannot be found, go up one folder (Xcode, Visual Studio and
  // other multi-target build setups)
  if (full_path.length() == 0) {
    full_path = PathFinder::FindPath(std::string(file_name), binary_path + "../");
  }
  
  std::ifstream kernel_file ( full_path, std::ios::in );

  if ( !kernel_file.good() ) {
    FATAL ( "Cannot open kernel: " << full_path );
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
