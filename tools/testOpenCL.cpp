/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testOpenCL.cpp
 * \brief Small test application for the OpenCL library
 * 
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <random>
#include <cn24.h>

#ifdef BUILD_OPENCL
#include <CL/cl.h>

int main() {
  
  cl_uint platform_count = 0;
  clGetPlatformIDs(0, 0, &platform_count);
  LOGINFO << "OpenCL platform(s) available: " << platform_count;
  
  cl_platform_id* platform_ids = new cl_platform_id[platform_count];
  clGetPlatformIDs(platform_count, platform_ids, 0);
  
  for(int i = 0; i < platform_count; i++) {
    cl_uint device_count = 0;
    clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, 0, &device_count);
    LOGINFO << "Platform " << i + 1 << ": " << device_count << " devices";
    
    cl_device_id* device_ids = new cl_device_id[device_count];
    clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, device_count,
      device_ids, 0);
    
    for(int j = 0; j < device_count; j++) {
      char buf[256];
      clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, 256, buf, 0);
      LOGINFO << "Platform " << i + 1 << ", device " << j + 1 << ": "
        << buf;
      
      clGetDeviceInfo(device_ids[j], CL_DEVICE_VERSION, 256, buf, 0);
      LOGINFO << "Platform " << i + 1 << ", device " << j + 1 << ": supports "
        << buf;
    }
    
  }

  Conv::System::Init();
  
  LOGEND;
  
  return 0;
}

#else

int main() {
  LOGERROR << "OpenCL not built in!";
  LOGEND;
  return -1;
}
#endif