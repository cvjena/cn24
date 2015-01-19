/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testBLAS.cpp
 * \brief Small test application for BLAS library
 * 
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <random>

#include <cn24.h>

#ifdef BLAS_ATLAS
  extern "C" { void ATL_buildinfo(void); }
#endif

int main() {
  int M = 1000;
  int K = 5000;
  int N = 6000;
  
  Conv::datum* A = new Conv::datum[M*K];
  Conv::datum* B = new Conv::datum[K*N];
  
  Conv::datum* RESULT = new Conv::datum[M*N];
  
  std::mt19937 rand;
  std::uniform_real_distribution<Conv::datum> dist(-5.0, 5.0);
  
  LOGINFO << "Filling...";
  for(int i = 0; i < M*K; i++)
    A[i] = dist(rand);
  
  for(int i = 0; i < K*N; i++)
    B[i] = dist(rand);
  
  LOGINFO << "Done filling!";
  
  GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, RESULT, N);
  
  LOGINFO << "Done calculating";
  
  LOGEND;
  
#ifdef BLAS_ATLAS
  ATL_buildinfo();
#endif
  
  return 0;
}