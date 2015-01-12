/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file Dataset.h
 * \brief Represents a dataset for different tasks
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_DATASET_H
#define CONV_DATASET_H

#include <vector>

#include "Config.h"
#include "Tensor.h"

namespace Conv
{
  
enum Task {
  SEMANTIC_SEGMENTATION
};

class Dataset
{
public:
  /**
   * @brief Gets the task this Dataset is designed for.
   */
  virtual Task GetTask() = 0;
  
  /**
   * @brief Gets the width of the images in this Dataset.
   */
  virtual unsigned int GetWidth() = 0;
  
  /**
   * @brief Gets the height of the images in this Dataset.
   */
  virtual unsigned int GetHeight() = 0;
  
  /**
   * @brief Gets the number of input maps in this Dataset.
   */
  virtual unsigned int GetInputMaps() = 0;
  
  /**
   * @brief Gets the number of label maps in this Dataset.
   */
  virtual unsigned int GetLabelMaps() = 0;
  
  /**
   * @brief Gets the number of classes in this Dataset.
   */
  virtual unsigned int GetClasses() = 0;
  
  /**
   * @brief Gets the names of the classes in this Dataset.
   */
  virtual std::vector<std::string> GetClassNames() = 0;
  
  /**
   * @brief Gets the number of training samples in this Dataset.
   */
  virtual unsigned int GetTrainingSamples() = 0;
  
  /**
   * @brief Gets the number of testing samples in this Dataset.
   */
  virtual unsigned int GetTestingSamples() = 0;
  
  /**
    * @brief Checks if this Dataset supports testing.
    * @returns True if testing is supported
    */
  virtual bool SupportsTesting () = 0;

  /**
    * @brief Fill the specified Tensors with the specified training sample.
    * @param data_tensor An empty Tensor
    * @param label_tensor An empty Tensor
    * @param weight_tensor An empty Tensor
    * @param sample The sample in the target Tensors
    * @param index The index of the training sample to load
    * @returns True on success
    */
  virtual bool GetTrainingSample ( Tensor& data_tensor, Tensor& label_tensor,
				   Tensor& weight_tensor, 
				   unsigned int sample, unsigned int index) = 0;

  /**
    * @brief Fill the specified Tensors with the specified testing sample.
    * @param data_tensor An empty Tensor
    * @param label_tensor An empty Tensor
    * @param weight_tensor An empty Tensor
    * @param sample The sample in the target Tensors
    * @param index The index of the testing sample to load
    * @returns True on success
    */
  virtual bool GetTestingSample ( Tensor& data_tensor, Tensor& label_tensor,
				  Tensor& weight_tensor, 
				   unsigned int sample, unsigned int index) = 0;
};
 
/*
 * \brief This type is used for function pointers to a localized error function.
 *
 * A localized error function weighs the network's error w.r.t x and y
 * coordinates.
 */
typedef datum (*dataset_localized_error_function) (unsigned int, unsigned int);
datum DefaultLocalizedErrorFunction (unsigned int x, unsigned int y);

class TensorStreamDataset : public Dataset {
public:
  TensorStreamDataset(std::istream& training_stream, std::istream& testing_stream, unsigned int classes, std::vector<std::string> class_names, dataset_localized_error_function error_function = DefaultLocalizedErrorFunction);
  
  // Dataset implementations
  virtual Task GetTask();
  virtual unsigned int GetWidth();
  virtual unsigned int GetHeight();
  virtual unsigned int GetInputMaps();
  virtual unsigned int GetLabelMaps();
  virtual unsigned int GetClasses();
  virtual std::vector< std::string > GetClassNames();
  virtual unsigned int GetTrainingSamples();
  virtual unsigned int GetTestingSamples();
  virtual bool SupportsTesting();
  virtual bool GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index);
  virtual bool GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& weight_tensor,  unsigned int sample, unsigned int index);
  
private:
  // Stored data
  Tensor* data_ = nullptr;
  Tensor* labels_ = nullptr;
  
  Tensor error_cache;
  
  unsigned int input_maps_ = 0;
  unsigned int label_maps_ = 0;
  unsigned int tensors_ = 0;
  
  unsigned int tensor_count_training_ = 0;
  unsigned int tensor_count_testing_ = 0;
  
  // Parameters
  std::vector<std::string> class_names_;
  unsigned int classes_;
};
}

#endif
