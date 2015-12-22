/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Dataset.h
 * @brief Represents a dataset for different tasks
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_DATASET_H
#define CONV_DATASET_H

#include <vector>

#include "Config.h"
#include "Tensor.h"
#include "TensorStream.h"

namespace Conv
{
  
enum Task {
  SEMANTIC_SEGMENTATION
};

enum Method {
  FCN,
  PATCH
};

class Dataset
{
public:
  /**
   * @brief Gets the task this Dataset is designed for.
   */
  virtual Task GetTask() const = 0;
  
  /**
   * @brief Gets the method this Dataset is designed for.
   */
  virtual Method GetMethod() const = 0;
  
  /**
   * @brief Gets the width of the largest image in this Dataset.
   */
  virtual unsigned int GetWidth() const = 0;
  
  /**
   * @brief Gets the height of the largest image in this Dataset.
   */
  virtual unsigned int GetHeight() const = 0;
  
  /**
   * @brief Gets the number of input maps in this Dataset.
   */
  virtual unsigned int GetInputMaps() const = 0;
  
  /**
   * @brief Gets the number of label maps in this Dataset.
   */
  virtual unsigned int GetLabelMaps() const = 0;
  
  /**
   * @brief Gets the number of classes in this Dataset.
   */
  virtual unsigned int GetClasses() const = 0;
  
  /**
   * @brief Gets the names of the classes in this Dataset.
   */
  virtual std::vector<std::string> GetClassNames() const = 0;
  
  /**
   * @brief Gets the colors of the classes in this Dataset.
   * The colors are unsigned ints of the format 0x00RRGGBB.
   */
  virtual std::vector<unsigned int> GetClassColors() const = 0;
  
  /**
   * @brief Gets the colors of the classes in this Dataset.
   * The colors are unsigned ints of the format 0x00RRGGBB.
   */
  virtual std::vector<datum> GetClassWeights() const = 0;
  
  /**
   * @brief Gets the number of training samples in this Dataset.
   */
  virtual unsigned int GetTrainingSamples() const = 0;
  
  /**
   * @brief Gets the number of testing samples in this Dataset.
   */
  virtual unsigned int GetTestingSamples() const = 0;
  
  /**
    * @brief Checks if this Dataset supports testing.
    * @returns True if testing is supported
    */
  virtual bool SupportsTesting () const = 0;

  /**
    * @brief Fill the specified Tensors with the specified training sample.
    * @param data_tensor An empty Tensor
    * @param label_tensor An empty Tensor
    * @param helper_tensor An empty tensor
    * @param weight_tensor An empty Tensor
    * @param sample The sample in the target Tensors
    * @param index The index of the training sample to load
    * @returns True on success
    */
  virtual bool GetTrainingSample ( Tensor& data_tensor, Tensor& label_tensor,
				   Tensor& helper_tensor, Tensor& weight_tensor, 
				   unsigned int sample, unsigned int index) = 0;

  /**
    * @brief Fill the specified Tensors with the specified testing sample.
    * @param data_tensor An empty Tensor
    * @param label_tensor An empty Tensor
    * @param helper_tensor An empty Tensor
    * @param weight_tensor An empty Tensor
    * @param sample The sample in the target Tensors
    * @param index The index of the testing sample to load
    * @returns True on success
    */
  virtual bool GetTestingSample ( Tensor& data_tensor, Tensor& label_tensor,
				  Tensor& helper_tensor, Tensor& weight_tensor, 
				   unsigned int sample, unsigned int index) = 0;
				   
  /**
   * @brief Uses this Dataset's colors to colorize a net output
   */
  virtual void Colorize ( Tensor& net_output_tensor, Tensor& target_tensor);
};
 
/*
 * @brief This type is used for function pointers to a localized error function.
 *
 * A localized error function weighs the network's error w.r.t x and y
 * coordinates.
 */
typedef datum (*dataset_localized_error_function) (unsigned int, unsigned int, unsigned int, unsigned int);
datum DefaultLocalizedErrorFunction (unsigned int x, unsigned int y, unsigned int w, unsigned int h);

enum DatasetLoadSelection {
  LOAD_TRAINING_ONLY,
  LOAD_TESTING_ONLY,
  LOAD_BOTH
};

class TensorStreamPatchDataset : public Dataset {
 public:
	 TensorStreamPatchDataset(std::istream& training_stream,
		 std::istream& testing_stream,
		 unsigned int classes,
		 std::vector<std::string> class_names,
		 std::vector<unsigned int> class_colors,
		 std::vector<datum> class_weights,
		 unsigned int patchsize_x,
		 unsigned int patchsize_y,
		 dataset_localized_error_function error_function = DefaultLocalizedErrorFunction,
    int training_fd = 0, int testing_fd = 0);
 
  // Dataset implementations
  virtual Task GetTask() const;
  virtual Method GetMethod() const { return PATCH; }
  virtual unsigned int GetWidth() const;
  virtual unsigned int GetHeight() const;
  virtual unsigned int GetInputMaps() const;
  virtual unsigned int GetLabelMaps() const;
  virtual unsigned int GetClasses() const;
  virtual std::vector< std::string > GetClassNames() const;
  virtual std::vector< unsigned int > GetClassColors() const;
  virtual std::vector< datum > GetClassWeights() const;
  virtual unsigned int GetTrainingSamples() const;
  virtual unsigned int GetTestingSamples() const;
  virtual bool SupportsTesting() const;
  virtual bool GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index);
  virtual bool GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index);
  
  static TensorStreamPatchDataset* CreateFromConfiguration(std::istream& file, bool dont_load, DatasetLoadSelection selection, unsigned int patchsize_x, unsigned int patchsize_y);
  
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
  
  unsigned int sample_count_training_ = 0;
  unsigned int sample_count_testing_ = 0;
  
  unsigned int* last_sample_ = nullptr;
  
  unsigned int patchsize_x_ = 0;
  unsigned int patchsize_y_ = 0;
  
  // Parameters
  std::vector<std::string> class_names_;
  std::vector<unsigned int> class_colors_;
  std::vector<datum> class_weights_;
  unsigned int classes_;
  dataset_localized_error_function error_function_;
}; 

class TensorStreamDataset : public Dataset {
public:
  TensorStreamDataset(/*std::istream& training_stream,
    std::istream& testing_stream,*/
    TensorStream* training_stream,
    TensorStream* testing_stream,
    unsigned int classes,
    std::vector<std::string> class_names,
    std::vector<unsigned int> class_colors,
		std::vector<datum> class_weights,
    dataset_localized_error_function error_function = DefaultLocalizedErrorFunction);
  
  // Dataset implementations
  virtual Task GetTask() const;
  virtual Method GetMethod() const { return FCN; }
  virtual unsigned int GetWidth() const;
  virtual unsigned int GetHeight() const;
  virtual unsigned int GetInputMaps() const;
  virtual unsigned int GetLabelMaps() const;
  virtual unsigned int GetClasses() const;
  virtual std::vector< std::string > GetClassNames() const;
  virtual std::vector< unsigned int > GetClassColors() const;
  virtual std::vector< datum > GetClassWeights() const;
  virtual unsigned int GetTrainingSamples() const;
  virtual unsigned int GetTestingSamples() const;
  virtual bool SupportsTesting() const;
  virtual bool GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index);
  virtual bool GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index);
  
  static TensorStreamDataset* CreateFromConfiguration(std::istream& file, bool dont_load = false, DatasetLoadSelection selection = LOAD_BOTH);
  
private:
  // Stored data
  /*
  Tensor* data_ = nullptr;
  Tensor* labels_ = nullptr;
  */
  TensorStream* training_stream_;
  TensorStream* testing_stream_;
  
  Tensor error_cache;
  
  unsigned int input_maps_ = 0;
  unsigned int label_maps_ = 0;
  unsigned int tensors_ = 0;
  
  unsigned int tensor_count_training_ = 0;
  unsigned int tensor_count_testing_ = 0;
  
  unsigned int max_width_ = 0;
  unsigned int max_height_ = 0;
  
  // Parameters
  std::vector<std::string> class_names_;
  std::vector<unsigned int> class_colors_;
  std::vector<datum> class_weights_;
  unsigned int classes_;
  dataset_localized_error_function error_function_;
};
}

#endif
