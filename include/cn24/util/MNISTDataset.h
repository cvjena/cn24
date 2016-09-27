/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#ifndef CONV_MNISTDATASET_H
#define CONV_MNISTDATASET_H

#include <vector>
#include "JSONParsing.h"

#include "Dataset.h"

namespace Conv {
class MNISTDataset : public Dataset {
public:
  explicit MNISTDataset(ClassManager* class_manager);
  ~MNISTDataset() {};
  virtual Task GetTask() const { return CLASSIFICATION; };
  virtual Method GetMethod() const { return FCN; }
  virtual unsigned int GetWidth() const { return 28; };
  virtual unsigned int GetHeight() const { return 28; };
  virtual unsigned int GetInputMaps() const { return 1; };
  virtual unsigned int GetLabelMaps() const { return 10; };
  virtual unsigned int GetClasses() const { return 10; };
  virtual std::vector< std::string > GetClassNames() const { return class_names_; };
  virtual std::vector< unsigned int > GetClassColors() const { return class_colors_; };
  virtual std::vector< datum > GetClassWeights() const { return class_weights_; };
  virtual unsigned int GetTrainingSamples() const { return 60000; };
  virtual unsigned int GetTestingSamples() const { return 10000; };
  virtual bool SupportsTesting() const { return true; };
  virtual bool GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index);
  virtual bool GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index);
  virtual bool GetTrainingMetadata(DatasetMetadataPointer* metadata_array, unsigned int sample, unsigned int index) {
      UNREFERENCED_PARAMETER(metadata_array);
      UNREFERENCED_PARAMETER(sample);
      UNREFERENCED_PARAMETER(index);
      return false; };
  virtual bool GetTestingMetadata(DatasetMetadataPointer* metadata_array, unsigned int sample, unsigned int index) {
      UNREFERENCED_PARAMETER(metadata_array);
      UNREFERENCED_PARAMETER(sample);
      UNREFERENCED_PARAMETER(index);
      return false; };

  void Load(JSON descriptor);

private:
  std::vector<std::string> class_names_;
  std::vector<unsigned int> class_colors_;
  std::vector<datum> class_weights_;

  uint8_t* train_images_;
  uint8_t* train_labels_;
  uint8_t* test_images_;
  uint8_t* test_labels_;
};
}

#endif
