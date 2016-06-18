/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#ifndef CONV_ILSVRCDATASET_H
#define CONV_ILSVRCDATASET_H

#include <vector>
#include <map>
#include "JSONParsing.h"
#include "MemoryMappedTar.h"

#include "Dataset.h"

namespace Conv {

struct ILSVRCSynsetInfo {
  std::string class_name; // "dog", "apple"...
  unsigned int ILSVRC_ID; // 1-1000
  unsigned int class_number;
};

class ILSVRCDataset : public Dataset {
public:
  ILSVRCDataset();
  ~ILSVRCDataset() {};
  virtual Task GetTask() const { return CLASSIFICATION; };
  virtual Method GetMethod() const { return FCN; }
  virtual unsigned int GetWidth() const { return 224; };
  virtual unsigned int GetHeight() const { return 224; };
  virtual unsigned int GetInputMaps() const { return 3; };
  virtual unsigned int GetLabelMaps() const { return classes_; };
  virtual unsigned int GetClasses() const { return classes_; };
  virtual std::vector< std::string > GetClassNames() const { return class_names_; };
  virtual std::vector< unsigned int > GetClassColors() const { return class_colors_; };
  virtual std::vector< datum > GetClassWeights() const { return class_weights_; };
  virtual unsigned int GetTrainingSamples() const { return training_samples_; };
  virtual unsigned int GetTestingSamples() const { return testing_samples_; };
  virtual bool SupportsTesting() const { return true; };
  virtual bool GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index);
  virtual bool GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index);

  void Load(JSON descriptor);

private:
  std::vector<std::string> class_names_;
  std::vector<unsigned int> class_colors_;
  std::vector<datum> class_weights_;

  unsigned int classes_;
  unsigned int training_samples_;
  unsigned int testing_samples_;

  std::map<std::string, ILSVRCSynsetInfo*> synsets_;
};
}

#endif
