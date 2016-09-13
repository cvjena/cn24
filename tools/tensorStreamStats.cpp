/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <fstream>
#include <vector>
#include <string>

#include <cn24.h>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset configuration file>";
    LOGEND;
    return -1;
  }

  Conv::System::Init();


  // Open tensor stream
  std::string dataset_config_file(argv[1]);
  
  std::ifstream dataset_config_fstream(dataset_config_file, std::ios::in);
  if(!dataset_config_fstream.good()) {
    FATAL("Cannot open " << dataset_config_file << "!");
  }

  Conv::ClassManager class_manager;
  Conv::Dataset* dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_fstream, false, Conv::LOAD_BOTH, &class_manager);

  Conv::Tensor data_tensor(1, dataset->GetWidth(), dataset->GetHeight(), dataset->GetInputMaps());
  Conv::Tensor weight_tensor(1, dataset->GetWidth(), dataset->GetHeight(), 1);
  Conv::Tensor label_tensor(1, dataset->GetWidth(), dataset->GetHeight(), dataset->GetLabelMaps());
  Conv::Tensor helper_tensor(1, dataset->GetWidth(), dataset->GetHeight(), 2);

  unsigned int max_class_id = class_manager.GetMaxClassId() + 1;
  long double* pixel_counts = new long double[max_class_id];
  long double* pixel_counts_weighted = new long double[max_class_id];
  for(unsigned int clazz = 0; clazz < max_class_id; clazz++) {
    pixel_counts[clazz] = 0;
    pixel_counts_weighted[clazz] = 0;
  }
  
  for(unsigned int sample = 0; sample < dataset->GetTrainingSamples(); sample++) {
    LOGINFO << "Processing sample " << sample+1 << "/" << dataset->GetTrainingSamples() << std::flush;
    dataset->GetTrainingSample(data_tensor, label_tensor, helper_tensor, weight_tensor, 0, sample);
    for(unsigned int y = 0; y < dataset->GetHeight(); y++) {
      for(unsigned int x = 0; x < dataset->GetWidth(); x++) {
        unsigned int pixel_class = label_tensor.PixelMaximum(x,y,0);
        Conv::datum weight = *weight_tensor.data_ptr_const(x,y);
        
        pixel_counts[pixel_class] += (long double)weight;
        pixel_counts_weighted[pixel_class] += (long double)(weight * class_manager.GetClassInfoById(pixel_class).second.weight);
      }
    }
  }
  
  long double total_pixels = 0;
  long double total_pixels_weighted = 0;
  long double total_classes = 0;
  long double total_classes_weighted = 0;
  for(Conv::ClassManager::const_iterator it = class_manager.begin(); it != class_manager.end(); it++) {
    unsigned int clazz = it->second.id;
    total_pixels += pixel_counts[clazz];
    total_pixels_weighted += pixel_counts_weighted[clazz];
    if(pixel_counts[clazz] > 0)
      total_classes++;
    if(pixel_counts_weighted[clazz] > 0)
      total_classes_weighted++;
  }
  long double expected_ratio = 1.0 / total_classes;
  long double expected_ratio_weighted = 1.0 / total_classes_weighted;
  long double correction_ratio_sum = 0;
  long double correction_ratio_sum_weighted = 0;
  for(Conv::ClassManager::const_iterator it = class_manager.begin(); it != class_manager.end(); it++) {
    unsigned int clazz = it->second.id;
    if(pixel_counts[clazz] > 0)
      correction_ratio_sum += expected_ratio/(pixel_counts[clazz]/total_pixels);
    if(pixel_counts_weighted[clazz] > 0)
      correction_ratio_sum_weighted += expected_ratio_weighted/(pixel_counts_weighted[clazz]/total_pixels_weighted);
  }
  
  // Ignoring weights
  LOGINFO << "Stats when ignoring weights";
  LOGINFO << "===========================";
  LOGINFO << "Classes counted: " << total_classes;
  LOGINFO << "Expected ratio: " << 100.0 * expected_ratio << "%";
  for(Conv::ClassManager::const_iterator it = class_manager.begin(); it != class_manager.end(); it++) {
    unsigned int clazz = it->second.id;
    long double actual_ratio = pixel_counts[clazz]/total_pixels;
    long double correction_ratio = 0;
    if(pixel_counts[clazz] > 0) {
      correction_ratio = expected_ratio / actual_ratio;
    }
    LOGINFO << "Class " << std::setw(30) << it->first << " | " << std::setw(14) << static_cast<long>(pixel_counts[clazz]) << std::setw(14) << 100.0 * actual_ratio << "%" << std::setw(14) << correction_ratio << std::setw(14) << static_cast<long>(correction_ratio * pixel_counts[clazz]);
  }
  
  // Not ignoring weights
  LOGINFO << "Stats when not ignoring weights";
  LOGINFO << "===========================";
  LOGINFO << "Classes counted: " << total_classes_weighted;
  LOGINFO << "Expected ratio: " << 100.0 * expected_ratio_weighted << "%";
  for(Conv::ClassManager::const_iterator it = class_manager.begin(); it != class_manager.end(); it++) {
    unsigned int clazz = it->second.id;
    long double actual_ratio = pixel_counts_weighted[clazz]/total_pixels_weighted;
    long double correction_ratio = 0;
    if(pixel_counts_weighted[clazz] > 0) {
      correction_ratio = expected_ratio_weighted / actual_ratio;
    }
    LOGINFO << "Class " << std::setw(30) << it->first << " | " << std::setw(14) << static_cast<long>(pixel_counts_weighted[clazz]) << std::setw(14) << 100.0 * actual_ratio << "%" << std::setw(14) << correction_ratio << std::setw(14) << static_cast<long>(correction_ratio * pixel_counts_weighted[clazz]);
  }
  

  LOGINFO << "DONE!";
  LOGEND;
  
  return 0;
}

