/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"
#include <iomanip>
namespace Conv {
  
CN24_SHELL_FUNC_IMPL(DataList) {
  CN24_SHELL_FUNC_DESCRIPTION("Lists all currently loaded data");
  CN24_SHELL_PARSE_ARGS;
  
  std::cout << std::endl;
  std::cout << std::setw(10) << "Area" << std::setw(30) << "Bundle";
  std::cout << std::setw(30) << "Segment" << std::setw(10) << "Samples" <<
  std::endl;
  
  // List training area
  std::cout << std::setw(10) << "Training" << std::endl;
  for(unsigned int b = 0; b < training_bundles_->size(); b++) {
    Bundle* bundle = training_bundles_->at(b);
    std::cout << std::setw(10) << "|" << std::setfill('.') << std::setw(30) << bundle->name << std::setfill(' ');
    std::cout << std::setw(40) << bundle->GetSampleCount() << std::endl;
    std::cout << std::setw(10) << "|" << std::setfill('.') << std::setw(25) << "Weight:" << std::setfill(' ') << std::setw(5) << training_weights_->at(b) << std::endl;
    for(unsigned int s = 0; s < bundle->GetSegmentCount(); s++) {
      Segment* segment = bundle->GetSegment(s); 
      if(b == training_bundles_->size() - 1) {
        std::cout << std::setw(40);
      } else {
        std::cout << std::setw(10) << "|" << std::setw(30);
      }
      std::cout << "|" << std::setfill('.') << std::setw(30) << segment->name << std::setfill(' ');
      std::cout << std::setw(10) << segment->GetSampleCount() << std::endl;
    }
  }
  std::cout << std::endl;
  
  // List staging and testing areas
  std::cout << std::setw(10) << "Staging" << std::endl;
  for(unsigned int b = 0; b < staging_bundles_->size(); b++) {
    Bundle* bundle = staging_bundles_->at(b);
    std::cout << std::setw(10) << "|" << std::setfill('.') << std::setw(30) << bundle->name << std::setfill(' ');
    std::cout << std::setw(40) << bundle->GetSampleCount() << std::endl;
    for(unsigned int s = 0; s < bundle->GetSegmentCount(); s++) {
      Segment* segment = bundle->GetSegment(s); 
      if(b == staging_bundles_->size() - 1) {
        std::cout << std::setw(40);
      } else {
        std::cout << std::setw(10) << "|" << std::setw(30);
      }
      std::cout << "|" << std::setfill('.') << std::setw(30) << segment->name << std::setfill(' ');
      std::cout << std::setw(10) << segment->GetSampleCount() << std::endl;
    }
  }
  std::cout << std::endl;
  
  std::cout << std::setw(10) << "Testing" << std::endl;
  for(unsigned int b = 0; b < testing_bundles_->size(); b++) {
    Bundle* bundle = testing_bundles_->at(b);
    std::cout << std::setw(10) << "|" << std::setfill('.') << std::setw(30) << bundle->name << std::setfill(' ');
    std::cout << std::setw(40) << bundle->GetSampleCount() << std::endl;
    for(unsigned int s = 0; s < bundle->GetSegmentCount(); s++) {
      Segment* segment = bundle->GetSegment(s); 
      if(b == testing_bundles_->size() - 1) {
        std::cout << std::setw(40);
      } else {
        std::cout << std::setw(10) << "|" << std::setw(30);
      }
      std::cout << "|" << std::setfill('.') << std::setw(30) << segment->name << std::setfill(' ');
      std::cout << std::setw(10) << segment->GetSampleCount() << std::endl;
    }
  }
  std::cout << std::endl;
  return SUCCESS;
}
}