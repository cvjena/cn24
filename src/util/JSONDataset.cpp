/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include "JSONParsing.h"

#include "Dataset.h"

namespace Conv {

	
JSONDataset::JSONDataset() {
	
}

JSONDataset::~JSONDataset() {
	
}

void JSONDataset::LoadFile(std::istream& file, bool dont_load, DatasetLoadSelection selection) {
	// TODO Validate JSON
	
	JSON dataset_json = JSON::parse(file);
	if(is_first_dataset) {
		
	} else {
		
	}
}	


bool JSONDataset::GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
	return false;
}

bool JSONDataset::GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index) {
	return false;
}

}