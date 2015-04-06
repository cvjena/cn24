/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "NetGraph.h"
#include "InputLayer.h"

namespace Conv {

InputLayer::InputLayer ( Tensor& data ) : InputLayer ( data,
      * ( new Tensor(data.samples(), data.width(), data.height(), 2) ) ) {

}

InputLayer::InputLayer ( Tensor& data, Tensor& helper ) {
  // Check if data and helper Tensor match
  if ( data.samples() != helper.samples() ) {
    FATAL ( "Dimensions don't match!" );
  }

  // Create layer outputs that match the dimensions of the user's
  // data and helper Tensors
  data_ = new CombinedTensor ( data.samples(), data.width(), data.height(),
                               data.maps() );

  helper_ = new CombinedTensor ( helper.samples(), helper.width(),
                                 helper.height(), helper.maps() );

  // Save some memory...
  data_->data.Shadow ( data );
  helper_->data.Shadow ( helper );

  LOGDEBUG << "Instance created.";
}

InputLayer::InputLayer ( Tensor& data, Tensor& label, Tensor& helper,
                         Tensor& weight ) {
  // Check if data and helper Tensor match
  if ( data.samples() != helper.samples() ) {
    FATAL ( "Dimensions don't match!" );
  }

  if ( data.samples() != label.samples() ) {
    FATAL ( "Dimensions don't match!" );
  }

  if ( data.samples() != weight.samples() ) {
    FATAL ( "Dimensions don't match!" );
  }

  // Create layer outputs that match the dimensions of the user's
  // data and helper Tensors
  data_ = new CombinedTensor ( data.samples(), data.width(), data.height(),
                               data.maps() );

  label_ = new CombinedTensor ( label.samples(), label.width(), label.height(),
                                label.maps() );

  helper_ = new CombinedTensor ( helper.samples(), helper.width(),
                                 helper.height(), helper.maps() );

  weight_ = new CombinedTensor ( weight.samples(), weight.width(),
                                 weight.height(), weight.maps() );



  // Save some memory...
  data_->data.Shadow ( data );
  label_->data.Shadow ( label );
  helper_->data.Shadow ( helper );
  weight_->data.Shadow ( weight );

  LOGDEBUG << "Instance created.";
}

bool InputLayer::Connect ( const std::vector< CombinedTensor* >& inputs,
                           const std::vector< CombinedTensor* >& outputs,
                           const Net* net ) {
  // Check if inputs were accidentally supplied
  if ( inputs.size() != 0 ) {
    LOGERROR << "Input layer cannot have inputs!";
    return false;
  }

  // Check the number of output nodes
  if ( outputs.size() != 4 && outputs.size() != 3 ) {
    LOGERROR << "Wrong number of output nodes!";
    return false;
  }

  CombinedTensor* output_data = outputs[0];
  CombinedTensor* output_labels = outputs[1];
  CombinedTensor* output_helper = outputs[2];

  if ( outputs.size() > 3 ) {
    CombinedTensor* output_weight = outputs[3];

    if ( output_weight == nullptr ) {
      LOGERROR << "Null pointer output node supplied!";
      return false;
    }
  }

  if ( output_data == nullptr || output_labels == nullptr ||
       output_helper == nullptr ) {
    LOGERROR << "Null pointer output node supplied!";
    return false;
  }

  bool valid = true;
  // FIXME validate completely
  /*    // Compare data Tensor dimensions
      output_data->data.samples() == data_->data.samples() &&
      output_data->data.width() == data_->data.width() &&
      output_data->data.height() == data_->data.height() &&
      output_data->data.maps() == data_->data.maps() &&
      // Check if label tensor has only one element
      output_labels->data.elements() == 1 &&
      // Compare helper data dimensions
      output_helper->data.samples() == helper_->data.samples() &&
      output_helper->data.width() == helper_->data.width() &&
      output_helper->data.height() == helper_->data.height() &&
      output_helper->data.maps() == helper_->data.maps();*/

  if ( !valid ) {
    LOGERROR << "Output nodes failed validation!";
    return false;
  } else {
    return true;
  }
}

bool InputLayer::CreateOutputs ( const std::vector< CombinedTensor* >& inputs,
                                 std::vector< CombinedTensor* >& outputs ) {
  // Check if inputs were accidentally supplied
  if ( inputs.size() != 0 ) {
    LOGERROR << "Input layer cannot have inputs!";
    return false;
  }

  // Tell the network about our outputs
  outputs.push_back ( data_ );

  if ( label_ != nullptr ) {
    outputs.push_back ( label_ );
  } else {
    outputs.push_back ( new CombinedTensor ( data_->data.samples() ) );
  }

  outputs.push_back ( helper_ );

  if ( weight_ != nullptr ) {
    outputs.push_back ( weight_ );
  }

  return true;
}

void InputLayer::CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers) {
	NetGraphBuffer data_buffer;
	NetGraphBuffer label_buffer;
	NetGraphBuffer helper_buffer;
	NetGraphBuffer weight_buffer;
	data_buffer.description = "Data Output";
	label_buffer.description = "Label";
	helper_buffer.description = "Helper";
	weight_buffer.description = "Weight";
	buffers.push_back(data_buffer);
	buffers.push_back(label_buffer);
	buffers.push_back(helper_buffer);
	buffers.push_back(weight_buffer);
}

}
