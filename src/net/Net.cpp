/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include "Log.h"
#include "Init.h"

#include "TensorViewer.h"

#include <sstream>

#include "Net.h"

namespace Conv {

int Net::AddLayer (Layer* layer, const std::vector< Connection >& connections) {
  // Check for null pointer
  if (layer == nullptr) {
    FATAL ("Null pointer supplied");
    return -1;
  }

  // Determine Layer's id
  int layer_id = layers_.size();

  // Add the layer
  layers_.push_back (layer);

  // Get inputs
  std::vector<CombinedTensor*> inputs;
  for (unsigned int i = 0; i < connections.size(); i++) {
    Connection connection = connections[i];
    CombinedTensor* buffer = buffers_[connection.net][connection.output];
    inputs.push_back (buffer);
    LOGDEBUG << "Layer " << layer_id << " input: layer " << connection.net <<
             ", output " << connection.output;
    if (i == 0 && connection.output == 0) {
      // Tell layer below
      Layer* below = layers_[connection.net];
      weight_connections_.push_back ( {below, layer});
    }
  }
  // These names are bad. inputs_ contains the input buffers for all the layers
  // and inputs contains the input buffers for the currently added layer.
  inputs_.push_back (inputs);

  // Ask the layer to create an output buffer
  std::vector<CombinedTensor*> outputs;
  bool result = layer->CreateOutputs (inputs, outputs);
  if (!result) {
    FATAL ("Layer will not create output buffer!");
    return -1;
  }

  for (unsigned int i = 0; i < outputs.size(); i++) {
    CombinedTensor* output = outputs[i];
    LOGDEBUG << "Layer " << layer_id << " output " << i << ": " <<
             output->data;
  }

  // Connect the layer
  bool connection_result = layer->Connect (inputs, outputs);
  if (!connection_result) {
    FATAL ("Layer failed to connect!");
    return -1;
  }

  // Save outputs
  buffers_.push_back (outputs);

  LOGDEBUG << "Layer " << layer_id << " added.";

#ifdef BUILD_OPENCL
  if (layer->IsOpenCLAware()) {
    LOGDEBUG << "Layer " << layer_id << " is OpenCL aware";
  } else {
    LOGWARN << "Layer " << layer_id << " is NOT OpenCL aware";
  }
#endif

  // Check if layer supports training
  if (dynamic_cast<TrainingLayer*> (layer) != NULL) {
    // If it does, save the pointer
    if (training_layer_ == nullptr) {
      LOGDEBUG << "Layer " << layer_id << " added as training layer.";
      training_layer_ = dynamic_cast<TrainingLayer*> (layer);
    } else {
      FATAL ("Cannot add another training layer!");
      return -1;
    }
  }

  // Check if layer is a binary stat layer
  if (dynamic_cast<BinaryStatLayer*> (layer) != NULL) {
    // If it is, save the pointer
    if (binary_stat_layer_ == nullptr) {
      LOGDEBUG << "Layer " << layer_id << " added as binary stat layer.";
      binary_stat_layer_ = dynamic_cast<BinaryStatLayer*> (layer);
    } else {
      FATAL ("Cannot add another binary stat layer!");
      return -1;
    }
  }

  // Check if layer is a confusion matrix layer
  if (dynamic_cast<ConfusionMatrixLayer*> (layer) != NULL) {
    // If it is, save the pointer
    if (confusion_matrix_layer_ == nullptr) {
      LOGDEBUG << "Layer " << layer_id << " added as confusion matrix layer.";
      confusion_matrix_layer_ = dynamic_cast<ConfusionMatrixLayer*> (layer);
    } else {
      FATAL ("Cannot add another confusion matrix layer!");
      return -1;
    }
  }

  // Check if layer is loss function layer
  if (dynamic_cast<LossFunctionLayer*> (layer) != NULL) {
    // If it is, save the pointer
    if (lossfunction_layer_ == nullptr) {
      LOGDEBUG << "Layer " << layer_id << " added as loss function layer.";
      lossfunction_layer_ = dynamic_cast<LossFunctionLayer*> (layer);
    } else {
      FATAL ("Cannot add another loss function layer!");
      return -1;
    }
  }


  // Check if layer is a stat layer
  if (dynamic_cast<StatLayer*> (layer) != NULL) {
    // If it is, add to vector
    StatLayer* stat_layer = dynamic_cast<StatLayer*> (layer);
    stat_layers_.push_back (stat_layer);

    LOGDEBUG << "Layer " << layer_id << " added as stat layer: " <<
             stat_layer->stat_name();
  }

  // Return the layer number
  return layer_id;
}

int Net::AddLayer (Layer* layer, const int input_layer) {
  return AddLayer (layer, {Connection (input_layer) });
}


void Net::InitializeWeights() {
  for (int l = weight_connections_.size() - 1; l > 0; l--) {
    std::pair<Layer*, Layer*> p = weight_connections_[l];
    p.first->OnLayerConnect (p.second);
  }
}



void Net::FeedForward() {
#ifdef LAYERTIME
  if (forward_durations_ == nullptr) {
    forward_durations_ = new std::chrono::duration<double>[layers_.size()];
    backward_durations_ = new std::chrono::duration<double>[layers_.size()];

    for (unsigned int l = 0; l < layers_.size(); l++) {
      forward_durations_[l] = std::chrono::duration<double>::zero();
      backward_durations_[l] = std::chrono::duration<double>::zero();
    }
  }
#endif
  for (unsigned int l = 0; l < layers_.size(); l++) {
    Layer* layer = layers_[l];

#ifdef LAYERTIME
    auto t_begin = std::chrono::system_clock::now();
#endif

#ifdef BUILD_OPENCL
    if (!layer->IsOpenCLAware()) {
      for (unsigned int i = 0; i < inputs_[l].size(); i++) {
        inputs_[l][i]->data.MoveToCPU();
        inputs_[l][i]->delta.MoveToCPU();
      }

      for (unsigned int i = 0; i < buffers_[l].size(); i++) {
        buffers_[l][i]->data.MoveToCPU();
        buffers_[l][i]->delta.MoveToCPU();
      }
    }
#endif
    Tensor* output0 = nullptr;
    if(buffers_[l].size() > 0)
      output0 = &(buffers_[l][0]->data);
    layer->FeedForward();
    
#ifdef LAYERVIEW
    if(output0 != nullptr && layer_view_enabled_) {
      std::stringstream sstr;
      sstr << "Tensor Viewer: Layer " << l;
#ifdef BUILD_OPENCL
      output0->MoveToCPU();
#endif
      System::viewer->show(output0, sstr.str());
    }
#endif
    
    output0 = nullptr;

#ifdef LAYERTIME
    auto t_end = std::chrono::system_clock::now();
    forward_durations_[l] += t_end - t_begin;
#endif
  }
}

void Net::FeedForward (const unsigned int last) {
  for (unsigned int l = 0; l <= last; l++) {
    Layer* layer = layers_[l];
#ifdef BUILD_OPENCL
    if (!layer->IsOpenCLAware()) {
      for (unsigned int i = 0; i < inputs_[l].size(); i++) {
        inputs_[l][i]->data.MoveToCPU();
        inputs_[l][i]->delta.MoveToCPU();
      }

      for (unsigned int i = 0; i < buffers_[l].size(); i++) {
        buffers_[l][i]->data.MoveToCPU();
        buffers_[l][i]->delta.MoveToCPU();
      }
    }
#endif
    layer->FeedForward();
  }
}


void Net::BackPropagate() {
  for (int l = (layers_.size() - 1); l >= 0; l--) {
    Layer* layer = layers_[l];

#ifdef LAYERTIME
    auto t_begin = std::chrono::system_clock::now();
#endif

#ifdef BUILD_OPENCL
    if (!layer->IsOpenCLAware()) {
      for (unsigned int i = 0; i < inputs_[l].size(); i++) {
        inputs_[l][i]->data.MoveToCPU();
        inputs_[l][i]->delta.MoveToCPU();
      }

      for (unsigned int i = 0; i < buffers_[l].size(); i++) {
        buffers_[l][i]->data.MoveToCPU();
        buffers_[l][i]->delta.MoveToCPU();
      }
    }
#endif
    layer->BackPropagate();

#ifdef LAYERTIME
    auto t_end = std::chrono::system_clock::now();
    backward_durations_[l] += t_end - t_begin;
#endif

  }
}

void Net::GetParameters (std::vector< CombinedTensor* >& parameters) {
  for (unsigned int l = 0; l < layers_.size(); l++) {
    Layer* layer = layers_[l];
    for (unsigned int p = 0; p < layer->parameters().size(); p++) {
      parameters.push_back (layer->parameters() [p]);
    }
  }
}

void Net::SerializeParameters (std::ostream& output) {
  for (unsigned int l = 0; l < layers_.size(); l++) {
    Layer* layer = layers_[l];
    for (unsigned int p = 0; p < layer->parameters().size(); p++) {
      layer->parameters() [p]->data.Serialize (output);
    }
  }
}

void Net::DeserializeParameters (std::istream& input, unsigned int last_layer) {
  if (last_layer == 0 || last_layer >= layers_.size())
    last_layer = layers_.size() - 1;
  for (unsigned int l = 0; l <= last_layer; l++) {
    Layer* layer = layers_[l];
    for (unsigned int p = 0; p < layer->parameters().size(); p++) {
      if (!input.good() || input.eof())
        break;
      layer->parameters() [p]->data.Deserialize (input);
      LOGINFO << "Loaded parameters for layer " << l << " parameter set " << p << ": " << layer->parameters()[p]->data;
      input.peek();
    }
  }
}

void Net::PrintAndResetLayerTime(datum samples) {
#ifdef LAYERTIME
    std::cout << std::endl << "LAYERTIME (" << samples << ")" << std::endl;
    datum tps_sum = 0.0;
    for(unsigned int l = 0; l < layers_.size(); l++) {
      std::cout << "forward " << l << "," << std::fixed << std::setprecision(9) << 1000000.0 * forward_durations_[l].count() / samples << "\n";
      std::cout << "backwrd " << l << "," << std::fixed << std::setprecision(9) << 1000000.0 * backward_durations_[l].count() / samples << "\n";
      tps_sum += 1000000.0 * forward_durations_[l].count() / samples;
      tps_sum += 1000000.0 * backward_durations_[l].count() / samples;
	   forward_durations_[l] = std::chrono::duration<double>::zero();
	   backward_durations_[l] = std::chrono::duration<double>::zero();
    }
    
    std::cout << "Total tps in net: " << tps_sum << " us" << std::endl;
#endif
  }

}
