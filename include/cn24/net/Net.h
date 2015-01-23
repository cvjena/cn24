/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file Net.h
 * \class Net
 * \brief This is a connected collection of Layers.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_NET_H
#define CONV_NET_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "Layer.h"
#include "DropoutLayer.h"
#include "LossFunctionLayer.h"
#include "TrainingLayer.h"
#include "StatLayer.h"
#include "BinaryStatLayer.h"
#include "ConfusionMatrixLayer.h"

namespace Conv {

class Trainer;
class GradientTester;

struct Connection {
public:
  Connection (const int net, const int output = 0) :
    net (net), output (output) { }
  int net;
  int output;
};

class Net {
  friend class Trainer;
  friend class GradientTester;
public:
  /**
   * \brief Adds a layer to the network.
   *
   * \param layer The layer to add
   * \param connections The inputs to the layer
   * \returns The id of the layer in the network
   */
  int AddLayer (Layer* layer, const std::vector<Connection>& connections =
                  std::vector<Connection>());
  
  /**
   * \brief Adds a layer to the network.
   * 
   * \param layer The layer to add
   * \param input_layer The input to the layer (output 0 is used)
   * \returns The id of the layer in the network
   */
  int AddLayer (Layer* layer, const int input_layer);
  
  /**
   * \brief Initializes the weights.
   */
  void InitializeWeights();
  
  /**
   * \brief Complete forward pass.
   * 
   * Calls every Layer's FeedForward function.
   */
  void FeedForward();
  void FeedForward(const unsigned int last);
  
  /**
   * \brief Complete backward pass.
   * 
   * Calls every Layer's BackPropagate function.
   */
  void BackPropagate();
  
  /**
   * \brief Collects every Layer's parameters.
   * 
   * \param parameters Vector to store the parameters in
   */
  void GetParameters(std::vector<CombinedTensor*>& parameters);
  
  /**
   * \brief Writes the params to a Tensor file.
   * 
   * \param output Stream to write the Tensors to
   */
  void SerializeParameters(std::ostream& output);
  
  /**
   * \brief Reads the parameters from a Tensor file.
   * 
   * \param input Stream to read the Tensors from
   */
  void DeserializeParameters(std::istream& input);
  
  /**
   * \brief Gets the training layer.
   */
  inline TrainingLayer* training_layer() {
#ifndef NDEBUG
    if(training_layer_ == nullptr) {
      FATAL ("Null pointer requested!");
    }
#endif
    return training_layer_;
  }
  
  /**
   * \brief Gets the loss function layer.
   */
  inline LossFunctionLayer* lossfunction_layer() {
#ifndef NDEBUG
    if(lossfunction_layer_ == nullptr) {
      FATAL ("Null pointer requested!");
    }
#endif
    return lossfunction_layer_;
  }   
  
  /**
   * \brief Gets the stat layers.
   */
  inline std::vector<StatLayer*>& stat_layers() {
    return stat_layers_;
  }
  
  /**
   * \brief Gets the binary stat layer.
   */
  inline BinaryStatLayer* binary_stat_layer() {
    return binary_stat_layer_;
  }
  
  /**
   * \brief Gets the confusion matrix layer.
   */
  inline ConfusionMatrixLayer* confusion_matrix_layer() {
    return confusion_matrix_layer_;
  }
  
  /**
   * \brief Gets the layer with the corresponding id
   */
  inline Layer* layer(int layer_id) const {
    return layers_[layer_id];
  }
  
  /**
   * \brief Returns the output buffer of the given layer
   */
  inline CombinedTensor* buffer(int layer_id, int buffer_id = 0) const {
    return buffers_[layer_id][buffer_id];
  }
  
  /**
   * \brief Enables or disables dropout in the net
   */
  inline void SetDropoutEnabled(const bool do_dropout) {
    for(unsigned int i = 0; i < dropout_layers_.size(); i++) {
      dropout_layers_[i]->SetDropoutEnabled(do_dropout);
    }
  }
  
  /**
   * \brief Enables or disables the binary stat layer
   */
  inline void SetTestOnlyStatDisabled(const bool disabled = false) {
    if(binary_stat_layer_ != nullptr) {
      LOGDEBUG << "Binary stat layer disabled: " << disabled;
      binary_stat_layer_->SetDisabled(disabled);
    }
    
    if(confusion_matrix_layer_ != nullptr) {
      LOGDEBUG << "Confusion matrix layer disabled: " << disabled;
      confusion_matrix_layer_->SetDisabled(disabled);
    }
  }
  
  
  /**
   * \brief Enables the built-in layer view GUI. Needs CMake build option.
   */
  inline void SetLayerViewEnabled(const bool enabled = true) {
    LOGDEBUG << "Layer view enabled: " << enabled;
    layer_view_enabled_ = enabled;
  }
#ifdef LAYERTIME
  void PrintAndResetLayerTime(datum samples) {
    std::cout << std::endl << "LAYERTIME (" << samples << ")" << std::endl;
    datum tps_sum = 0.0;
    for(unsigned int l = 0; l < layers_.size(); l++) {
      std::cout << "forward " << l << "," << std::fixed << std::setprecision(9) << 1000000.0 * forward_durations_[l].count() / samples << "\n";
      std::cout << "backwrd " << l << "," << std::fixed << std::setprecision(9) << 1000000.0 * backward_durations_[l].count() / samples << "\n";
      tps_sum += 1000000.0 * forward_durations_[l].count() / samples;
      tps_sum += 1000000.0 * backward_durations_[l].count() / samples;
/*      LOGDEBUG << "Layer " << l << " forward time : " <<
	forward_durations_[l].count();
      LOGDEBUG << "Layer " << l << " backwrd time : " <<
	backward_durations_[l].count();*/
	
	forward_durations_[l] = std::chrono::duration<double>::zero();
	backward_durations_[l] = std::chrono::duration<double>::zero();
    }
    
    std::cout << "Total tps in net: " << tps_sum << " us" << std::endl;
  }
#endif

private:
  TrainingLayer* training_layer_ = nullptr; 
  LossFunctionLayer* lossfunction_layer_ = nullptr;
  BinaryStatLayer* binary_stat_layer_ = nullptr;
  ConfusionMatrixLayer* confusion_matrix_layer_ = nullptr;
  std::vector<StatLayer*> stat_layers_;
  std::vector<Layer*> layers_;
  std::vector<DropoutLayer*> dropout_layers_;
  std::vector<std::vector<CombinedTensor*>> buffers_;
  std::vector<std::vector<CombinedTensor*>> inputs_;
  std::vector<std::pair<Layer*, Layer*>> weight_connections_;
  
  bool layer_view_enabled_ = false;
  
#ifdef LAYERTIME
  std::chrono::duration<double>* forward_durations_ = nullptr;
  std::chrono::duration<double>* backward_durations_ = nullptr;
#endif
  
};

}

#endif
