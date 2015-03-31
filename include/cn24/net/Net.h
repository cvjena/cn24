/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Net.h
 * @class Net
 * @brief This is a connected collection of Layers.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_NET_H
#define CONV_NET_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "Layer.h"
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
  Connection () : net(0), output(0) {}
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
   * @brief Adds a layer to the network.
   *
   * @param layer The layer to add
   * @param connections The inputs to the layer
   * @returns The id of the layer in the network
   */
  int AddLayer (Layer* layer, const std::vector<Connection>& connections =
                  std::vector<Connection>());
  
  /**
   * @brief Adds a layer to the network.
   * 
   * @param layer The layer to add
   * @param input_layer The input to the layer (output 0 is used)
   * @returns The id of the layer in the network
   */
  int AddLayer (Layer* layer, const int input_layer);
  
  /**
   * @brief Initializes the weights.
   */
  void InitializeWeights();
  
  /**
   * @brief Complete forward pass.
   * 
   * Calls every Layer's FeedForward function.
   */
  void FeedForward();

  /**
   * @brief Forward pass up to the specified layer
   * 
	* @param last Layer id of the last layer to process
   */
  void FeedForward(const unsigned int last);
  
  /**
   * @brief Complete backward pass.
   * 
   * Calls every Layer's BackPropagate function.
   */
  void BackPropagate();
  
  /**
   * @brief Collects every Layer's parameters.
   * 
   * @param parameters Vector to store the parameters in
   */
  void GetParameters(std::vector<CombinedTensor*>& parameters);
  
  /**
   * @brief Writes the params to a Tensor file.
   * 
   * @param output Stream to write the Tensors to
   */
  void SerializeParameters(std::ostream& output);
  
  /**
   * @brief Reads the parameters from a Tensor file.
   * 
   * @param input Stream to read the Tensors from
   * @param last_layer The id of the last layer to load parameters into,
   *   for fine-tuning. Set to zero for all layers.
   */
  void DeserializeParameters(std::istream& input, unsigned int last_layer = 0);
  
  /**
   * @brief Gets the training layer.
   */
  inline TrainingLayer* training_layer() {
    return training_layer_;
  }
  
  /**
   * @brief Gets the loss function layer.
   */
  inline LossFunctionLayer* lossfunction_layer() {
    return lossfunction_layer_;
  }   
  
  /**
   * @brief Gets the stat layers.
   */
  inline std::vector<StatLayer*>& stat_layers() {
    return stat_layers_;
  }
  
  /**
   * @brief Gets the binary stat layer.
   */
  inline BinaryStatLayer* binary_stat_layer() {
    return binary_stat_layer_;
  }
  
  /**
   * @brief Gets the confusion matrix layer.
   */
  inline ConfusionMatrixLayer* confusion_matrix_layer() {
    return confusion_matrix_layer_;
  }
  
  /**
   * @brief Gets the layer with the corresponding id
   */
  inline Layer* layer(int layer_id) const {
    return layers_[layer_id];
  }
  
  /**
   * @brief Returns the output buffer of the given layer
   */
  inline CombinedTensor* buffer(int layer_id, int buffer_id = 0) const {
    return buffers_[layer_id][buffer_id];
  }
  
  /**
   * @brief Enables or disables the binary stat layer
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
   * @brief Returns true if the net is currently testing
   */
  inline bool IsTesting() const { return is_testing_; } 
  
  /**
   * @brief Sets this net's testing status
   * 
   * @param is_testing The new testing status
   */
  inline void SetIsTesting(bool is_testing) { is_testing_ = is_testing; }
  /**
   * @brief Enables the built-in layer view GUI. Needs CMake build option.
   */
  inline void SetLayerViewEnabled(const bool enabled = true) {
    LOGDEBUG << "Layer view enabled: " << enabled;
    layer_view_enabled_ = enabled;
  }

  void PrintAndResetLayerTime(datum samples);
private:
  TrainingLayer* training_layer_ = nullptr; 
  LossFunctionLayer* lossfunction_layer_ = nullptr;
  BinaryStatLayer* binary_stat_layer_ = nullptr;
  ConfusionMatrixLayer* confusion_matrix_layer_ = nullptr;
  std::vector<StatLayer*> stat_layers_;
  std::vector<Layer*> layers_;
  std::vector<std::vector<CombinedTensor*>> buffers_;
  std::vector<std::vector<CombinedTensor*>> inputs_;
  std::vector<std::pair<Layer*, Layer*>> weight_connections_;
  
  bool layer_view_enabled_ = false;
  
  std::chrono::duration<double>* forward_durations_ = nullptr;
  std::chrono::duration<double>* backward_durations_ = nullptr;
  
  bool is_testing_ = false;
};

}

#endif
