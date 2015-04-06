/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file ConfigurableFactory.h
 * @class ConfigurableFactory
 * @brief This class can parse network configuration files and construct network layers.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CONFIGURABLEFACTORY_H
#define CONV_CONFIGURABLEFACTORY_H

#include <iostream>

#include "Trainer.h"
#include "Net.h"
#include "Dataset.h"
#include "Log.h"
#include "NetGraph.h"

namespace Conv {

class ConfigurableFactory {
public:
  /**
	* @brief Builds a ConfigurableFactory using an input stream and a random seed
	*
	* @param file The input stream to read the configuration from
	* @param seed The random seed used to initialize the layers
        * @param is_training_factory This needs to be true for ConfigurableFactory
        *   to read the "method=" setting
	*/
  ConfigurableFactory(std::istream& file, const unsigned seed = 0, bool is_training_factory = false);
  
  /**
	* @brief Adds the configured layers to a network using the specified input layer
	*
	* @param net The net to add the layers to
	* @param data_layer_connection Input to the first layer of this configuration
	* @param output_classes The number of output neurons. This also affects the activation function of
	* 	 the last layer: for output_classes=1, tanh is used. Otherwise, sigm is used.
	* @param add_loss_layer If set to true, the factory also adds a matching loss layer
	* @param graph_output An output stream. The factory will write the layout in graphviz format into this string.
	* 
	* @returns The layer id of the output layer
	*/
  virtual int AddLayers(Net& net, Connection data_layer_connection, const unsigned int output_classes, bool add_loss_layer = false, std::ostream& graph_output = std::cout);

  virtual bool AddLayers(NetGraph& graph, NetGraphConnection data_layer_connection, const unsigned int output_classes, bool add_loss_layer = false);

  /**
	* @returns The horizontal size of the receptive field
	*/
  virtual int patchsizex() { return receptive_field_x_; }

  /**
	* @returns The vertical size of the receptive field
	*/
  virtual int patchsizey() { return receptive_field_y_; }

  /**
	* @brief Create a loss layer for this configuration
	*
	* @param output_classes Number of output neurons
	* @returns Pointer to the layer instance of the created loss layer
	*/
  virtual Layer* CreateLossLayer(const unsigned int output_classes);

  /**
	* @brief Read the optimal training settings from the configuration file
	*/
  virtual void InitOptimalSettings();


  /**
	* @returns The optimal training settings for this configuration
	*/
  TrainerSettings optimal_settings() const { return optimal_settings_; } 
  
  /**
   * @returns The current method for this net
   */
  Method method() const { return method_; }
private:
	void WriteNode(std::ostream& graph_output, Layer* layer, int source_id, int source_port, int node_id, int outputs);
  Method method_;
  
  int receptive_field_x_ = 0;
  int receptive_field_y_ = 0;

  int patch_field_x_ = 0;
  int patch_field_y_ = 0;
  
  std::istream& file_;
  
  int factorx = 1;
  int factory = 1;

  unsigned int seed_ = 0;
  TrainerSettings optimal_settings_;
};
  
}

#endif
