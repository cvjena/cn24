#ifndef CONV_SKIPLAYERNETWORKFACTORY_H
#define CONV_SKIPLAYERNETWORKFACTORY_H

#include <iostream>

#include "../net/Net.h"
#include "../net/NetGraph.h"
#include "../net/Trainer.h"
#include "../util/Dataset.h"
#include "../util/Log.h"
#include "ConfigurableFactory.h"

namespace Conv {

class SkipLayerNetworkFactory : public Factory {
  int AddLayers(Net& net, Connection data_layer_connection, const unsigned int output_classes, bool add_loss_layer = false, std::ostream& graph_output = std::cout);
  bool AddLayers(NetGraph& graph, NetGraphConnection data_layer_connection, const unsigned int output_classes, bool add_loss_layer = false);
  int patchsizex();
  int patchsizey();
  Layer* CreateLossLayer(const unsigned int output_classes, const datum loss_weight = 1.0);
  void InitOptimalSettings();
  TrainerSettings optimal_settings() const;
  Method method() const;
};
  
}

#endif