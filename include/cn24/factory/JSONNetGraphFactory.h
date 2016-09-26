/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file NetGraphFactory.h
 * @class NetGraphFactory
 * @brief This class can parse network configuration files and construct network layers.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_JSONNETGRAPHFACTORY_H
#define CONV_JSONNETGRAPHFACTORY_H

#include <iostream>

#include "../net/NetGraph.h"
#include "../net/Trainer.h"
#include "../util/Dataset.h"
#include "../util/Log.h"
#include "../util/JSONParsing.h"

namespace Conv {
  
class JSONNetGraphFactory {
public:
  explicit JSONNetGraphFactory(std::istream& file, unsigned int seed = 0) : seed_(seed) {
    JSON file_json = JSON::parse(file);
    net_json_ = file_json["net"];
    hyperparameters_json_ = file_json["hyperparameters"];
  }

  explicit JSONNetGraphFactory(JSON json, unsigned int seed = 0) : seed_(seed) {
    net_json_ = json["net"];
    hyperparameters_json_ = json["hyperparameters"];
  }

  JSON GetHyperparameters() {
    return hyperparameters_json_;
  }

  bool AddLayers(NetGraph& graph, ClassManager* class_manager, unsigned int seed = 0);

private:
  JSON net_json_;
  JSON hyperparameters_json_;
  unsigned int seed_ = 0;
};
  
}

#endif
