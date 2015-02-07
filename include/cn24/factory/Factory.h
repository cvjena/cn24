/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file Factory.h
 * \class Factory
 * \brief Interface for net factories.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_FACTORY_H
#define CONV_FACTORY_H

#include "Net.h"
#include "Trainer.h"
#include "ErrorLayer.h"
#include "MultiClassErrorLayer.h"
#include "CrossEntropyErrorLayer.h"

namespace Conv {

class Factory {
public:
  Factory (const unsigned int seed = 0) : seed_ (seed) { }

  virtual int AddLayers (Net& net, Connection data_layer_connection) {
    LOGDEBUG << "This factory needs a number of output classes, setting to 1";
    return AddLayers(net, data_layer_connection, 1);
  }
  virtual int AddLayers (Net& net, Connection data_layer_connection,
    const unsigned int output_classes ) {
    LOGDEBUG << "This factory has a preset number of output classes";
    return AddLayers(net, data_layer_connection);
  }
  virtual Layer* CreateLossLayer(const unsigned int output_classes) = 0;
  TrainerSettings optimal_settings() const { return optimal_settings_; }
  
  virtual int patchsizex() = 0;
  virtual int patchsizey() = 0;
  virtual void InitOptimalSettings() = 0;
protected:
  unsigned int seed_ = 0;
  TrainerSettings optimal_settings_;
};

}

#endif
