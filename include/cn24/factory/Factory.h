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

namespace Conv {

class Factory {
public:
  Factory (const unsigned int seed = 0) : seed_ (seed) { }

  virtual int AddLayers (Net& net, Connection data_layer_connection, unsigned int output_classes) = 0;
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
