
/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file ConfigurableFactory.h
 * \class ConfigurableFactory
 * \brief This class can parse network configuration files.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CONFIGURABLEFACTORY_H
#define CONV_CONFIGURABLEFACTORY_H

#include <iostream>

#include "Dataset.h"
#include "Log.h"

#include "Factory.h"

namespace Conv {

class ConfigurableFactory : public Factory {
public:
  explicit ConfigurableFactory(std::istream& file, Method method, const unsigned seed = 0);
  
  virtual int AddLayers(Net& net, Connection data_layer_connection) {
    LOGDEBUG << "Assuming one class only!";
    return AddLayers(net, data_layer_connection, 1);
  }
  virtual int AddLayers(Net& net, Connection data_layer_connection, const unsigned int output_classes);

  virtual int patchsizex() { return receptive_field_x_; }
  virtual int patchsizey() { return receptive_field_y_; }

  virtual Layer* CreateLossLayer(const unsigned int output_classes);

  virtual void InitOptimalSettings();
private:
  Method method_;
  
  int receptive_field_x_ = 0;
  int receptive_field_y_ = 0;

  int patch_field_x_ = 0;
  int patch_field_y_ = 0;
  
  std::istream& file_;
  
  int factorx = 1;
  int factory = 1;

};
  
}

#endif
