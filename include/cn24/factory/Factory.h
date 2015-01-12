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
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_FACTORY_H
#define CONV_FACTORY_H

#include "Net.h"
#include "Trainer.h"
#include "ErrorLayer.h"
#include "MultiClassErrorLayer.h"
#include "CrossEntropyErrorLayer.h"

namespace Conv {

#define FACTORY(x,px,py) class x##Factory : public Factory { \
  public:\
    x##Factory(const unsigned int seed = 0) : Factory(seed) { \
      InitOptimalSettings();\
    }; \
    int AddLayers(Net& net, Connection data_layer_connection); \
  Layer* CreateLossLayer(const unsigned int output_classes) {\
  return new ErrorLayer();\
}\
    void InitOptimalSettings(); \
    int patchsizex() { return px; }\
    int patchsizey() { return py; }\
  };
  
#define MFACTORY(x,px,py,ll) class x##Factory : public Factory { \
  public:\
    x##Factory(const unsigned int seed = 0) : Factory(seed) { \
      InitOptimalSettings();\
    }; \
    int AddLayers(Net& net, Connection data_layer_connection, \
      const unsigned int output_classes); \
  Layer* CreateLossLayer(const unsigned int output_classes) {\
  return new ll (output_classes);\
}\
    void InitOptimalSettings(); \
    int patchsizex() { return px; }\
    int patchsizey() { return py; }\
  };
 
  
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
  
  static Factory* getNetFactory(char net_id, const unsigned int seed = 0);

  virtual int patchsizex() = 0;
  virtual int patchsizey() = 0;
protected:
  virtual void InitOptimalSettings() = 0;
  unsigned int seed_ = 0;
  TrainerSettings optimal_settings_;
};

FACTORY (CNetA,24,24)
MFACTORY (CNetM,24,24,CrossEntropyErrorLayer)
MFACTORY (CNetN,24,24,MultiClassErrorLayer)
MFACTORY (CNetO,24,24,MultiClassErrorLayer)
MFACTORY (CNetP,28,28,MultiClassErrorLayer)
MFACTORY (CNetQ,28,28,MultiClassErrorLayer)
MFACTORY (CNetR,28,28,CrossEntropyErrorLayer)
MFACTORY (CNetS,44,44,MultiClassErrorLayer)
}

#endif
