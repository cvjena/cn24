/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cstdio>

#include "ErrorLayer.h"

#include "ConvolutionLayer.h"
#include "LocalResponseNormalizationLayer.h"
#include "ResizeLayer.h"
#include "MaxPoolingLayer.h"
#include "AdvancedMaxPoolingLayer.h"
#include "InputDownSamplingLayer.h"
#include "NonLinearityLayer.h"
#include "UpscaleLayer.h"
#include "SpatialPriorLayer.h"
#include "ConcatenationLayer.h"
#include "ConfigParsing.h"
#include "NetGraph.h"

#include "SkipLayerNetworkFactory.h"

namespace Conv {

bool SkipLayerNetworkFactory::AddLayers(NetGraph& graph, NetGraphConnection data_layer_connection, const unsigned int output_classes, bool add_loss_layer)
{
  return false;
}

int SkipLayerNetworkFactory::AddLayers(Net& net, Connection data_layer_connection, const unsigned int output_classes, bool add_loss_layer, std::ostream& graph_output)
{
  return 0;
}

Layer* SkipLayerNetworkFactory::CreateLossLayer(const unsigned int output_classes, const datum loss_weight)
{
  return nullptr;
}

void SkipLayerNetworkFactory::InitOptimalSettings()
{
  
}

Method SkipLayerNetworkFactory::method() const
{
  return Method::FCN;
}

TrainerSettings SkipLayerNetworkFactory::optimal_settings() const
{
  TrainerSettings s;
  return s;
}

int SkipLayerNetworkFactory::patchsizex()
{
  return 0;
}

int SkipLayerNetworkFactory::patchsizey()
{
  return 0;
}

}
