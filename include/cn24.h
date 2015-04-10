/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file cn24.h
 * @brief Includes all the other headers.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CN24_H
#define CONV_CN24_H

#include "cn24/util/Config.h"
#include "cn24/util/Dataset.h"
#include "cn24/util/Tensor.h"
#include "cn24/util/TensorViewer.h"
#include "cn24/util/CombinedTensor.h"
#include "cn24/util/PNGUtil.h"
#include "cn24/util/JPGUtil.h"
#include "cn24/util/Log.h"
#include "cn24/util/KITTIData.h"
#include "cn24/util/Init.h"
#include "cn24/util/GradientTester.h"

#include "cn24/net/Layer.h"
#include "cn24/net/InputLayer.h"
#include "cn24/net/TrainingLayer.h"
#include "cn24/net/SimpleLayer.h"
#include "cn24/net/NonLinearityLayer.h"
#include "cn24/net/DatasetInputLayer.h"
#include "cn24/net/ResizeLayer.h"
#include "cn24/net/ConvolutionLayer.h"
#include "cn24/net/MaxPoolingLayer.h"
#include "cn24/net/UpscaleLayer.h"
#include "cn24/net/LossFunctionLayer.h"
#include "cn24/net/ErrorLayer.h"
#include "cn24/net/StatLayer.h"
#include "cn24/net/BinaryStatLayer.h"
#include "cn24/net/ConfusionMatrixLayer.h"
#include "cn24/net/SpatialPriorLayer.h"
#include "cn24/net/Net.h"
#include "cn24/net/Trainer.h"
#include "cn24/net/NetGraph.h"
#include "cn24/net/NetStatus.h"

#include "cn24/factory/ConfigurableFactory.h"

#endif
