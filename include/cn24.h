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
#include "cn24/util/CompressedTensor.h"
#include "cn24/util/TensorViewer.h"
#include "cn24/util/CombinedTensor.h"
#include "cn24/util/TensorStream.h"
#include "cn24/util/CompressedTensorStream.h"
#include "cn24/util/FloatTensorStream.h"
#include "cn24/util/ListTensorStream.h"
#include "cn24/util/PNGUtil.h"
#include "cn24/util/JPGUtil.h"
#include "cn24/util/Log.h"
#include "cn24/util/KITTIData.h"
#include "cn24/util/Init.h"
#include "cn24/util/GradientTester.h"
#include "cn24/util/StatAggregator.h"
#include "cn24/util/StatSink.h"
#include "cn24/util/ConsoleStatSink.h"
#include "cn24/util/CSVStatSink.h"
#include "cn24/util/JSONParsing.h"
#include "cn24/util/MNISTDataset.h"
#include "cn24/util/MemoryMappedFile.h"
#include "cn24/util/MemoryMappedTar.h"
#include "cn24/util/BoundingBox.h"
#include "cn24/util/Test.h"
#include "cn24/util/ClassManager.h"
#include "cn24/util/Segment.h"
#include "cn24/util/SegmentSet.h"
#include "cn24/util/PathFinder.h"
#include "cn24/util/TensorRegistry.h"

#include "cn24/math/TensorMath.h"
#include "cn24/math/Optimizer.h"
#include "cn24/math/SGDOptimizer.h"
#include "cn24/math/AdamOptimizer.h"

#include "cn24/net/Layer.h"
#include "cn24/net/InputLayer.h"
#include "cn24/net/TrainingLayer.h"
#include "cn24/net/SimpleLayer.h"
#include "cn24/net/NonLinearityLayer.h"
#include "cn24/net/DatasetInputLayer.h"
#include "cn24/net/SegmentSetInputLayer.h"
#include "cn24/net/ResizeLayer.h"
#include "cn24/net/ConvolutionLayer.h"
#include "cn24/net/DropoutLayer.h"
#include "cn24/net/MaxPoolingLayer.h"
#include "cn24/net/AdvancedMaxPoolingLayer.h"
#include "cn24/net/InputDownSamplingLayer.h"
#include "cn24/net/LocalResponseNormalizationLayer.h"
#include "cn24/net/UpscaleLayer.h"
#include "cn24/net/LossFunctionLayer.h"
#include "cn24/net/ErrorLayer.h"
#include "cn24/net/DummyErrorLayer.h"
#include "cn24/net/YOLOLossLayer.h"
#include "cn24/net/YOLODetectionLayer.h"
#include "cn24/net/YOLODynamicOutputLayer.h"
#include "cn24/net/StatLayer.h"
#include "cn24/net/BinaryStatLayer.h"
#include "cn24/net/ConfusionMatrixLayer.h"
#include "cn24/net/DetectionStatLayer.h"
#include "cn24/net/SpatialPriorLayer.h"
#include "cn24/net/ConcatenationLayer.h"
#include "cn24/net/GradientAccumulationLayer.h"
#include "cn24/net/SumLayer.h"
#include "cn24/net/Trainer.h"
#include "cn24/net/NetGraph.h"
#include "cn24/net/NetGraphNode.h"
#include "cn24/net/NetStatus.h"
#include "cn24/net/LayerFactory.h"
#include "cn24/net/HMaxActivationFunction.h"
#include "cn24/net/SparsityReLULayer.h"
#include "cn24/net/SparsityLossLayer.h"

#include "cn24/factory/JSONNetGraphFactory.h"
#include "cn24/factory/JSONDatasetFactory.h"
#include "cn24/factory/JSONOptimizerFactory.h"

#endif
