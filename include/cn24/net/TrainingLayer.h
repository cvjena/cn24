/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file TrainingLayer.h
 * @class TrainingLayer
 * @brief Layer that supports switching between training and testing.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CONV_TRAININGLAYER_H
#define CONV_TRAININGLAYER_H

#include "../util/JSONParsing.h"

namespace Conv {
  
class TrainingLayer {
public:
  /**
   * @brief Set testing or training mode.
   *
   * @param mode True sets testing mode, false sets training mode
   */
  virtual void SetTestingMode(bool testing) = 0;
  
  /**
   * @brief Gets number of samples in training set.
   */
  virtual unsigned int GetSamplesInTrainingSet() = 0;
  
  /**
   * @brief Gets number of samples in testing set.
   */
  virtual unsigned int GetSamplesInTestingSet() = 0;
  
  /**
   * @brief Gets the size of a batch.
   */
  virtual unsigned int GetBatchSize() = 0;

  /**
   * @brief Gets the probability for loss sampling
   */
  virtual datum GetLossSamplingProbability() = 0;
  
  /**
   * @brief Gets the width of the label image (expected output)
   */
  virtual unsigned int GetLabelWidth() = 0;
  
  /**
   * @brief Gets the height of the label image (expected output)
   */
  virtual unsigned int GetLabelHeight() = 0;

  /**
   * @brief Selects a batch of random samples and loads it into an output buffer
   */
  virtual void SelectAndLoadSamples() = 0;

  /**
   * @brief Loads a detection sample from the specified JSON to the specified index
   */
  virtual bool ForceLoadDetection(JSON& sample, unsigned int index) { LOGERROR << "Not implemented!"; return false; }

  /**
   * @brief Sets all the weights in the helper Tensor to zero
   */
  virtual void ForceWeightsZero() { LOGERROR << "Not implements!"; }
};

}

#endif
