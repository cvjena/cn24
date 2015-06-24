/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cstdio>

#include "ErrorLayer.h"

#include "ConvolutionLayer.h"
#include "ResizeLayer.h"
#include "MaxPoolingLayer.h"
#include "NonLinearityLayer.h"
#include "UpscaleLayer.h"
#include "SpatialPriorLayer.h"
#include "ConfigParsing.h"

#include "ConfigurableFactory.h"

namespace Conv {



ConfigurableFactory::ConfigurableFactory (std::istream& file, const unsigned int seed, bool is_training_factory) :
  seed_ (seed), file_ (file), method_ (FCN) {
  file_.clear();
  file_.seekg (0, std::ios::beg);

  receptive_field_x_ = 0;
  receptive_field_y_ = 0;

  patch_field_x_ = 0;
  patch_field_y_ = 0;

  // Calculate patch size / receptive field size
  while (! file_.eof()) {
    std::string line;
    std::getline (file_, line);

    std::string method;
    ParseStringParamIfPossible (line, "method", method);

    if (method.compare (0, 5, "patch") == 0) {
      if (is_training_factory) {
        method_ = PATCH;
        LOGINFO << "Setting method to PATCH";
      }
    }

    if (line.compare (0, 1, "?") == 0) {
      line = line.substr (1);

      if (StartsWithIdentifier (line, "convolutional")) {
        unsigned int kx, ky;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        LOGDEBUG << "Adding convolutional layer to receptive field (" << kx << "," << ky << ")";
        receptive_field_x_ += factorx * (kx - 1);
        receptive_field_y_ += factory * (ky - 1);
      }

      if (StartsWithIdentifier (line, "maxpooling")) {
        unsigned int kx, ky;
        LOGDEBUG << "Convolutional layer";
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        LOGDEBUG << "Adding maxpooling layer to receptive field (" << kx << "," << ky << ")";
        factorx *= kx;
        factory *= ky;
      }
    }
  }

  if (method_ == PATCH) {
    receptive_field_x_ += factorx;
    receptive_field_y_ += factory;
  }

  patch_field_x_ = receptive_field_x_ + factorx;
  patch_field_y_ = receptive_field_y_ + factory;
}

Layer* ConfigurableFactory::CreateLossLayer (const unsigned int output_classes) {
  return new ErrorLayer();
}

int ConfigurableFactory::AddLayers (Net& net, Connection data_layer_connection, const unsigned int output_classes) {
  std::mt19937 rand (seed_);
  file_.clear();
  file_.seekg (0, std::ios::beg);
  int last_layer_output = data_layer_connection.output;
  int last_layer_id = data_layer_connection.net;
  int current_receptive_field_x = patch_field_x_;
  int current_receptive_field_y = patch_field_y_;
  datum llr_factor = 1.0;

  if (method_ == FCN) {
    /*Tensor* const net_output = &net.buffer (data_layer_connection.net, data_layer_connection.output)->data;
    datum net_output_size_x = net_output->width();
    datum net_output_size_y = net_output->height();
    llr_factor /= net_output_size_x;
    llr_factor /= net_output_size_y;
#ifdef CN24_EMULATE_PATCH_LEARNING
    llr_factor /= patch_field_x_;
    llr_factor /= patch_field_y_;
#endif
    LOGINFO << "Local learning rate factor is (initially): " << llr_factor;*/
    last_layer_id = net.AddLayer (new ResizeLayer (receptive_field_x_, receptive_field_y_), { data_layer_connection });
    last_layer_output = 0;
  }

  bool first_layer = true;

  while (! file_.eof()) {
    std::string line;
    std::getline (file_, line);

    // Replace number of output neurons
    if (line.find ("(o)") != std::string::npos) {
      char buf[64];
      sprintf (buf, "%d", output_classes);
      line.replace (line.find ("(o)"), 3, buf);
    }

    // Replace fully connected layers
    if (line.find ("fullyconnected") != std::string::npos) {
      line.replace (line.find ("fullyconnected"), 14, "convolutional size=1x1");
      line.replace (line.find ("neurons="), 8, "kernels=");
    }

    if (method_ == FCN) {
      // Remove flatten layers
      if (line.find ("flatten") != std::string::npos) {
        line = "";
      }
    }

    if (line.compare ("?output") == 0) {
      if (output_classes == 1) {
        line = "?tanh";
      } else {
        line = "?sigm";
      }
    }

    if (line.compare (0, 1, "?") == 0) {
      line = line.substr (1);
      LOGDEBUG << "Parsing layer: " << line;

      if (StartsWithIdentifier (line, "convolutional")) {
        unsigned int kx = 1, ky = 1, k = 1;
        datum llr = 1;
        datum dropout_fraction = 0.0;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        ParseCountIfPossible (line, "kernels", k);
        ParseDatumParamIfPossible (line, "dropout", dropout_fraction);
        ParseDatumParamIfPossible (line, "llr", llr);
        LOGDEBUG << "Parsed dropout fraction: " << dropout_fraction;

        ConvolutionLayer* cl = new ConvolutionLayer (kx, ky, k, rand(), dropout_fraction);

        // if (method_ == FCN) {
          LOGDEBUG << "LLR factor: " << llr_factor << ", RFX: " << current_receptive_field_x;
        /*  cl->SetLocalLearningRate (llr * llr_factor * (datum) current_receptive_field_x * (datum) current_receptive_field_y);
#ifdef CN24_EMULATE_PATCH_LEARNING
          current_receptive_field_x -= (kx - 1);
          current_receptive_field_y -= (ky - 1);
#endif
        } else { */
          cl->SetLocalLearningRate (llr * llr_factor);
        /* } */

        if (first_layer)
          cl->SetBackpropagationEnabled (false);

        last_layer_id = net.AddLayer (cl ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;
        first_layer = false;

      }

      if (StartsWithIdentifier (line, "maxpooling")) {
        unsigned int kx = 1, ky = 1;
        ParseKernelSizeIfPossible (line, "size", kx, ky);

        /*if (method_ == FCN) {
#ifdef CN24_EMULATE_PATCH_LEARNING
          current_receptive_field_x /= kx;
          current_receptive_field_y /= ky;
          llr_factor *= (datum) (kx * ky);
#endif
        }*/

        MaxPoolingLayer* mp = new MaxPoolingLayer (kx, ky);
        last_layer_id = net.AddLayer (mp ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;
      }

      if (StartsWithIdentifier (line, "sigm")) {
        SigmoidLayer* l = new SigmoidLayer();
        last_layer_id = net.AddLayer (l ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;
      }

      if (StartsWithIdentifier (line, "relu")) {
        ReLULayer* l = new ReLULayer();
        last_layer_id = net.AddLayer (l ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;
      }

      if (StartsWithIdentifier (line, "tanh")) {
        TanhLayer* l = new TanhLayer();
        last_layer_id = net.AddLayer (l ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;
      }

      if (StartsWithIdentifier (line, "spatialprior")) {
        if (method_ == FCN) {
          SpatialPriorLayer* l = new SpatialPriorLayer();
          last_layer_id = net.AddLayer (l ,
          { Connection (last_layer_id, last_layer_output) });
          last_layer_output = 0;
        }
      }
    }
  }

  if (method_ == FCN && (factorx != 1 || factory != 1)) {
    last_layer_id = net.AddLayer (new UpscaleLayer (factorx, factory),
    { Connection (last_layer_id, last_layer_output) });
    last_layer_output = 0;
    LOGDEBUG << "Added upscaling layer for FCN";
  }

  return last_layer_id;
}

void ConfigurableFactory::InitOptimalSettings() {
  file_.clear();
  file_.seekg (0, std::ios::beg);

  while (!file_.eof()) {
    std::string line;
    std::getline (file_, line);

    ParseDatumIfPossible (line, "l1", optimal_settings_.l1_weight);
    ParseDatumIfPossible (line, "l2", optimal_settings_.l2_weight);
    ParseDatumIfPossible (line, "lr", optimal_settings_.learning_rate);
    ParseDatumIfPossible (line, "gamma", optimal_settings_.gamma);
    ParseDatumIfPossible (line, "momentum", optimal_settings_.momentum);
    ParseDatumIfPossible (line, "exponent", optimal_settings_.exponent);
    ParseUIntIfPossible (line, "iterations", optimal_settings_.iterations);
    ParseUIntIfPossible (line, "sbatchsize", optimal_settings_.sbatchsize);
    ParseUIntIfPossible (line, "pbatchsize", optimal_settings_.pbatchsize);
    
    std::string method;
    ParseStringIfPossible(line, "optimization", method);
    if(method.compare("gradient_descent") == 0) {
      optimal_settings_.optimization_method = GRADIENT_DESCENT;
    } else if(method.compare("quickprop") == 0) {
      optimal_settings_.optimization_method = QUICKPROP;
    }
  }
}


}
