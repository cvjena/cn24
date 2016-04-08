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
#include "HMaxActivationFunction.h"
#include "UpscaleLayer.h"
#include "SpatialPriorLayer.h"
#include "ConcatenationLayer.h"
#include "SumLayer.h"
#include "ConfigParsing.h"
#include "NetGraph.h"
#include "NetGraphNode.h"

#include "ConfigurableFactory.h"

namespace Conv {



ConfigurableFactory::ConfigurableFactory (std::istream& file, const unsigned int seed, bool is_training_factory) :
  method_ (FCN), file_ (file), seed_(seed) {
  bool used_pad = false;
  file_.clear();
  file_.seekg (0, std::ios::beg);

  receptive_field_x_ = 0;
  receptive_field_y_ = 0;

  patch_field_x_ = 0;
  patch_field_y_ = 0;
  
  bool ignore_layers = false;

  // Calculate patch size / receptive field size
  while (! file_.eof()) {
    std::string line;
    std::getline (file_, line);
    if (line.compare (0, 1, "#") == 0) {
        continue;
    }    

    std::string method;
    ParseStringParamIfPossible (line, "method", method);

    if (method.compare (0, 5, "patch") == 0) {
      if (is_training_factory) {
        method_ = PATCH;
        LOGDEBUG << "Setting method to PATCH";
      }
    }
    
    if (line.compare (0, 6, "manual") == 0) {
      unsigned int rfx = 0, rfy = 0, fx = 0, fy = 0;
      ParseCountIfPossible(line, "rfx", rfx);
      ParseCountIfPossible(line, "rfy", rfy);
      ParseCountIfPossible(line, "factorx", fx);
      ParseCountIfPossible(line, "factory", fy);
      if(rfx > 0 && rfy > 0) {
        LOGDEBUG << "Using manual receptive field method";
        receptive_field_x_ = rfx;
        receptive_field_y_ = rfy;
        factorx = fx;
        factory = fy;
        ignore_layers = true;
      }
    }

    if (line.compare (0, 1, "?") == 0 && !ignore_layers) {
      line = line.substr (1);

      if (StartsWithIdentifier (line, "convolutional")) {
        unsigned int kx, ky, stridex = 1, stridey = 1, padx = 0, pady = 0;
        
        ParseKernelSizeIfPossible (line, "stride", stridex, stridey);
        ParseKernelSizeIfPossible (line, "pad", padx, pady);
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        
        if(padx > 0 || pady > 0) {
          used_pad = true;
        }
        
        LOGDEBUG << "Adding convolutional layer to receptive field (" << kx << "," << ky << "s" << stridex << "," << stridey << "p" << padx << "," << pady << ")";
        receptive_field_x_ += factorx * ((int)kx - 1 - (int)padx - (int)padx);
        receptive_field_y_ += factory * ((int)ky - 1 - (int)pady - (int)pady);
        factorx *= stridex;
        factory *= stridey;
      }
      
      if (StartsWithIdentifier (line, "maxpooling")) {
        unsigned int kx, ky;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        LOGDEBUG << "Adding maxpooling layer to receptive field (" << kx << "," << ky << ")";
        factorx *= kx;
        factory *= ky;
      }

      if (StartsWithIdentifier (line, "downsampling")) {
        unsigned int kx, ky;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        LOGDEBUG << "Adding down-sampling layer to receptive field (" << kx << "," << ky << ")";
        factorx *= kx;
        factory *= ky;
      }
      
      if (StartsWithIdentifier (line, "amaxpooling")) {
        unsigned int kx, ky, sx, sy;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        sx = kx; sy = ky;
        ParseKernelSizeIfPossible (line, "stride", sx, sy);
        LOGDEBUG << "Adding adv. maxpooling layer to receptive field (" << kx << "," << ky << "s" << sx << "," << sy << ")";
        receptive_field_x_ += factorx * ((int)kx - 1);
        receptive_field_y_ += factory * ((int)ky - 1);
        factorx *= sx;
        factory *= sy;
      }
    }
  }
  
  LOGDEBUG << "To achieve this receptive field size manually, start net config with manual rfx=" << receptive_field_x_
    << " rfy=" << receptive_field_y_ << " factorx=" << factorx << " factory=" << factory;

  if (method_ == PATCH) {
    receptive_field_x_ += factorx;
    receptive_field_y_ += factory;
    if(used_pad) {
      LOGWARN << "Using padding in hybrid mode may have undesired consequences!";
    }
  LOGDEBUG << "Patch size would be " << receptive_field_x_ << "x" << receptive_field_y_;
  }

  patch_field_x_ = receptive_field_x_ + factorx;
  patch_field_y_ = receptive_field_y_ + factory;
}

Layer* ConfigurableFactory::CreateLossLayer (const unsigned int output_classes, const datum loss_weight) {
  UNREFERENCED_PARAMETER(output_classes);
  return new ErrorLayer(loss_weight);
}

bool ConfigurableFactory::AddLayers(NetGraph& net, NetGraphConnection data_layer_connection, const unsigned int output_classes, bool add_loss_layer) {
  std::mt19937 rand (seed_);
  file_.clear();
  file_.seekg (0, std::ios::beg);

	NetGraphConnection last_connection = data_layer_connection;

  NetGraphConnection stack_a[64];
  NetGraphConnection stack_b[64];
  int stack_a_pos = -1;
  int stack_b_pos = -1;

	bool already_upscaled = (factorx == 1) && (factory == 1);

  if (method_ == FCN && (receptive_field_x_ > 0) && (receptive_field_y_ > 0)) {
		ResizeLayer* rl = new ResizeLayer(receptive_field_x_, receptive_field_y_);
		NetGraphNode* node = new NetGraphNode(rl, last_connection);
		net.AddNode(node);

		last_connection.node = node;
		last_connection.buffer = 0;
		last_connection.backprop = false;
  }

  while (! file_.eof()) {
    std::string line;
    std::getline (file_, line);
    if (line.compare (0, 1, "#") == 0) {
        continue;
    }    

    datum loss_weight = 1.0;

    /*
     * PREPROCESSING
     */
    
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
    
		bool is_output = false;
    if (line.compare (0, 7, "?output") == 0) {
			ParseDatumParamIfPossible(line, "weight", loss_weight);
      if (output_classes == 1) {
        line = "?tanh";
      } else {
        line = "?sigm";
      }
			is_output = true;
    }
    
    /*
     * STACK OPERATIONS
     */
    if (line.compare(0, 5, "pusha") == 0) {
      stack_a[++stack_a_pos] = last_connection;
    }

    if (line.compare(0, 5, "pushb") == 0) {
      stack_b[++stack_b_pos] = last_connection;
    }
    
    if (line.compare(0, 4, "popa") == 0) {
      last_connection = stack_a[stack_a_pos--];
    }
    
    if (line.compare(0, 4, "popb") == 0) {
      last_connection = stack_b[stack_b_pos--];
    }
    
    /*
     * PARSING
     */
    if (line.compare (0, 1, "?") == 0) {
      line = line.substr (1);
      LOGDEBUG << "Parsing layer: " << line;

      if (StartsWithIdentifier (line, "convolutional")) {
        unsigned int kx = 1, ky = 1, k = 1, stridex = 1, stridey = 1, padx = 0, pady = 0, group = 1;
        datum llr = 1;
        datum dropout_fraction = 0.0;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        ParseKernelSizeIfPossible (line, "stride", stridex, stridey);
        ParseKernelSizeIfPossible (line, "pad", padx, pady);
        ParseCountIfPossible (line, "kernels", k);
        ParseCountIfPossible (line, "group", group);
        ParseDatumParamIfPossible (line, "dropout", dropout_fraction);
        ParseDatumParamIfPossible (line, "llr", llr);
        LOGDEBUG << "Parsed dropout fraction: " << dropout_fraction;

        ConvolutionLayer* cl = new ConvolutionLayer (kx, ky, k, stridex, stridey, padx, pady, group, rand(), dropout_fraction);
				cl->SetLocalLearningRate (llr);

				NetGraphNode* node = new NetGraphNode(cl, last_connection);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
      }
      
      if (StartsWithIdentifier (line, "lrn")) {
        LocalResponseNormalizationLayer::NormalizationMethod normalization_method;
        std::string method_string;
        ParseStringParamIfPossible(line, "method", method_string);
        if(method_string.compare(0,6,"across") == 0)
          normalization_method = LocalResponseNormalizationLayer::ACROSS_CHANNELS;
        else if(method_string.compare(0,6,"within") == 0)
          normalization_method = LocalResponseNormalizationLayer::WITHIN_CHANNELS;
        else {
          FATAL("Cannot initialize LRN layer: method missing!");
        }
        
        unsigned int size = 1;
        datum alpha = 1, beta = 1;
        ParseCountIfPossible(line, "size", size);
        ParseDatumParamIfPossible(line, "alpha", alpha);
        ParseDatumParamIfPossible(line, "beta", beta);
        
        LocalResponseNormalizationLayer* lrn = new LocalResponseNormalizationLayer(size, alpha, beta, normalization_method);
        NetGraphNode* node = new NetGraphNode(lrn, last_connection);
        net.AddNode(node);
        last_connection.buffer = 0;
        last_connection.node = node;
        last_connection.backprop = true;
      }

      if (StartsWithIdentifier (line, "maxpooling")) {
        unsigned int kx = 1, ky = 1;
        ParseKernelSizeIfPossible (line, "size", kx, ky);

        MaxPoolingLayer* mp = new MaxPoolingLayer (kx, ky);

				NetGraphNode* node = new NetGraphNode(mp, last_connection);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
      }
      
      if (StartsWithIdentifier (line, "downsampling")) {
        unsigned int kx = 1, ky = 1;
        ParseKernelSizeIfPossible (line, "size", kx, ky);

        InputDownSamplingLayer* mp = new InputDownSamplingLayer (kx, ky);

        NetGraphNode* node = new NetGraphNode(mp, last_connection);
        net.AddNode(node);
        last_connection.buffer = 0;
        last_connection.node = node;
        last_connection.backprop = false;
      }
      
      if (StartsWithIdentifier (line, "amaxpooling")) {
        unsigned int kx = 1, ky = 1, sx, sy;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        sx = kx; sy = ky;
        ParseKernelSizeIfPossible (line, "stride", sx, sy);

        AdvancedMaxPoolingLayer* mp = new AdvancedMaxPoolingLayer (kx, ky, sx, sy);

        NetGraphNode* node = new NetGraphNode(mp, last_connection);
        net.AddNode(node);
        last_connection.buffer = 0;
        last_connection.node = node;
        last_connection.backprop = true;
      }

      if (StartsWithIdentifier (line, "sigm")) {
        SigmoidLayer* l = new SigmoidLayer();
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				node->is_output = is_output && (method_ == PATCH || already_upscaled);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
      }

      if (StartsWithIdentifier (line, "hmax")) {
        datum mu = 1; datum loss_weight_ = 0;
        ParseDatumParamIfPossible(line, "mu", mu);
        ParseDatumParamIfPossible(line, "weight", loss_weight_);
        
        HMaxActivationFunction* l = new HMaxActivationFunction(mu, loss_weight_);
        l->SetLocalLearningRate(1.0);
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				node->is_output = is_output && (method_ == PATCH || already_upscaled);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
      }
      
      if (StartsWithIdentifier (line, "relu")) {
        ReLULayer* l = new ReLULayer();
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				node->is_output = is_output && (method_ == PATCH || already_upscaled);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
      }

      if (StartsWithIdentifier (line, "tanh")) {
        TanhLayer* l = new TanhLayer();
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				node->is_output = is_output && (method_ == PATCH || already_upscaled);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
      }

      if (StartsWithIdentifier (line, "spatialprior")) {
				if (!already_upscaled && method_ == FCN) {
					UpscaleLayer* l = new UpscaleLayer(factorx, factory);
					NetGraphNode* node = new NetGraphNode(l, last_connection);
					net.AddNode(node);
					last_connection.buffer = 0;
					last_connection.node = node;
					last_connection.backprop = true;
					
					LOGDEBUG << "Added upscaling layer for FCN (spatial prior)";
					already_upscaled = true;
				}
				ConcatenationLayer* l = new ConcatenationLayer();
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				node->input_connections.push_back(NetGraphConnection(data_layer_connection.node, 2, false));
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
      }

			if (StartsWithIdentifier(line, "concat")){
				std::string stack_name;
				NetGraphConnection* stack_ptr;
				int stack_pos;
				ParseStringParamIfPossible(line, "stack", stack_name);
				if (stack_name.compare(0, 1, "b") == 0){
					stack_ptr = stack_b;
					stack_pos = stack_b_pos;
				}
				else {
					stack_ptr = stack_a;
					stack_pos = stack_a_pos;
				}
				ConcatenationLayer* l = new ConcatenationLayer();
				NetGraphNode* node = new NetGraphNode(l);
				for (int p = stack_pos; p >= 0; p--) {
					node->input_connections.push_back(stack_ptr[p]);
				}
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
			}
			
      if (StartsWithIdentifier(line, "sum")){
        std::string stack_name;
        NetGraphConnection* stack_ptr;
        int stack_pos;
        ParseStringParamIfPossible(line, "stack", stack_name);
        if (stack_name.compare(0, 1, "b") == 0){
          stack_ptr = stack_b;
          stack_pos = stack_b_pos;
        }
        else {
          stack_ptr = stack_a;
          stack_pos = stack_a_pos;
        }
        SumLayer* l = new SumLayer();
        NetGraphNode* node = new NetGraphNode(l);
        for (int p = stack_pos; p >= 0; p--) {
          node->input_connections.push_back(stack_ptr[p]);
        }
        net.AddNode(node);
        last_connection.buffer = 0;
        last_connection.node = node;
        last_connection.backprop = true;
      }
      
      if (StartsWithIdentifier(line, "upscale")){
        unsigned int ufx = 1, ufy = 1;
        unsigned int o = 0;
        ParseKernelSizeIfPossible(line, "factor", ufx, ufy);
        ParseCountIfPossible(line, "is_output", o);
        UpscaleLayer* l = new UpscaleLayer(ufx, ufy);
        NetGraphNode* node = new NetGraphNode(l, last_connection);
        node->is_output = (o == 1);
        net.AddNode(node);
        last_connection.buffer = 0;
        last_connection.node = node;
        last_connection.backprop = true; 
        is_output = (o == 1);
        already_upscaled = true;
      }

			if (is_output && !already_upscaled && method_ == FCN && (factorx != 1 || factory != 1)) {
				UpscaleLayer* l = new UpscaleLayer(factorx, factory);
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				node->is_output = true;
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
				last_connection.backprop = true;
				
				LOGDEBUG << "Added upscaling layer for FCN";
			}

			if (is_output && add_loss_layer) {
				NetGraphNode* output_node = last_connection.node;

				// Collect inputs
				NetGraphConnection label_connection = data_layer_connection;
				label_connection.buffer = 1;

				NetGraphConnection weight_connection = data_layer_connection;
				weight_connection.buffer = 3;

				Layer* loss_layer = CreateLossLayer(output_classes, loss_weight);
				NetGraphNode* node = new NetGraphNode(loss_layer);
				node->input_connections.push_back(NetGraphConnection(output_node));
				node->input_connections.push_back(label_connection);
				node->input_connections.push_back(weight_connection);

				net.AddNode(node);
			}
    }

  }

	return net.IsComplete();
}

void ConfigurableFactory::InitOptimalSettings() {
  file_.clear();
  file_.seekg (0, std::ios::beg);

  while (!file_.eof()) {
    std::string line;
    std::getline (file_, line);
    if (line.compare (0, 1, "#") == 0) {
        continue;
    }    

    ParseDatumIfPossible (line, "l1", optimal_settings_.l1_weight);
    ParseDatumIfPossible (line, "l2", optimal_settings_.l2_weight);
    ParseDatumIfPossible (line, "lr", optimal_settings_.learning_rate);
    ParseDatumIfPossible (line, "gamma", optimal_settings_.gamma);
    ParseDatumIfPossible (line, "momentum", optimal_settings_.momentum);
    ParseDatumIfPossible (line, "exponent", optimal_settings_.exponent);
    ParseDatumIfPossible (line, "eta", optimal_settings_.eta);
    ParseDatumIfPossible (line, "mu", optimal_settings_.mu);
    ParseUIntIfPossible (line, "iterations", optimal_settings_.iterations);
    ParseUIntIfPossible (line, "sbatchsize", optimal_settings_.sbatchsize);
    ParseUIntIfPossible (line, "pbatchsize", optimal_settings_.pbatchsize);
    
    std::string method;
    ParseStringIfPossible(line, "optimization", method);
    if(method.compare(0, 16, "gradient_descent") == 0) {
      optimal_settings_.optimization_method = GRADIENT_DESCENT;
    } else if(method.compare(0, 9, "quickprop") == 0) {
      optimal_settings_.optimization_method = QUICKPROP;
    }
  }
}

void ConfigurableFactory::WriteNode(std::ostream& graph_output, Layer* layer, int source_id, int source_port, int node_id, int outputs) {
	graph_output << "node" << node_id << " [shape=record, label=\"" <<
		"{" << layer->GetLayerDescription();
	if (outputs > 1) {
		graph_output << " | {";
		for (int i = 0; i < outputs; i++) {
			if (i == 0)
				graph_output << " | ";
			graph_output << "<o" << i << "> Output " << i;
		}
		graph_output << "}";
	}
	else if (outputs == 1) {
		graph_output << " | <o0> Output";
	} 
	graph_output << " }\"];\n";
	graph_output << "node" << source_id << ":o" << source_port
		<< " -> node" << node_id << ";\n";
}

}
