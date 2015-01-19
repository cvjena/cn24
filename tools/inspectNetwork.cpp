/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file inspectNetwork.cpp
 * \brief Application that uses a pretrained net to segment images.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <cstring>

#include <cn24.h>

int main (int argc, char* argv[]) {
  unsigned int BATCHSIZE = 96;
  unsigned int PATCHSIZEX = 32;
  unsigned int PATCHSIZEY = 32;

  if (argc < 10) {
    LOGERROR << "USAGE: " << argv[0] << " <net> <param Tensor> <image> <lid> <ox> <oy> <omap> <damp> <sr>";
    LOGEND;
    return -1;
  }

  unsigned int layer_id = atoi(argv[4]);
  unsigned int output_x = atoi(argv[5]);
  unsigned int output_y = atoi(argv[6]);
  unsigned int output_map = atoi(argv[7]);
  Conv::datum damp = (Conv::datum)atoi(argv[8]);
  bool sr = (argv[9][0] == 'r');

  LOGDEBUG << "Using layer id " << layer_id << ", map " << output_map << "(" << output_x
           << "," << output_y << ")";


  // Load image
  std::ifstream image_stream (argv[3], std::ios::binary | std::ios::in);
  if (!image_stream.good()) {
    FATAL ("Cannot open " << argv[3] << "!");
    return -1;
  }
  
  Conv::Factory* factory = Conv::Factory::getNetFactory (argv[1][0], 49932);
  if (factory == nullptr) {
    FATAL ("Unknown net: " << argv[1]);
  }
  
  PATCHSIZEX = factory->patchsizex();
  PATCHSIZEY = factory->patchsizey();
  
  std::string infilename(argv[3]);
  auto slashpos = infilename.rfind('/');
  if(slashpos == std::string::npos)
    slashpos = 0;
  else
    slashpos++;
  
  std::string outfilename = "usenet/" + std::string(argv[7]) + infilename.substr(slashpos) + ".data";
  LOGDEBUG << "Writing to " << outfilename;

  Conv::Tensor image_tensor;
  Conv::PNGLoader::LoadFromStream (image_stream, image_tensor);

  LOGDEBUG << "Loaded image: " << image_tensor << ", extracting...";

  Conv::Tensor patch_tensor;
  Conv::Tensor helper_tensor;
  Conv::Segmentation::ExtractPatches (PATCHSIZEX, PATCHSIZEY, patch_tensor, helper_tensor, image_tensor, 0, true);
  

  unsigned int patches = patch_tensor.samples();
  LOGDEBUG << "Calculating " << patches << " patches.";

  Conv::Tensor data_tensor (BATCHSIZE, PATCHSIZEX,
                            PATCHSIZEY, image_tensor.maps());
  Conv::Tensor helper_data_tensor (BATCHSIZE, 2);

  Conv::Net net;
  Conv::InputLayer input_layer (data_tensor, helper_data_tensor);
  int data_layer_id = net.AddLayer (&input_layer);

  // int output_layer_id =
    factory->AddLayers (net, {Conv::Connection (data_layer_id) });

  Conv::Tensor* net_output_tensor = & (net.buffer (layer_id)->data);
  Conv::Tensor output_tensor(1, image_tensor.width() * image_tensor.height());

  /*
   * Load net params
   */
  std::ifstream param_stream (argv[2], std::ios::in | std::ios::binary);
  if (!param_stream.good()) {
    FATAL ("Cannot open " << argv[2] << "!");
    return -1;
  }

  net.DeserializeParameters (param_stream);
  
  std::cout << "\n" << std::flush;

  net.SetDropoutEnabled(false);

  /*
   * Run
   */
  unsigned int fiftieth = 0;
  unsigned int tenth = 0;
  for (unsigned int sample = 0; sample < patches; sample += BATCHSIZE) {
    for (unsigned int s = 0; (s < BATCHSIZE) && ( (sample + s) < patches); s++) {
      Conv::Tensor::CopySample (patch_tensor, sample + s,
                                data_tensor, s);
      Conv::Tensor::CopySample (helper_tensor, sample + s, helper_data_tensor, s);
    }
    net.FeedForward(layer_id);
    
    // Copy
    //const Conv::datum* src = net_output_tensor->data_ptr_const();
    //Conv::datum* tgt = output_tensor.data_ptr(sample);
    
    std::size_t samples_in = (((sample + BATCHSIZE) > patches) ? (patches - sample) : BATCHSIZE);
    
    for(unsigned int s = 0; s < samples_in; s++) {
      const Conv::datum* src = net_output_tensor->data_ptr_const(output_x, output_y, output_map, s);
      Conv::datum* tgt = output_tensor.data_ptr(sample + s);
      *tgt = (*src / damp) - (sr ? 1 : 0);
    }
    
    //..
    if(((50*sample)/patches) > fiftieth) {
      fiftieth = (50*sample)/patches;
      std::cout << "." << std::flush;
    }
    if(((10*sample)/patches) > tenth) {
      tenth = (10*sample)/patches;
      std::cout << tenth * 10 << "%" << std::flush;
    }
  }
  std::ofstream output_img(outfilename, std::ios::out | std::ios::binary);
  
  /*output_tensor.Reshape(1, image_tensor.width() - (PATCHSIZEX -1),
    image_tensor.height() - (PATCHSIZEY - 1) );*/
  output_tensor.Serialize(output_img, true);
  output_img.close();
}
