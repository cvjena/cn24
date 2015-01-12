/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file classifyImageLabelMeFacade.cpp
 * \brief Application that uses a pretrained net to segment images.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#include <iostream>
#include <fstream>
#include <cstring>

#include <cn24.h>

int main (int argc, char* argv[]) {
  // TODO delete this program
  unsigned int BATCHSIZE = 1024;
  unsigned int PATCHSIZEX = 32;
  unsigned int PATCHSIZEY = 32;
  unsigned int GLOBAL_STEP = 70;
  double COMPACTNESS = 9.0;

  if (argc < 4) {
    LOGERROR << "USAGE: " << argv[0] << " <net> <param Tensor> <image>";
    LOGEND;
    return -1;
  }

  Conv::System::Init();

  Conv::Factory* factory = Conv::Factory::getNetFactory (argv[1][0], 49932);
  if (factory == nullptr) {
    FATAL ("Unknown net: " << argv[1]);
  }

  PATCHSIZEX = factory->patchsizex();
  PATCHSIZEY = factory->patchsizey();

  std::string infilename (argv[3]);
  auto slashpos = infilename.rfind ('/');
  if (slashpos == std::string::npos)
    slashpos = 0;
  else
    slashpos++;

  std::string outfilename = "usenet/" + infilename.substr (slashpos) + ".data";
  LOGDEBUG << "Writing to " << outfilename;

  Conv::Tensor image_tensor;
  Conv::JPGLoader::LoadFromFile (argv[3], image_tensor);

  LOGDEBUG << "Loaded image: " << image_tensor << ", extracting...";

  Conv::Tensor patch_tensor;
  Conv::Tensor helper_tensor;
  Conv::Segmentation::ExtractPatches (PATCHSIZEX, PATCHSIZEY, patch_tensor, helper_tensor, image_tensor, 0, false);


  unsigned int patches = patch_tensor.samples();
  LOGDEBUG << "Calculating " << patches << " patches.";

  Conv::Tensor data_tensor (BATCHSIZE, PATCHSIZEX,
                            PATCHSIZEY, image_tensor.maps());
  Conv::Tensor helper_data_tensor (BATCHSIZE, 2);

  Conv::Net net;
  Conv::InputLayer input_layer (data_tensor, helper_data_tensor);
  int data_layer_id = net.AddLayer (&input_layer);

  int output_layer_id =
    factory->AddLayers (net, {Conv::Connection (data_layer_id) } , 9);

  Conv::Tensor* net_output_tensor = & (net.buffer (output_layer_id)->data);
  Conv::Tensor output_tensor (1, image_tensor.width() * image_tensor.height(), 3);
  Conv::Tensor saved_output_tensor (image_tensor.width() * image_tensor.height(), net_output_tensor->width());

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

  net.SetDropoutEnabled (false);

  /*
   * Run
   */
  unsigned int fiftieth = 0;
  unsigned int tenth = 0;
  for (unsigned int sample = 0; sample < patches; sample += BATCHSIZE) {


#ifdef BUILD_OPENCL
	data_tensor.MoveToCPU();
	helper_data_tensor.MoveToCPU();
#endif

    for (unsigned int s = 0; (s < BATCHSIZE) && ( (sample + s) < patches); s++) {
      Conv::Tensor::CopySample (patch_tensor, sample + s,
                                data_tensor, s);
      Conv::Tensor::CopySample (helper_tensor, sample + s, helper_data_tensor, s);
    }
    net.FeedForward();

#ifdef BUILD_OPENCL
  net_output_tensor->MoveToCPU();
#endif


    for (unsigned int s = 0; (s < BATCHSIZE) && ( (sample + s) < patches); s++) {
      Conv::Tensor::CopySample (*net_output_tensor, s, saved_output_tensor, sample + s);
    }
    /*
    [colors]
    0: various = 0:0:0
    1: building = 128:0:0
    2: car = 128:0:128
    3: door = 128:128:0
    4: pavement = 128:128:128
    5: road = 128:64:0
    6: sky = 0:128:128
    7: vegetation = 0:128:0
    8: window = 0:0:128
    */

    /*for (unsigned int s = 0; (s < BATCHSIZE) && ( (sample + s) < patches); s++) {
      Conv::duint s_class = net_output_tensor->Maximum(s);
      Conv::datum r = 0, g = 0, b = 0;
      switch(s_class) {
    case 1:
    r = 0.5;
    break;
    case 2:
    r = 0.5;
    b = 0.5;
    break;
    case 3:
    r = 0.5;
    b = 0.5;
    break;
    case 4:
    r = 0.5;
    b = 0.5;
    g = 0.5;
    break;
    case 5:
    r = 0.5;
    g = 0.25;
    break;
    case 6:
    g = 0.5;
    b = 0.5;
    break;
    case 7:
    g = 0.5;
    break;
    case 8:
    b = 0.5;
    break;
      }

      r -= 0.5;
      r *= 2.0;
      g -= 0.5;
      g *= 2.0;
      b -= 0.5;
      b *= 2.0;

      output_tensor[3 * (sample + s)] = r;
      output_tensor[3 * (sample + s) + 1] = g;
      output_tensor[3 * (sample + s) + 2] = b;
    }*/

    //..
    if ( ( (50 * sample) / patches) > fiftieth) {
      fiftieth = (50 * sample) / patches;
      std::cout << "." << std::flush;
    }
    if ( ( (10 * sample) / patches) > tenth) {
      tenth = (10 * sample) / patches;
      std::cout << tenth * 10 << "%" << std::flush;
    }
  }

  for (unsigned int p = 0; p < patches; p ++) {
    Conv::duint s_class = saved_output_tensor.Maximum (p);
    Conv::datum r = 0, g = 0, b = 0;
    switch (s_class) {
    case 1:
      r = 0.5;
      break;
    case 2:
      r = 0.5;
      b = 0.5;
      break;
    case 3:
      r = 0.5;
      b = 0.5;
      break;
    case 4:
      r = 0.5;
      b = 0.5;
      g = 0.5;
      break;
    case 5:
      r = 0.5;
      g = 0.25;
      break;
    case 6:
      g = 0.5;
      b = 0.5;
      break;
    case 7:
      g = 0.5;
      break;
    case 8:
      b = 0.5;
      break;
    }

    r -= 0.5;
    r *= 2.0;
    g -= 0.5;
    g *= 2.0;
    b -= 0.5;
    b *= 2.0;

    output_tensor[3 * p] = r;
    output_tensor[3 * p + 1] = g;
    output_tensor[3 * p + 2] = b;
  }


std::ofstream output_img (outfilename, std::ios::out | std::ios::binary);

/*output_tensor.Reshape(1, image_tensor.width() - (PATCHSIZEX -1),
  image_tensor.height() - (PATCHSIZEY - 1) );*/
output_tensor.Serialize (output_img, true);
output_img.close();

LOGEND;
}
