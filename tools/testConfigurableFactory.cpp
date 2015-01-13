/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <cn24.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <ctime>
#include <cstring>

#include <cn24.h>

int main (int argc, char* argv[]) {
  if (argc < 2) {
    LOGERROR << "USAGE: " << argv[0] << " <net configuration file>";
    LOGEND;
    return -1;
  }
  
  Conv::System::Init();
  
  std::ifstream file(argv[1],std::ios::in);
  Conv::ConfigurableFactory factory(file, Conv::PATCH);
  
  factory.InitOptimalSettings();
  
  LOGDEBUG << "Factory receptive field is " << factory.patchsizex() << "x" << factory.patchsizey();
  
  Conv::Tensor data_tensor(1,30,30,3);
  Conv::Tensor label_tensor(1,30,30,5);
  Conv::Tensor weight_tensor(1,30,30,1);
  Conv::Tensor helper_tensor(1,30,30,2);
  
  Conv::InputLayer input_layer(data_tensor, label_tensor, helper_tensor, weight_tensor);
  Conv::Net net;
  
  int data_layer_id = net.AddLayer(&input_layer);
  int output_layer_id = factory.AddLayers(net, Conv::Connection(data_layer_id), 5);
  
  LOGEND;
 
}