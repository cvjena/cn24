/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file visualizeTensor.cpp
 * \brief Tool to view Tensor contents
 * 
 * \author Clemens-Alexander Brust (ikosa.de@gmail.com)
 */

#include <iostream>
#include <fstream>
#include <string>

#include <cn24.h>

bool parseCommand(Conv::Tensor& tensor, std::string filename);

int main(int argc, char* argv[]) {
  if(argc != 3) {
    LOGERROR << "USAGE: " << argv[0] << " <tensor file> <tensor id in file>";
    LOGEND;
    return -1;
  }
  
  Conv::System::Init();
  
  std::string s_tid = argv[2];
  unsigned int tid = atoi(s_tid.c_str());
  
  LOGINFO << "Reading " << argv[1] << " (id: " << tid << ")";
  
  std::ifstream file(std::string(argv[1]), std::ios::in | std::ios::binary);
  
  Conv::Tensor tensor;
  for(unsigned int t = 0; t < (tid + 1); t++) {
    tensor.Deserialize(file);
  }
  
  file.close();
  
  LOGINFO << "Tensor: " << tensor;
  LOGEND;
  
  bool result;
  do {
    std::cout << "VIS> ";
    result = parseCommand(tensor, argv[1]);
  } while(result);
  
  LOGEND;
  return 0;
}

bool parseCommand (Conv::Tensor& tensor, std::string filename) {
  std::string entry;
  std::cin >> entry;
  
  if(entry.compare("q") == 0)
    return false;
  
  else if(entry.compare("size") == 0)
    std::cout << tensor << std::endl;
  
  else if(entry.compare("transpose") == 0)
    tensor.Transpose();
  
  else if(entry.compare("reshape") == 0) {
    unsigned int w, h, m, s;
    std::cin >> w >> h >> m >> s;
    std::cout << "Reshaping... ";
    bool result = tensor.Reshape(s, w, h, m);
    std::cout << result << std::endl;
  }
  
  else if(entry.compare("extractpatches") == 0) {
    unsigned int psx, psy, s;
    std::cin >> psx >> psy >> s;
    std::cout << "Extracting...\n";
    Conv::Tensor* target = new Conv::Tensor();
    Conv::Tensor* helper = new Conv::Tensor();
    Conv::Segmentation::ExtractPatches(psx, psy, *target, *helper, tensor, s);
    tensor.Shadow(*target);
  }
  
  else if(entry.compare("write") == 0) {
    std::ofstream o(filename, std::ios::out | std::ios::binary);
    tensor.Serialize(o);
    o.close();
  }
  
  else if(entry.compare("bin") == 0) {
    std::ofstream o("binoutput.data", std::ios::out | std::ios::binary);
    if(!o.good()) {
      std::cout << "Did not work." << std::endl;
      return true;
    }
    tensor.Serialize(o, true);
    o.close();
  }
  
  else if (entry.compare("show") == 0) {
    unsigned int s, m;
    std::cin >> s >> m;
    Conv::System::viewer->show(&tensor, "visualizeTensor", false, m, s);
  }
  
  else {
    std::cout << "Command not recognized: " << entry << std::endl;
  }
  return true;
}
