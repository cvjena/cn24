/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <cn24.h>

#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
  Conv::System::Init();
  
  if(argc != 3) {
    LOGERROR << "USAGE: " << argv[0] << " <input (uncompressed) tensor stream> <output (compressed) tensor stream>";
  }
  
  std::string input_file_name(argv[1]);
  std::string output_file_name(argv[2]);
  
  std::ifstream input_tensor_stream(input_file_name, std::ios::in | std::ios::binary);
  std::ofstream output_tensor_stream(output_file_name, std::ios::out | std::ios::binary);
  
  if(!input_tensor_stream.good())
    FATAL("Cannot open " << input_file_name);
  
  if(!output_tensor_stream.good())
    FATAL("Cannot open " << output_file_name);
  
  long uncompressed_total = 0;
  long compressed_total = 0;
  
  Conv::Tensor tensor;
  
  uint64_t magic = CN24_CTS_MAGIC;
  output_tensor_stream.write((char*)&magic, sizeof(uint64_t)/sizeof(char));
  
  while(!input_tensor_stream.eof()) {
    tensor.Deserialize(input_tensor_stream);
    
    LOGDEBUG << "Input tensor: " << tensor;
    
    unsigned int original_size = tensor.elements() * sizeof(Conv::datum)/sizeof(char);
    LOGDEBUG << "Size: " << original_size;
    
    Conv::CompressedTensor ctensor;
    ctensor.Compress(tensor);
    
    ctensor.Serialize(output_tensor_stream);
    
    LOGDEBUG << "RLE Size: " << ctensor.compressed_length();
    
    ctensor.Decompress(tensor);
    unsigned int bytes_out = tensor.elements() * sizeof(Conv::datum)/sizeof(char);
    
    if(bytes_out != original_size) {
      FATAL("Size mismatch! Expected: " << (tensor.elements() * sizeof(Conv::datum)/sizeof(char)) << ", actual: " << bytes_out);
    }
    
    LOGINFO << "Ratio: " << 100.0 * (double)ctensor.compressed_length() / (double)(tensor.elements() * sizeof(Conv::datum)/sizeof(char)) << "%" << std::flush;
    compressed_total += ctensor.compressed_length();
    uncompressed_total += tensor.elements() * sizeof(Conv::datum)/sizeof(char);
    
    input_tensor_stream.peek();
  }
  LOGINFO << "Overall ratio: " << 100.0 * (double)compressed_total / (double)uncompressed_total << "%";
  LOGINFO << "Uncompressed: " << uncompressed_total;
  LOGINFO << "Compressed  : " << compressed_total;
  LOGEND;
}