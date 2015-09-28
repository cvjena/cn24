/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <cn24.h>

#include <iostream>
#include <fstream>

struct CompressedTensor {
public:
  unsigned int compressed_length;
  void* compressed_data;
};

const unsigned char rl_marker = 'X';
const unsigned char rl_doublemarker = 'X';
const unsigned char rl_rle = 'Y';
const unsigned int rl_bytes = 1;
const unsigned int chars_per_datum = sizeof(Conv::datum)/sizeof(char);
const unsigned int rl_max = (unsigned int)((1L << (8L * (unsigned long)rl_bytes)) - 3L);
const unsigned int rl_min = 1 + (5 + rl_bytes) / chars_per_datum;

void Compress(const Conv::Tensor& tensor, CompressedTensor* compressed);
unsigned int Decompress(Conv::Tensor& tensor, CompressedTensor* compressed);

int main(int argc, char** argv) {
  Conv::System::Init();
  
  if(argc != 2) {
    FATAL("Needs exactly two arguments!");
  }
  
  LOGINFO << "rl_bytes: " << rl_bytes;
  LOGINFO << "rl_max  : " << rl_max;
  LOGINFO << "rl_min  : " << rl_min;
  
  CompressedTensor compressed;
  std::string input_file_name(argv[1]);
  //std::string output_file_name(argv[2]);
  
  std::ifstream input_tensor_stream(input_file_name);
  //std::ofstream output_tensor_stream(output_file_name);
  if(!input_tensor_stream.good())
    FATAL("Cannot open " << input_file_name);
  
  long uncompressed_total = 0;
  long compressed_total = 0;
  
  Conv::Tensor tensor;
  while(!input_tensor_stream.eof()) {
    tensor.Deserialize(input_tensor_stream);
    
    LOGDEBUG << "Input tensor: " << tensor;
    
    LOGDEBUG << "Size: " << tensor.elements() * sizeof(Conv::datum)/sizeof(char);
    
    Compress(tensor, &compressed);
    LOGDEBUG << "RLE Size: " << compressed.compressed_length;
    unsigned int bytes_out = Decompress(tensor, &compressed);
    
    //tensor.Serialize(output_tensor_stream,false);
    
    free(compressed.compressed_data);
    
    if(bytes_out != (tensor.elements() * sizeof(Conv::datum)/sizeof(char))) {
      FATAL("Size mismatch! Expected: " << (tensor.elements() * sizeof(Conv::datum)/sizeof(char)) << ", actual: " << bytes_out);
    }
    
    LOGINFO << "Ratio: " << 100.0 * (double)compressed.compressed_length / (double)(tensor.elements() * sizeof(Conv::datum)/sizeof(char)) << "%" << std::flush;
    compressed_total += compressed.compressed_length;
    uncompressed_total += tensor.elements() * sizeof(Conv::datum)/sizeof(char);
    
    input_tensor_stream.peek();
  }
  LOGINFO << "Overall ratio: " << 100.0 * (double)compressed_total / (double)uncompressed_total << "%";
  LOGINFO << "Uncompressed: " << uncompressed_total;
  LOGINFO << "Compressed  : " << compressed_total;
  LOGEND;
}




void Compress(const Conv::Tensor& tensor, CompressedTensor* compressed) {
  
  unsigned int bytes_out = 0;
  
  Conv::datum last_symbol = 0;
  unsigned int running_length = 0;
  
  unsigned char* output_ptr = new unsigned char[2 * tensor.elements() * chars_per_datum];
  compressed->compressed_data = (void*)output_ptr;
  for(unsigned int pos = 0; pos <= tensor.elements(); pos++) {
    Conv::datum current_symbol;
    if(pos < tensor.elements()) {
      current_symbol = tensor.data_ptr_const()[pos];
      if(current_symbol == last_symbol) {
        // Increase running length
        running_length++;
      }
    } else {
      // Force emission of last symbol
    }
    
    
    if(
    // EOF reached
    (pos == (tensor.elements())) ||
    // Different symbol
    (current_symbol != last_symbol) ||
    // Maxmimum run length reached
    (running_length == rl_max)) {
        
      // Emit...
      if(running_length > 0 && running_length < rl_min) {
        // Emit single symbol(s)
        for(unsigned int r = 0; r < running_length; r++) {
          for(unsigned int b = 0; b < chars_per_datum; b++) {
            char char_to_emit = ((char*)&last_symbol)[b];
            if(char_to_emit == rl_marker) {
              // Emit escaped
              *output_ptr = rl_marker;
              output_ptr++; bytes_out++;
              *output_ptr = rl_doublemarker;
              output_ptr++; bytes_out++;
            } else {
              // Emit directly
              *output_ptr = char_to_emit;
              output_ptr++; bytes_out++;
            }
          }
        }
      } else if(running_length >= rl_min) {
        // Emit encoded
        *output_ptr = rl_marker;
        output_ptr++; bytes_out++;
        *output_ptr = rl_rle;
        output_ptr++; bytes_out++;
        
        // Running length output
        for(unsigned int b = 0; b < rl_bytes; b++) {
          *output_ptr = (running_length >> ((rl_bytes - (b+1)) * 8)) & 0xFF;
          output_ptr++; bytes_out++;
        }
        
        for(unsigned int b = 0; b < chars_per_datum; b++) {
          unsigned char char_to_emit = ((char*)&last_symbol)[b];
          *output_ptr = char_to_emit;
          output_ptr++; bytes_out++;
        }
      }
        
      // ...and reset
      if(running_length == rl_max)
        running_length = 0;
      else
        running_length = 1;
    }
      
    last_symbol = current_symbol;
  }
  LOGDEBUG << "Bytes out: " << bytes_out;
  compressed->compressed_length = bytes_out;
  
}

unsigned int Decompress(Conv::Tensor& tensor, CompressedTensor* compressed) {
  unsigned int bytes_out = 0;
  unsigned char* output_ptr = (unsigned char*)tensor.data_ptr();
  const unsigned char* input_ptr = (const unsigned char*)compressed->compressed_data;
  
  for(unsigned int pos = 0; pos < compressed->compressed_length; pos++) {
    unsigned char current_symbol = input_ptr[pos];
    if(current_symbol == rl_marker) {
      pos++; current_symbol = input_ptr[pos];
      if(current_symbol == rl_doublemarker) {
        // Emit single marker
        *output_ptr = rl_marker;
        output_ptr++; bytes_out++;
      } else if(current_symbol == rl_rle) {
        unsigned int running_length = 0;
        
        // Running length input
        for(unsigned int b = 0; b < rl_bytes; b++) {
          pos++; current_symbol = input_ptr[pos];
          running_length += current_symbol;
          if((b+1) != rl_bytes)
            running_length <<= 8;
        }
        
        for(unsigned int r = 0; r < running_length; r++) {
          for(unsigned int b = 0; b < chars_per_datum; b++) {
            pos++; current_symbol = input_ptr[pos];
            *output_ptr = current_symbol;
            output_ptr++; bytes_out++;
          }
          pos -= chars_per_datum;
        }
        pos += chars_per_datum;
      } else {
        FATAL("Incorrect encoding!");
      }
    } else {
      // Emit directly
      *output_ptr = current_symbol;
      output_ptr++; bytes_out++;
    }
  }
  LOGDEBUG << "Bytes out: " << bytes_out;
  return bytes_out;
}