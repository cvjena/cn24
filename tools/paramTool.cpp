/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file paramTool.cpp
 * @brief Tool to manage parameter tensors
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <cn24.h>
#include <private/ConfigParsing.h>

struct NamedTensorArray {
public:
  NamedTensorArray(std::string name) : name(name) {};
  std::string name;
  std::vector<Conv::Tensor*> tensors;
};

int main(int argc, char **argv) {
  std::vector<NamedTensorArray*> tensors;
  Conv::System::Init();

  LOGINFO << "Enter \"help\" for information on how to use this program";
  // Main loop
  while(true) {
    std::cout << '\n' << tensors.size() << " parameter sets loaded.\n > ";

    // Read user input
    std::string command;
    std::getline(std::cin, command);

    // Process command
    if (command.compare(0, 5, "load ") == 0) {
      std::string param_file_name;
      Conv::ParseStringParamIfPossible (command, "file", param_file_name);

      // Check if input file exists
      std::ifstream input (param_file_name, std::ios::in | std::ios::binary);
      if (input.good()) {
        LOGINFO << "Loading model file " << param_file_name;
      } else {
        LOGERROR << "Cannot open " << param_file_name;
        continue;
      }

      // Check for right magic number
      uint64_t magic = 0;
      input.read((char*)&magic, sizeof(uint64_t)/sizeof(char));

      if(magic != CN24_PAR_MAGIC) {
        LOGINFO << "No magic, assuming old format";
        input.seekg (0, std::ios::beg);

        int node_number = 1;
        // Load file using old format
        while(!input.eof() && input.good()) {
          // Prepare node name
          std::stringstream ss;
          ss << "node" << node_number;
          std::string node_name = ss.str();
          LOGINFO << "Using automatic node name " << node_name;

          // Deserialize
          Conv::Tensor* tensor_w = new Conv::Tensor();
          tensor_w->Deserialize(input);
          LOGINFO << "Loaded parameter tensor (w): " << (*tensor_w);
          Conv::Tensor* tensor_b = new Conv::Tensor();
          tensor_b->Deserialize(input);
          LOGINFO << "Loaded parameter tensor (b): " << (*tensor_b);

          NamedTensorArray* array = new NamedTensorArray(node_name);
          array->tensors.push_back(tensor_w);
          array->tensors.push_back(tensor_b);

          tensors.push_back(array);

          input.peek();
          node_number++;
        }

      } else {
        LOGINFO << "Magic found, assuming new format";

        // Load file using new format
        while(input.good() && !input.eof()) {
          // Read node name length
          unsigned int node_unique_name_length;
          unsigned int parameter_set_size;
          input.read((char*)&node_unique_name_length, sizeof(unsigned int) / sizeof(char));
          input.read((char*)&parameter_set_size, sizeof(unsigned int) / sizeof(char));

          // Read node name
          char* node_name_cstr = new char[node_unique_name_length + 1];
          input.read(node_name_cstr, node_unique_name_length);
          node_name_cstr[node_unique_name_length] = '\0';
          std::string node_name(node_name_cstr);
          LOGINFO << "Using node name " << node_name;

          NamedTensorArray* array = new NamedTensorArray(node_name);

          // Read parameters
          for(unsigned int j = 0; j < parameter_set_size; j++) {
            Conv::Tensor* tensor = new Conv::Tensor();
            tensor->Deserialize(input);
            LOGINFO << "Loaded parameter tensor (" << j << "): " << *(tensor);
            array->tensors.push_back(tensor);
          }

          tensors.push_back(array);

          // Update EOF flag
          input.peek();
        }
      }
    }
    if (command.compare(0, 13, "load_darknet ") == 0) {
      std::string param_file_name;
      Conv::ParseStringParamIfPossible(command, "file", param_file_name);

      std::string net_file_name;
      Conv::ParseStringParamIfPossible(command, "net", net_file_name);

      // Check if input file exists
      std::ifstream input (param_file_name, std::ios::in | std::ios::binary);
      if (input.good()) {
        LOGINFO << "Loading model file " << param_file_name;
      } else {
        LOGERROR << "Cannot open " << param_file_name;
        continue;
      }

      input.ignore(4 * sizeof(int));

      // Check if input file exists
      std::ifstream net_input (net_file_name, std::ios::in);
      if (net_input.good()) {
        LOGINFO << "Loading net file " << net_file_name;
      } else {
        LOGERROR << "Cannot open " << net_file_name;
        continue;
      }

      Conv::JSON net_json = Conv::JSON::parse(net_input);
      Conv::JSON nodes_json = net_json["net"]["nodes"];

      LOGINFO << "Processing " << nodes_json.size() << " nodes";

      unsigned int input_maps = 3;

      for(Conv::JSON::iterator node_json_iterator = nodes_json.begin(); node_json_iterator != nodes_json.end(); ++node_json_iterator) {
        Conv::JSON node_json = node_json_iterator.value();
        if(node_json["layer"].is_object()) {
          std::string layer_type = node_json["layer"]["type"];
          Conv::JSON layer_json = node_json["layer"];
          if(layer_type.compare("convolution") == 0) {
            LOGINFO << "Processing layer " << node_json_iterator.key();
            unsigned int kernel_width = layer_json["size"][0];
            unsigned int kernel_height = layer_json["size"][1];
            unsigned int kernel_count = layer_json["kernels"];
            NamedTensorArray* array = new NamedTensorArray(node_json_iterator.key());

            Conv::Tensor* bias_tensor = new Conv::Tensor(1, kernel_count, 1, 1);
            // Read bias tensor
            input.read((char*) bias_tensor->data_ptr(), sizeof(Conv::datum) * kernel_count);

            Conv::Tensor* weight_tensor = new Conv::Tensor(kernel_count, kernel_width, kernel_height, input_maps);
            if(layer_json.count("transpose") == 1 && layer_json["transpose"].is_number() && layer_json["transpose"] == 1) {
              LOGINFO << "Transposing weights";
              Conv::Tensor* temp_tensor = new Conv::Tensor(input_maps, kernel_width, kernel_height, kernel_count);
              input.read((char *) temp_tensor->data_ptr(),
                         sizeof(Conv::datum) * kernel_count * kernel_width * kernel_height * input_maps);

              for(unsigned int s = 0; s < kernel_count; s++) {
                for(unsigned int m = 0; m < input_maps; m++) {
                  for(unsigned int y = 0; y < kernel_height; y++) {
                    for(unsigned int x = 0; x < kernel_width; x++) {
                      weight_tensor->data_ptr()[(input_maps * kernel_width * kernel_height) * s + (kernel_width * kernel_height * m) + (kernel_width * y) + x] =
                          temp_tensor->data_ptr_const()[(kernel_count * kernel_width * kernel_height * m) + (kernel_width * kernel_height * s) + (kernel_width * y) + x];
                    }
                  }
                }
              }

              delete temp_tensor;
            } else {
              input.read((char *) weight_tensor->data_ptr(),
                         sizeof(Conv::datum) * kernel_count * kernel_width * kernel_height * input_maps);
            }

            array->tensors.push_back(weight_tensor);
            array->tensors.push_back(bias_tensor);

            tensors.push_back(array);
            input_maps = kernel_count;
          }
        }
      }

    } else if (command.compare(0, 5, "save ") == 0) {
      std::string param_file_name;
      Conv::ParseStringParamIfPossible(command, "file", param_file_name);

      // Check if output can be written to
      std::ofstream output (param_file_name, std::ios::out | std::ios::binary);
      if (output.good()) {
        LOGINFO << "Writing model file " << param_file_name;
      } else {
        LOGERROR << "Cannot write to " << param_file_name;
        continue;
      }

      uint64_t magic = CN24_PAR_MAGIC;
      output.write((char*)&magic, sizeof(uint64_t)/sizeof(char));

      for (unsigned int i = 0; i < tensors.size(); i++) {
        unsigned int layer_parameters = tensors[i]->tensors.size();
        if (layer_parameters > 0) {
          // Write length of node name
          unsigned int node_unique_name_length = tensors[i]->name.length();
          output.write((const char *) &node_unique_name_length, sizeof(unsigned int) / sizeof(char));
          output.write((const char*)&layer_parameters, sizeof(unsigned int)/sizeof(char));

          // Write node name
          output.write(tensors[i]->name.c_str(), node_unique_name_length);

          // Write parameters
          for(unsigned int j=0; j < tensors[i]->tensors.size(); j++) {
            tensors[i]->tensors[j]->Serialize(output);
          }
        }
      }
    }
    else if (command.compare(0, 7, "rename ") == 0) {
      // Rename parameter sets for new models
      unsigned int id = 0;
      std::string new_node_name = "unnamed";
      Conv::ParseCountIfPossible(command, "id", id);
      Conv::ParseStringParamIfPossible(command, "to", new_node_name);

      if(id < tensors.size()) {
        tensors[id]->name = new_node_name;
        LOGINFO << "Renamed parameter set with id " << id << " to " << new_node_name;
      }
      else {
        LOGERROR << "No parameter set with id " << id;
      }
    }
    else if (command.compare(0, 4, "list") == 0) {
      // List all tensors
      for(unsigned int i=0; i < tensors.size(); i++) {
        LOGINFO << "(" << i << ")\tName: " << tensors[i]->name;
        for(unsigned int j=0; j < tensors[i]->tensors.size(); j++) {
          LOGINFO << "  Size: " << *(tensors[i]->tensors[j]);
        }
      }
    }
    else if(command.compare(0, 4, "help") == 0) {
      std::cout << "You can use the following commands:\n";
      std::cout
          << "  load file=<name>\n"
          << "    Loads parameter sets from the specified file. Format will be detected automatically.\n\n"
          << "  save file=<name>\n"
          << "    Writes the parameters sets to the specified file in the new format.\n\n"
          << "  list\n"
          << "    Lists all parameters sets with their names and ids.\n\n"
          << "  rename id=<id> to=<new name>\n"
          << "    Renames the parameter set with the specified id (see \"list\" command output).\n\n";

    }
    else if (command.compare(0, 1, "q") == 0) {
      // Quit
      break;
    }
    else {
      LOGWARN << "Unknown command: " << command;
    }
  }

  LOGEND;
}
