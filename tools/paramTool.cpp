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
  UNREFERENCED_PARAMETER(argc);
  UNREFERENCED_PARAMETER(argv);
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
    } else if (command.compare(0, 13, "load_darknet ") == 0) {
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

      uint32_t major;
      uint32_t minor;
      uint32_t revision;
      uint32_t seen;
      input.read((char*)&major, sizeof(uint32_t));
      input.read((char*)&minor, sizeof(uint32_t));
      input.read((char*)&revision, sizeof(uint32_t));
      input.read((char*)&seen, sizeof(uint32_t));

      bool do_transpose = (major > 1000) || (minor > 1000);
      LOGINFO << "Transpose: " << do_transpose;

      // Check if input file exists
      std::ifstream net_input (net_file_name, std::ios::in);
      if (net_input.good()) {
        LOGINFO << "Loading net file " << net_file_name;
      } else {
        LOGERROR << "Cannot open " << net_file_name;
        continue;
      }

      Conv::JSON net_json = Conv::JSON::parse(net_input)["net"];
      Conv::JSON nodes_json = net_json["nodes"];

      LOGINFO << "Processing " << nodes_json.size() << " nodes";

      unsigned int input_maps = 3;

      for(Conv::JSON::iterator node_json_iterator = nodes_json.begin(); node_json_iterator != nodes_json.end(); ++node_json_iterator) {
        Conv::JSON node_json = node_json_iterator.value();
        if(node_json["layer"].is_object()) {
          std::string layer_type = node_json["layer"]["type"];
          Conv::JSON layer_json = node_json["layer"];
          if(layer_type.compare("convolution") == 0 || layer_type.compare("yolo_output") == 0) {
            LOGINFO << "Processing layer " << node_json_iterator.key();
            unsigned int kernel_width = 1;
            unsigned int kernel_height = 1;
            unsigned int kernel_count = 0;
            if(layer_type.compare("convolution") == 0) {
              kernel_width = layer_json["size"][0];
              kernel_height = layer_json["size"][1];
              kernel_count = layer_json["kernels"];
            } else if(layer_type.compare("yolo_output") == 0) {
              Conv::JSON yolo_config_json = net_json["yolo_configuration"];
              try {
                unsigned int boxes_per_cell = yolo_config_json["boxes_per_cell"];
                unsigned int horizontal_cells = yolo_config_json["horizontal_cells"];
                unsigned int vertical_cells = yolo_config_json["vertical_cells"];
                unsigned int original_classes = yolo_config_json["original_classes"];
                kernel_count = horizontal_cells * vertical_cells * (boxes_per_cell * 5 + original_classes);
                LOGINFO << "Calculated output length of " << kernel_count;
              } catch (const std::exception& ex) {
                FATAL("Exception: " << ex.what() << ". Please check the yolo_configuration part of your JSON file and remember to insert the original_classes property");
              }
            }
            NamedTensorArray* array = new NamedTensorArray(node_json_iterator.key());

            Conv::Tensor* bias_tensor = new Conv::Tensor(1, kernel_count, 1, 1);
            // Read bias tensor
            input.read((char*) bias_tensor->data_ptr(), sizeof(Conv::datum) * kernel_count);

            Conv::Tensor* weight_tensor = new Conv::Tensor(kernel_count, kernel_width, kernel_height, input_maps);
            if(do_transpose) {
              if((layer_json.count("transpose") == 1 && layer_json["transpose"].is_number() && (unsigned int)(layer_json["transpose"]) == 1) || ((layer_type.compare("yolo_output") == 0 ))) {
                if(kernel_width == 1) {
                  LOGINFO << "Transposing weights (simple) for " << layer_type;
                  Conv::Tensor *temp_tensor = new Conv::Tensor(input_maps, kernel_width, kernel_height, kernel_count);
                  input.read((char *) temp_tensor->data_ptr(),
                             sizeof(Conv::datum) * kernel_count * kernel_width * kernel_height * input_maps);

                  for (unsigned int s = 0; s < kernel_count; s++) {
                    for (unsigned int m = 0; m < input_maps; m++) {
                          weight_tensor->data_ptr()[input_maps * s + m] =
                              temp_tensor->data_ptr_const()[kernel_count * m + s];
                    }
                  }

                  delete temp_tensor;
                } else {
                  if(do_transpose) {
                    LOGINFO << "Transposing weights (complex) for " << layer_type;
                    Conv::Tensor *temp_tensor = new Conv::Tensor(input_maps, kernel_width, kernel_height, kernel_count);
                    input.read((char *) temp_tensor->data_ptr(),
                               sizeof(Conv::datum) * kernel_count * kernel_width * kernel_height * input_maps);

                    for (unsigned int s = 0; s < kernel_count; s++) {
                      for (unsigned int m = 0; m < input_maps; m++) {
                        for (unsigned int y = 0; y < kernel_height; y++) {
                          for (unsigned int x = 0; x < kernel_width; x++) {
                            weight_tensor->data_ptr()[(input_maps * kernel_width * kernel_height) * s +
                                                      (kernel_width * kernel_height * m) + (kernel_width * y) + x] =

                                temp_tensor->data_ptr_const()[(kernel_count * kernel_width * kernel_height * m) +
                                                              (kernel_width * kernel_count * y) + (kernel_count * x) + s];
                          }
                        }
                      }
                    }
                    delete temp_tensor;
                  } else {
                  }
                }
              } else {
                input.read((char *) weight_tensor->data_ptr(),
                           sizeof(Conv::datum) * kernel_count * kernel_width * kernel_height * input_maps);
              }
            } else {
              // Old YOLO
              
              if((layer_json.count("transpose") == 1 && layer_json["transpose"].is_number() && (unsigned int)(layer_json["transpose"]) == 1) || ((layer_type.compare("yolo_output") == 0 ))) {
                LOGINFO << "Transposing weights (OLD DN) for " << layer_type;
                Conv::Tensor *temp_tensor = new Conv::Tensor(input_maps, kernel_width, kernel_height, kernel_count);
                input.read((char *) temp_tensor->data_ptr(),
                           sizeof(Conv::datum) * kernel_count * kernel_width * kernel_height * input_maps);
                if(kernel_width == 1 && kernel_height == 1) {
                  LOGINFO << "FLAT (N)";
                  for (unsigned int s = 0; s < kernel_count; s++) {
                    for (unsigned int m = 0; m < input_maps; m++) {
                      weight_tensor->data_ptr()[input_maps * s + m] =
                        // T
                        // temp_tensor->data_ptr_const()[kernel_count * m + s];
                        // N
                        temp_tensor->data_ptr_const()[input_maps * s + m];
                    }
                  }
                } else {
                  LOGINFO << "CONV_CONN";
                  for (unsigned int s = 0; s < kernel_count; s++) {
                    for (unsigned int m = 0; m < input_maps; m++) {
                      for (unsigned int y = 0; y < kernel_height; y++) {
                        for (unsigned int x = 0; x < kernel_width; x++) {
                          weight_tensor->data_ptr()[(input_maps * kernel_width * kernel_height) * s + (kernel_width * kernel_height * m) + (kernel_width * y) + x] =
                              //       T  N N++
                              // SXMY (0  0  0)
                              // temp_tensor->data_ptr_const()[(kernel_height * kernel_width * input_maps) * s + (kernel_height * input_maps) * x + (kernel_height) * m + y];
                              // XSYM (0  0  1)
                              // temp_tensor->data_ptr_const()[(kernel_height * kernel_count * input_maps) * x + (kernel_height * input_maps) * s + (input_maps) * y + m];
                              // MYSX (4  0  1)
                              //temp_tensor->data_ptr_const()[(kernel_width * kernel_count * kernel_height) * m + (kernel_width * kernel_count) * y + (kernel_width) * s + x];
                              // YMXS (4  0)
                              //temp_tensor->data_ptr_const()[(kernel_width * kernel_count * input_maps) * y + (kernel_width * kernel_count) * m + (kernel_count) * x + s];
                              // YXMS (4)
                              //temp_tensor->data_ptr_const()[(kernel_width * kernel_count * input_maps) * y + (input_maps * kernel_count) * x + (kernel_count) * m + s];
                              // MXYS (4)
                              //temp_tensor->data_ptr_const()[(kernel_width * kernel_count * kernel_height) * m + (kernel_height * kernel_count) * x + (kernel_count) * y + s];
                              // MSXY (5)
                              //temp_tensor->data_ptr_const()[(kernel_width * kernel_count * kernel_height) * m + (kernel_height * kernel_width) * s + (kernel_height) * x + y];
                              // MSYX (5)
                              //temp_tensor->data_ptr_const()[(kernel_width * kernel_count * kernel_height) * m + (kernel_height * kernel_width) * s + (kernel_width) * y + x];
                              // SMXY (8 0)
                              //temp_tensor->data_ptr_const()[(kernel_width * input_maps * kernel_height) * s + (kernel_height * kernel_width) * m + (kernel_height) * x + y];
                              // SMYX (8 0)
                              temp_tensor->data_ptr_const()[(kernel_width * input_maps * kernel_height) * s + (kernel_height * kernel_width) * m + (kernel_width) * y + x];
                              // SXYM (  0)
                              //temp_tensor->data_ptr_const()[(kernel_width * input_maps * kernel_height) * s + (kernel_height * input_maps) * x + (input_maps) * y + m];
                              // SYMX (  0)
                              //temp_tensor->data_ptr_const()[(kernel_width * input_maps * kernel_height) * s + (input_maps * kernel_width) * y + (kernel_width) * m + x];
                              // SYXM (  0)
                              //temp_tensor->data_ptr_const()[(kernel_width * input_maps * kernel_height) * s + (input_maps * kernel_width) * y + (input_maps) * x + m];

                              // MYXS (     1)
                              //temp_tensor->data_ptr_const()[(kernel_width * input_maps * kernel_count * y) + (input_maps * kernel_count * x) + (kernel_count * m) + s];
                              //temp_tensor->data_ptr_const()[(kernel_count * kernel_width * kernel_height * m) +
                              //                              (kernel_width * kernel_height * s) + (kernel_width * y) + x];
                              //temp_tensor->data_ptr_const()[(kernel_count * kernel_width * kernel_height * m) +
                              //                              (kernel_width * kernel_count * y) + (kernel_count * x) + s];
                              //temp_tensor->data_ptr_const()[(input_maps * kernel_width * kernel_height * s) +
                              //                              (kernel_width * input_maps * y) + (input_maps * x) + m];
                        }
                      }
                    }
                  }
                }
                delete temp_tensor;
              } else {
                
                LOGINFO << "Not transposing weights for " << layer_type;
                input.read((char *) weight_tensor->data_ptr(),
                           sizeof(Conv::datum) * kernel_count * kernel_width * kernel_height * input_maps);
              }
            }

            // If this is a yolo output layer, we need to move the weights around to support CN24's arrangement of outputs
            if(layer_type.compare("yolo_output") == 0) {
              LOGINFO << "Moving weights around for YOLO output";
              Conv::Tensor* box_weights = new Conv::Tensor, *box_biases = new Conv::Tensor, *class_weights = new Conv::Tensor, *class_biases = new Conv::Tensor;
              Conv::JSON yolo_config_json = net_json["yolo_configuration"];
              unsigned int boxes_per_cell = yolo_config_json["boxes_per_cell"];
              unsigned int horizontal_cells = yolo_config_json["horizontal_cells"];
              unsigned int vertical_cells = yolo_config_json["vertical_cells"];
              unsigned int original_classes = yolo_config_json["original_classes"];

              box_weights->Resize(horizontal_cells * vertical_cells * boxes_per_cell * 5, 1, 1, input_maps);
              box_biases->Resize(horizontal_cells * vertical_cells * boxes_per_cell * 5, 1, 1, 1);
              class_weights->Resize(horizontal_cells * vertical_cells * original_classes, 1, 1, input_maps);
              class_biases->Resize(horizontal_cells * vertical_cells * original_classes, 1, 1, 1);

              unsigned int iou_index = original_classes * vertical_cells * horizontal_cells;
              unsigned int coords_index = iou_index + vertical_cells * horizontal_cells * boxes_per_cell;

              // Loop over all cells
              for (unsigned int vcell = 0; vcell < vertical_cells; vcell++) {
                for (unsigned int hcell = 0; hcell < horizontal_cells; hcell++) {
                  unsigned int cell_id = vcell * horizontal_cells + hcell;
                  // Loop over all possible boxes
                  for (unsigned int b = 0; b < boxes_per_cell; b++) {
                    unsigned int box_coords_index = coords_index + 4 * (boxes_per_cell * cell_id + b);
                    unsigned int box_iou_index = iou_index + cell_id * boxes_per_cell + b;
                    // IoU weights
                    Conv::Tensor::CopySample(*weight_tensor, box_iou_index, *box_weights, ((cell_id * boxes_per_cell + b) * 5) + 4);

                    // IoU biases
                    *(box_biases->data_ptr(0, 0, 0, ((cell_id * boxes_per_cell + b) * 5) + 4)) = bias_tensor->data_ptr_const()[box_iou_index];

                    // Coord weights
                    Conv::Tensor::CopySample(*weight_tensor, box_coords_index, *box_weights, ((cell_id * boxes_per_cell + b) * 5));
                    Conv::Tensor::CopySample(*weight_tensor, box_coords_index + 1, *box_weights, ((cell_id * boxes_per_cell + b) * 5) + 1);
                    Conv::Tensor::CopySample(*weight_tensor, box_coords_index + 2, *box_weights, ((cell_id * boxes_per_cell + b) * 5) + 2);
                    Conv::Tensor::CopySample(*weight_tensor, box_coords_index + 3, *box_weights, ((cell_id * boxes_per_cell + b) * 5) + 3);

                    // Coord biases
                    *(box_biases->data_ptr(0, 0, 0, ((cell_id * boxes_per_cell + b) * 5))) = bias_tensor->data_ptr_const()[box_coords_index];
                    *(box_biases->data_ptr(0, 0, 0, ((cell_id * boxes_per_cell + b) * 5) + 1)) = bias_tensor->data_ptr_const()[
                        box_coords_index + 1];
                    *(box_biases->data_ptr(0, 0, 0, ((cell_id * boxes_per_cell + b) * 5) + 2)) = bias_tensor->data_ptr_const()[
                        box_coords_index + 2];
                    *(box_biases->data_ptr(0, 0, 0, ((cell_id * boxes_per_cell + b) * 5) + 3)) = bias_tensor->data_ptr_const()[
                        box_coords_index + 3];

                  }
                  for (unsigned int c = 0; c < original_classes; c++) {
                    // Class weights
                    Conv::Tensor::CopySample(*weight_tensor, (cell_id * original_classes) + c, *class_weights, (horizontal_cells * vertical_cells * c) + cell_id);
                    // Class biases
                    *(class_biases->data_ptr(0,0,0,(horizontal_cells * vertical_cells * c) + cell_id)) = bias_tensor->data_ptr_const()[(cell_id * original_classes) + c];
                  }
                }
              }
              array->tensors.push_back(box_weights);
              array->tensors.push_back(box_biases);
              array->tensors.push_back(class_weights);
              array->tensors.push_back(class_biases);
            } else {
              array->tensors.push_back(weight_tensor);
              array->tensors.push_back(bias_tensor);
            }

            tensors.push_back(array);
            input_maps = kernel_count;
          }
        }
      }
      
      if(!net_input.eof()) {
        LOGWARN << "Net input has not set EOF bit!";
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
    } else if (command.compare(0, 7, "rename ") == 0) {
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
    } else if (command.compare(0, 4, "dump") == 0) {
      // Rename parameter sets for new models
      unsigned int id = 0;
      Conv::ParseCountIfPossible(command, "id", id);
      unsigned int t = 0;
      Conv::ParseCountIfPossible(command, "tensor", t);

      if(id < tensors.size()) {
        std::string target_file = "binoutput.data";
        Conv::ParseStringParamIfPossible(command, "file", target_file);

        std::ofstream o ( target_file, std::ios::out | std::ios::binary );
        if(!o.good()) {
          LOGERROR << "Could not write to " << target_file << "!";
        } else {
          tensors[id]->tensors[t]->Serialize(o, true);
        }
      }
      else {
        LOGERROR << "No parameter set with id " << id;
      }
    } else if (command.compare(0, 4, "list") == 0) {
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
          << "  dump id=<id> tensor=<tensor> file=<name>\n"
          << "    Writes the specified tensor of the parameter set with the specified id to file in binary format\n\n"
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
