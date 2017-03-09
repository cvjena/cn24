/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>

#include <fstream>
#include <cmath>

int main(int argc, char** argv) {
  Conv::System::Init();

  if(argc != 3) {
    LOGERROR << "Usage: " << argv[0] << " <file1> <file2>";
    return -1;
  }

  std::ifstream file1(argv[1], std::ios::in | std::ios::binary);
  std::ifstream file2(argv[1], std::ios::in | std::ios::binary);

  if(!file1.good() || !file2.good()) {
    LOGERROR << "Cound not open inputs!";
    return -1;
  }

  // Read magic bits
  uint64_t magic = 0;
  file1.read((char*)&magic, sizeof(uint64_t)/sizeof(char));
  if(magic != CN24_PAR_MAGIC) {
    LOGERROR << "File 1 has wrong magic!";
    return -1;
  }

  file2.read((char*)&magic, sizeof(uint64_t)/sizeof(char));
  if(magic != CN24_PAR_MAGIC) {
    LOGERROR << "File 2 has wrong magic!";
    return -1;
  }

  std::cout << "\n\n";

  while(!file1.eof()) {

    // Read header
    unsigned int node_unique_name_length1;
    unsigned int parameter_set_size1;
    file1.read((char *) &node_unique_name_length1, sizeof(unsigned int) / sizeof(char));
    file1.read((char *) &parameter_set_size1, sizeof(unsigned int) / sizeof(char));

    unsigned int node_unique_name_length2;
    unsigned int parameter_set_size2;
    file2.read((char *) &node_unique_name_length2, sizeof(unsigned int) / sizeof(char));
    file2.read((char *) &parameter_set_size2, sizeof(unsigned int) / sizeof(char));

    if(node_unique_name_length1 != node_unique_name_length2 || parameter_set_size1 != parameter_set_size2) {
      LOGERROR << "Header match error, aborting...";
      return -1;
    }

    // Read node name
    char *node_name_cstr = new char[node_unique_name_length1 + 1];

    file1.read(node_name_cstr, node_unique_name_length1);
    file2.read(node_name_cstr, node_unique_name_length1);
    node_name_cstr[node_unique_name_length1] = '\0';

    std::string node_name(node_name_cstr);

    // Read data
    for(unsigned int p = 0; p < parameter_set_size1; p++) {
      Conv::Tensor t1, t2;
      t1.Deserialize(file1);
      t2.Deserialize(file2);
      if(t1.elements() != t2.elements()) {
        LOGERROR << "Data match error, aborting...";
        return -1;
      }

      // Calculate L2 diff
      Conv::datum sq_sum = 0;
      for(unsigned int e = 0; e < t1.elements(); e++) {
        sq_sum += (t1(e) - t2(e)) * (t1(e) - t2(e));
      }

      sq_sum = (Conv::datum)std::sqrt(sq_sum);

      //std::cout << node_name << "(" << p << ");" << sq_sum << "\n";
      if(p== 0) {
        std::cout << node_name << ";" << sq_sum << "\n";
      }
    }

    // Update EOF flag
    file1.peek();
    file2.peek();
  }

  std::cout << "\n\n";
  LOGEND;
}
