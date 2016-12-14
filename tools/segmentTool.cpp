/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>
#include <iostream>
#include <fstream>

#include <private/ConfigParsing.h>

int main(int argc, char** argv) {
  Conv::System::Init();

  Conv::SegmentSet set("Unnamed set");
  while(true) {
    std::cout << "\n > ";

    // Read user input
    std::string command;
    std::getline(std::cin, command);

    if(command.compare(0, 10, "seg-import") == 0) {
      // Import segment from old style list file
      std::string segment_name = "Unnamed segment";
      std::string image_list_fname;
      std::string metadata_fname;
      std::string folder_hint;

      Conv::ParseStringParamIfPossible(command, "name", segment_name);
      Conv::ParseStringParamIfPossible(command, "list", image_list_fname);
      Conv::ParseStringParamIfPossible(command, "meta", metadata_fname);
      Conv::ParseStringParamIfPossible(command, "hint", folder_hint);

      // Open image list
      if(image_list_fname.length() == 0) {
        LOGWARN << "No image list file given!";
        continue;
      }

      std::ifstream image_list(image_list_fname, std::ios::in);
      if(!image_list.good()) {
        LOGWARN << "Cannot open " << image_list_fname << "!";
        continue;
      }

      Conv::JSON metadata = Conv::JSON::object();
      bool metadata_valid = false;
      if(metadata_fname.length() > 0) {
        std::ifstream metadata_file(metadata_fname, std::ios::in);
        if(!metadata_file.good()) {
          LOGWARN << "Cannot open " << metadata_fname << "!";
          continue;
        }
        metadata = Conv::JSON::parse(metadata_file);
        if(metadata.count("samples") == 1 && metadata["samples"].is_array()) {
          metadata_valid = true;
        }
      }

      // Create segment
      Conv::Segment* segment = new Conv::Segment(segment_name);
      unsigned int processed_samples = 0;
      while ( !image_list.eof() ) {
        std::string image_fname;
        std::getline(image_list, image_fname);

        // Skip emtpy lines
        if(image_fname.length() == 0)
          continue;

        // Add metadata
        Conv::JSON sample_json = Conv::JSON::object();
        if(metadata_valid) {
          if(metadata["samples"][processed_samples].is_object()) {
            sample_json = metadata["samples"][processed_samples];
          } else {
            LOGWARN << "Sample " << processed_samples << " is missing metadata!";
            LOGWARN << "Sample " << processed_samples << " filename is \"" << image_fname << "\"";
          }
        }

        // Add filename to sample descriptor
        sample_json["image_filename"] = image_fname;
        bool result = segment->AddSample(sample_json, folder_hint);
        if(!result) {
          LOGERROR << "Could not add sample " << processed_samples << " to segment!";
          LOGERROR << "Sample dump: " << sample_json.dump();
          LOGERROR << "Stopping import.";
          goto done;
        }

        processed_samples++;
      }

      if(metadata_valid && processed_samples != metadata["samples"].size()) {
        LOGWARN << "Image list and metadata count don't match!";
      }

      set.AddSegment(segment);
      LOGINFO << "Added segment \"" << segment->name << "\" with " << segment->GetSampleCount() << " samples.";
    } else if(command.compare(0, 1, "q") == 0) {
      return 0;
    } else {
      LOGWARN << "Unknown command: " << command;
    }
    done:
    continue;
  }
}
