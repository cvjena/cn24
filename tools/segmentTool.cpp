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
      goto list;
    } else if(command.compare(0, 8, "seg-save") == 0) {
      std::string segment_name = "Unnamed segment";
      std::string file_name;
      Conv::ParseStringParamIfPossible(command, "name", segment_name);
      Conv::ParseStringParamIfPossible(command, "file", file_name);

      int segment_index = set.GetSegmentIndex(segment_name);
      if(segment_index >= 0) {
        Conv::Segment* segment = set.GetSegment(segment_index);
        Conv::JSON serialized_segment = segment->Serialize();
        std::string serialized_segment_dump = serialized_segment.dump();
        std::ofstream output(file_name, std::ios::out);
        if(output.good()) {
          output.write(serialized_segment_dump.c_str(), serialized_segment_dump.length());
        } else {
          LOGERROR << "Could not open " << file_name;
        }
      } else {
        LOGERROR << "Segment \"" << segment_name << "\" could not be found!";
      }

    } else if(command.compare(0, 8, "seg-load") == 0) {
      std::string file_name, folder_hint;
      unsigned int range_begin = 0, range_end = (unsigned int)-1;
      Conv::ParseStringParamIfPossible(command, "file", file_name);
      Conv::ParseStringParamIfPossible(command, "hint", folder_hint);
      Conv::ParseCountIfPossible(command, "range_begin", range_begin);
      Conv::ParseCountIfPossible(command, "range_end", range_end);

      Conv::Segment* segment = new Conv::Segment("Unnamed segment");
      Conv::JSON segment_descriptor = Conv::JSON::parse(std::ifstream(file_name, std::ios::in));
      segment->Deserialize(segment_descriptor, folder_hint, range_begin, (signed int)range_end);

      set.AddSegment(segment);
      LOGINFO << "Added segment \"" << segment->name << "\" with " << segment->GetSampleCount() << " samples.";
      goto list;
    } else if(command.compare(0, 8, "set-load") == 0) {
      std::string file_name, folder_hint;
      Conv::ParseStringParamIfPossible(command, "file", file_name);
      Conv::ParseStringParamIfPossible(command, "hint", folder_hint);

      Conv::JSON segment_set_descriptor = Conv::JSON::parse(std::ifstream(file_name, std::ios::in));
      set.Deserialize(segment_set_descriptor, folder_hint);
      LOGINFO << "Deserialized segment set " << set.name;
      goto list;
    } else if(command.compare(0, 8, "set-save") == 0) {
      std::string file_name;
      Conv::ParseStringParamIfPossible(command, "file", file_name);

      Conv::JSON segment_set_descriptor = set.Serialize();
      std::ofstream output(file_name, std::ios::out);
      std::string segment_set_descriptor_serialized = segment_set_descriptor.dump();
      output.write(segment_set_descriptor_serialized.c_str(), segment_set_descriptor_serialized.length());
    } else if(command.compare(0, 4, "list") == 0) {
      list:
      LOGINFO << "Listing segment set \"" << set.name << "\":";
      for(unsigned int s = 0; s < set.GetSegmentCount(); s++) {
        Conv::Segment* segment = set.GetSegment(s);
        LOGINFO << "  Segment " << s << " \"" << segment->name << "\": " << segment->GetSampleCount() << " samples";
      }
    } else if(command.compare(0, 1, "q") == 0) {
      return 0;
    } else {
      LOGWARN << "Unknown command: " << command;
    }
    done:
    continue;
  }
}
