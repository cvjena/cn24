/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#include <cn24.h>

class TarTestSink : public Conv::MemoryMappedTarFileInfoSink {
public:
  unsigned long mmfile_begin = 0;
  unsigned long mmfile_end = 0;
  void Process(const Conv::MemoryMappedTarFileInfo& info) {
    LOGINFO << "File: " << info.filename << " (" << info.length / 1024 << " KB)";
    if(((unsigned long)info.data) < mmfile_begin || ((unsigned long)info.data) >= mmfile_end) {
      LOGERROR << "  Out of bounds!";
    }
  };
};

int main(int argc, char* argv[]) {
  Conv::System::Init();
  if(argc!=2) {
    LOGINFO << "USAGE: " << argv[0] << " <tar file>";
    LOGEND;
    return 0;
  }

  TarTestSink test_sink;
  Conv::MemoryMappedFile* mmfile = new Conv::MemoryMappedFile(std::string(argv[1]));
  test_sink.mmfile_begin = (unsigned long)mmfile->GetAddress();
  test_sink.mmfile_end = test_sink.mmfile_begin + (unsigned long)mmfile->GetLength();

  Conv::MemoryMappedTar* mmtar = new Conv::MemoryMappedTar(mmfile->GetAddress(), mmfile->GetLength(), &test_sink);
  unsigned int mmtar_file_count = mmtar->GetFileCount();
  if(mmtar_file_count > 0) {
    LOGERROR << "Wrong file count, should be 0 in this case!";
  }
  delete mmtar;
  delete mmfile;
  LOGEND;
  return 0;
}