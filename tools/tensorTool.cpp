/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file tensorTool.cpp
 * @brief Tool to view and edit Tensor contents
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include <cn24.h>
#include <private/Segmentation.h>

bool parseCommand ( Conv::Tensor& tensor, std::string filename );
void help();

int main ( int argc, char* argv[] ) {
  if ( argc < 2 ) {
    LOGERROR << "USAGE: " << argv[0] << " <tensor stream file> <tensor id in file>";
    LOGEND;
    return -1;
  }

  Conv::System::Init();

  // Read tensor id from command line
  std::string s_tid;

  if ( argc > 2 )
    s_tid = argv[2];
  else
    s_tid = "0";

  unsigned int tid = atoi ( s_tid.c_str() );

  // Open tensor stream
  std::ifstream file ( std::string ( argv[1] ), std::ios::in | std::ios::binary );

  // Seek until id is hit
  Conv::Tensor tensor;

  for ( unsigned int t = 0; t < ( tid + 1 ); t++ ) {
    tensor.Deserialize ( file );
  }

  file.close();

  LOGINFO << "Tensor: " << tensor;
  LOGINFO << "Enter \"help\" for information on how to use this program";
  LOGEND;

  bool result;

  do {
    std::cout << "VIS> ";
    result = parseCommand ( tensor, argv[1] );
  } while ( result );

  LOGEND;
  return 0;
}

bool parseCommand ( Conv::Tensor& tensor, std::string filename ) {
  std::string entry;
  std::cin >> entry;

  if ( entry.compare ( "q" ) == 0 || entry.compare ( "quit" ) == 0 )
    return false;

  else if ( entry.compare ( "help" ) ==0 )
    help();

  else if ( entry.compare ( "size" ) == 0 )
    std::cout << tensor << std::endl;

  else if ( entry.compare ( "transpose" ) == 0 )
    tensor.Transpose();

  else if ( entry.compare ( "reshape" ) == 0 ) {
    unsigned int w, h, m, s;
    std::cout << "New width   : ";
    std::cin >> w;
    std::cout << "New height  : ";
    std::cin >> h;
    std::cout << "New maps    : ";
    std::cin >> m;
    std::cout << "New samples : ";
    std::cin >> s;
    std::cout << "Reshaping... ";
    bool result = tensor.Reshape ( s, w, h, m );
    std::cout << result << std::endl;
  }

  else if ( entry.compare ( "extractpatches" ) == 0 ) {
    unsigned int psx, psy, s = 0;
    if(tensor.samples() != 1) {
      std::cout << "Extract patches from sample [0-" << tensor.samples() - 1 << "]: "; std::cin >> s;
    }
    std::cout << "Patch size (x): "; std::cin >> psx;
    std::cout << "Patch size (y): "; std::cin >> psy;
    std::cout << "Extracting...\n";
    Conv::Tensor* target = new Conv::Tensor();
    Conv::Tensor* helper = new Conv::Tensor();
    Conv::Segmentation::ExtractPatches ( psx, psy, *target, *helper, tensor, s );
    tensor.Shadow ( *target );
  }

  else if ( entry.compare ( "write" ) == 0 ) {
    std::ofstream o ( filename, std::ios::out | std::ios::binary );
    tensor.Serialize ( o );
    o.close();
  }

  else if ( entry.compare ( "bin" ) == 0 ) {
    std::ofstream o ( "binoutput.data", std::ios::out | std::ios::binary );

    if ( !o.good() ) {
      std::cout << "Did not work." << std::endl;
      return true;
    }

    tensor.Serialize ( o, true );
    o.close();
  }

#ifdef BUILD_GUI
  else if ( entry.compare ( "show" ) == 0 ) {
    unsigned int s = 0, m = 0;

    if ( tensor.samples() != 1 ) {
      std::cout << "Show sample [0-" << tensor.samples() - 1 << "]: ";
      std::cin >> s;
    }

    if ( tensor.maps() != 1 && tensor.maps() != 3 ) {
      std::cout << "Show map [0-"<< tensor.maps() - 1 << "]: ";
      std::cin >> m;
    }

    Conv::System::viewer->show ( &tensor, filename, false, m, s );
  }

#endif

  else {
    std::cout << "Command not recognized: " << entry << std::endl;
  }

  return true;
}

void help() {
  std::cout << "You can use the following commands:\n";
  std::cout << "  transpose       Transposes every map in every\n"
            << "                  sample (width becomes height)\n"
            << "  bin             Writes the tensors maps to a binary\n"
            << "                  file \"binoutput.data\" using 8 bits per channel\n"
            << "  size            Display the size of the tensor\n"
            << "  reshape         Change the size of the tensor while\n"
            << "                  keeping the number of elements constant\n"
            << "  write		  Writes the tensor back to the file\n"
#ifdef BUILD_GUI
            << "  show            Displays the tensor in a window\n"
#endif
            << "  extractpatches  Extracts patches from the tensor\n"
            << "  help            Displays this information\n"
            << "  q, quit         Quits this program\n"
            << "";

}
