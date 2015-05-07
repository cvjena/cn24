/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Config.h"
#include "Log.h"

#include "ConfigParsing.h"
#include <cstdlib>

#include <sstream>

namespace Conv {
bool StartsWithIdentifier ( std::string line, std::string identifier ) {
  return ( line.compare ( 0, identifier.length(), identifier ) == 0 );
}

unsigned int ParseUInt ( std::string line, std::string identifier ) {
  if ( line.compare ( 0, identifier.length(), identifier ) == 0 ) {
    return std::atoi ( line.substr ( identifier.length() +1 ).c_str() );
  } else {
    return 0;
  }
}

datum ParseDatum ( std::string line, std::string identifier ) {
  if ( line.compare ( 0, identifier.length(), identifier ) == 0 ) {
    std::stringstream ss;
    ss << line.substr ( identifier.length() +1 );
    datum d;
    ss >> d;
//    return std::strtof ( line.substr ( identifier.length() +1 ).c_str() ,nullptr);
    return d;
  } else {
    return 0;
  }
}

std::string ParseString ( std::string line, std::string identifier ) {
  if ( line.compare ( 0, identifier.length(), identifier ) == 0 ) {
    return line.substr ( identifier.length() +1 );
  } else {
    return 0;
  }
}

void ParseStringIfPossible ( std::string line, std::string identifier, std::string& value ) {
  if ( StartsWithIdentifier ( line,identifier ) ) {
    value = ParseString ( line,identifier );
  }
}

void ParseStringParamIfPossible(std::string line, std::string identifier, std::string& s) {
  std::size_t ilen = identifier.length() + 1;
  std::size_t size_pos = line.find ( identifier + "=" );

  if ( size_pos == std::string::npos )
    return;

  std::size_t end_pos = line.find ( " ", size_pos );

  if ( end_pos != std::string::npos )
    line = line.substr ( 0, end_pos );

  std::string sub = line.substr ( size_pos + ilen );
  s = sub;
}


void ParseDatumIfPossible ( std::string line, std::string identifier, datum& value ) {
  if ( StartsWithIdentifier ( line,identifier ) ) {
    value = ParseDatum ( line,identifier );
  }
}

void ParseUIntIfPossible ( std::string line, std::string identifier, unsigned int& value ) {
  if ( StartsWithIdentifier ( line,identifier ) ) {
    value = ParseUInt ( line,identifier );
  }
}

void ParseKernelSizeIfPossible ( std::string line, std::string identifier, unsigned int& kx, unsigned int& ky ) {
  std::size_t ilen = identifier.length() + 1;
  std::size_t size_pos = line.find ( identifier + "=" );

  if ( size_pos == std::string::npos )
    return;

  std::size_t x_pos = line.find ( "x",size_pos );

  if ( x_pos == std::string::npos )
    return;

  std::size_t end_pos = line.find ( " ", x_pos );

  if ( end_pos != std::string::npos )
    line = line.substr ( 0, end_pos );

  std::string size = line.substr ( size_pos + ilen );
  std::string sizex = size.substr ( 0, x_pos - ( size_pos + ilen ) );
  std::string sizey = size.substr ( 1 + x_pos - ( size_pos + ilen ) );

  kx = std::atoi ( sizex.c_str() );
  ky = std::atoi ( sizey.c_str() );
}

void ParseCountIfPossible ( std::string line, std::string identifier, unsigned int& k ) {
  std::size_t ilen = identifier.length() + 1;
  std::size_t size_pos = line.find ( identifier + "=" );

  if ( size_pos == std::string::npos )
    return;

  std::size_t end_pos = line.find ( " ", size_pos );

  if ( end_pos != std::string::npos )
    line = line.substr ( 0, end_pos );

  std::string size = line.substr ( size_pos + ilen );

  k = std::atoi ( size.c_str() );
}

void ParseDatumParamIfPossible ( std::string line, std::string identifier, datum& k ) {
  std::size_t ilen = identifier.length() + 1;
  std::size_t size_pos = line.find ( identifier + "=" );

  if ( size_pos == std::string::npos )
    return;

  std::size_t end_pos = line.find ( " ", size_pos );

  if ( end_pos != std::string::npos )
    line = line.substr ( 0, end_pos );

  std::string size = line.substr ( size_pos + ilen );

  std::stringstream ss;
  ss << size;
  datum d;
  ss >> d;
  
  k = d; //std::atof ( size.c_str() );
}

}
