/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#include <string>

#ifndef CONV_CONFIGPARSING_H
#define CONV_CONFIGPARSING_H

namespace Conv {
/*
 * Parsing utilities
 */
bool StartsWithIdentifier ( std::string line, std::string identifier );
unsigned int ParseUInt ( std::string line, std::string identifier );
datum ParseDatum ( std::string line, std::string identifier );
std::string ParseString ( std::string line, std::string identifier );
void ParseStringParamIfPossible(std::string line, std::string identifier, std::string& s);
void ParseStringIfPossible( std::string line, std::string identifier, std::string& value );
void ParseDatumIfPossible ( std::string line, std::string identifier, datum& value );
void ParseUIntIfPossible ( std::string line, std::string identifier, unsigned int& value ) ;
void ParseKernelSizeIfPossible ( std::string line, std::string identifier, unsigned int& kx, unsigned int& ky );
void ParseCountIfPossible ( std::string line, std::string identifier, unsigned int& k );
void ParseDatumParamIfPossible ( std::string line, std::string identifier, datum& k );

}

#endif