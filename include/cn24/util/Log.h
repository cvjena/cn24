/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file Log.h
 * \brief Contains logging macros.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_LOG_H
#define CONV_LOG_H

#include <iostream>
#include <string>
#include <stdexcept>

#ifdef __GNUC__
inline std::string methodName (const std::string& prettyFunction) {
  size_t colons = prettyFunction.find ("::");
  if (colons == std::string::npos)
    colons = prettyFunction.find ("(");
  size_t begin = prettyFunction.substr (0, colons).rfind (" ") + 1;
  size_t mid = prettyFunction.find ("Conv::");
  if (mid != std::string::npos)
    begin = mid + 6;

  size_t end = prettyFunction.rfind ("(") - begin;

  return prettyFunction.substr (begin, end);
}

#define LOGERROR (std::cerr << std::endl << "\033[1;31mERR [ " << \
                  methodName(__PRETTY_FUNCTION__) << "(" << __LINE__ << ") ] " \
                  "\033[0m" << std::flush )
#define LOGWARN (std::cerr << std::endl << "\033[1;33mWRN [ " << \
                  methodName(__PRETTY_FUNCTION__) << "(" << __LINE__ << ") ] " \
                  "\033[0m" << std::flush )
#define LOGINFO (std::cout << std::endl << "\033[1mINF [ " << \
                 methodName(__PRETTY_FUNCTION__) << "(" << __LINE__ << ") ] " \
                 "\033[0m" << std::flush )
#define LOGDEBUG (std::cout << std::endl << "DBG [ " << \
                  methodName(__PRETTY_FUNCTION__) << "(" << __LINE__ << ") ] " \
                  << std::flush )

#else

#define LOGERROR (std::cerr << std::endl << "ERR [ " << __FUNCTION__ << "(" << \
                  __LINE__ << ") ] " << std::flush)
#define LOGWARN (std::cerr << std::endl << "WRN [ " << __FUNCTION__ << "(" << \
                  __LINE__ << ") ] " << std::flush)
#define LOGINFO (std::cout << std::endl << "INF [ " << __FUNCTION__ << "(" << \
                 __LINE__ << ") ] " << std::flush)
#define LOGDEBUG (std::cout << std::endl << "DBG [ " << __FUNCTION__ << "(" << \
                  __LINE__ << ") ] " << std::flush)
#endif

#define LOGRESULT (std::cout << std::endl << "\033[1;32mRESULT --- ")
#define LOGTRESULT (std::cout << std::endl << "\033[1;34mRESULT --- ")
#define LOGRESULTEND "\033[0m" << std::flush

#define LOGEND {std::cout << std::endl; std::cerr << std::endl;}

#define FATAL(x) { LOGERROR << "FATAL: " << x << std::endl; \
    throw(std::runtime_error("See log for details.")); }

#endif
