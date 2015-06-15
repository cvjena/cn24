%{
#include "y.tab.h"
%}
 /*tokens CONFSECTION IDENTIFIER PROPVALN PROPVALDIM SEPARATOR*/

%%
network                 return NETSECTION;
config                  return CONFSECTION;
[a-zA-Z][a-zA-Z0-9]+    return IDENTIFIER;
[0-9\.]+                return PROPVALN;
[0-9\.]+x[0-9\.]+       return PROPVALDIM;
[\t ]+                  ;
(\(|\)|,|\?|=|\{|\}|-)  return yytext[0];
[\n\;]+                 return SEPARATOR;
.                       ;

%%
