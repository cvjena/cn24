%{
  #include <stdio.h>
%}

%token NETSECTION CONFSECTION IDENTIFIER PROPVALN PROPVALDIM SEPARATOR
%%
network_file:  section
              | section SEPARATOR
              | section SEPARATOR network_file
              | SEPARATOR network_file
              | SEPARATOR;

section:       conf_section
                  { printf("Finished conf section\n"); }
              | net_section
                  { printf("Finished net section\n"); };

conf_section:   CONFSECTION '{' conf_slist '}'
              | CONFSECTION '-';
conf_slist:     conf_statement
              | conf_statement SEPARATOR
              | conf_statement SEPARATOR conf_slist
              | SEPARATOR conf_slist
              | SEPARATOR;
conf_statement: property;

net_section:    NETSECTION '{' node_slist '}'
              | NETSECTION '-';
node_slist:     node_statement
              | node_statement SEPARATOR
              | node_statement SEPARATOR node_slist
              | SEPARATOR node_slist
              | SEPARATOR;

node_statement: node_list IDENTIFIER
                  { printf("Node without name, without props\n"); }
              | IDENTIFIER node_list IDENTIFIER
                  { printf("Node with name, without props\n"); }
              | IDENTIFIER node_list IDENTIFIER property_list
                  { printf("Complete node\n"); };
property_list:  property property_list
              | property;
property:        IDENTIFIER '=' PROPVALN
              | IDENTIFIER '=' PROPVALDIM;
node_list:      '?'
              | '(' IDENTIFIER ')'
              | '(' IDENTIFIER ',' IDENTIFIER ')';

%%
int main(void) {
  return yyparse();
}

int yywrap() {
  return 1;
}

int yyerror(const char* str)
{
  printf("Error: %s\n", str);
}
