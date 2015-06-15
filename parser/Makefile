gnet: lex.yy.c y.tab.c y.tab.h
	gcc -o gnet lex.yy.c y.tab.c
lex.yy.c: gnet.l
	lex gnet.l

y.tab.c: gnet.y
	yacc -d gnet.y

clean:
	rm lex.yy.c y.tab.c y.tab.h gnet

.PHONY: clean
