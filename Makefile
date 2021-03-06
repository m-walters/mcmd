CPP=g++
DEBUG=-g
LINKER=g++
FFLAGS=-O3 -std=c++11
LFLAGS=
CFLAGS= -c -Wall $(DEBUG)

COMMONSOURCES=main.cc master.cc obj.cc
COMMONOBJS=main.o master.o obj.o 

CLEAN=rm *.o

default: mcrod

mcrod: $(COMMONOBJS)
	$(LINKER) $(LFLAGS) $(DEBUG) $(COMMONOBJS) -o $@
	$(CLEAN)

$(COMMONOBJS):
	$(CPP) $(FFLAGS) $(CFLAGS) $(COMMONSOURCES)

main.o: main.cc master.h
master.o: master.h
obj.o: obj.h

clean:
	$(CLEAN)
