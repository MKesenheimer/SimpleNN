########################################################################
#                          -*- Makefile -*-                            #
########################################################################

COMPILER = gcc

########################################################################
## Flags
FLAGS   = -g
LDFLAGS =
PREPRO  =
##verbose level 1
#DEBUG   = -D DEBUGV1
##verbose level 2
#DEBUG  += -D DEBUGV2
##verbose level 3
#DEBUG  += -D DEBUGV3
OPT     = -O2
WARN    =

### generate directory obj, if not yet existing
$(shell mkdir -p build)

########################################################################
## Paths

WORKINGDIR = $(shell pwd)
PARENTDIR  = $(WORKINGDIR)/..

########################################################################
## search for the files and set paths

vpath %.c $(WORKINGDIR)
vpath %.o $(WORKINGDIR)/build
UINCLUDE = $(WORKINGDIR)

########################################################################
## Includes
CXX  = $(COMPILER) $(FLAGS) $(OPT) $(WARN) $(DEBUG) $(PREPRO) -I$(UINCLUDE)
INCLUDE = $(wildcard *.h $(UINCLUDE)/*.h)

%.o: %.c $(INCLUDE)
	$(CXX) -c -o build/$@ $<

# Libraries
LIB =

# Frameworks
FRM = 

########################################################################
## Linker files

### USER Files ###
USER = SimpleNN.o NN.o


########################################################################
## Rules
## type make -j4 [rule] to speed up the compilation

BUILD = $(USER)

SimpleNN: $(BUILD)
	  $(CXX) $(patsubst %,build/%,$(BUILD)) $(LDFLAGS) $(LIB) $(FRM) -o $@

clean:
	rm -f build/*.o SimpleNN

do:
	make && ./SimpleNN

########################################################################
#                       -*- End of Makefile -*-                        #
########################################################################
