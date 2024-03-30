CXX      ?= g++
CXXFLAGS ?= -std=c++20
CPPFLAGS ?= -O3 -Wall -I. -Wno-conversion-null -Wno-deprecated-declarations 
# It is better to use a macro, so you can set is as an environmental variable
# set in the .bashrc (or .bash_profile) file
ROOT=$(PACS_ROOT)
CPPFLAGS += -I$(ROOT)/include/muparserx #muparserx path
#CPPFLAGS += -I../../pacs-examples/Extras/muparserx/parser 
CPPFLAGS += -I$(ROOT)/include #json path

EXEC     = main
LDFLAGS ?= -L$(ROOT)/lib -L$(ROOT)/lib64 #muparserx library paths
LIBS  ?= -lmuparserx

all: $(EXEC)

%.o: %.cpp VectorOperations.hpp  Optimization.hpp  muParserXInterface.hpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $<

$(EXEC): %: %.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< $(LIBS) -o $@

clean:
	$(RM) *.o

distclean: clean
	$(RM) *~
