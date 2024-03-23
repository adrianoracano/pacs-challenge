CXX      ?= g++
CXXFLAGS ?= -std=c++20
CPPFLAGS ?= -O3 -Wall -I. -Wno-conversion-null -Wno-deprecated-declarations 
CPPFLAGS += -I../../pacs-examples/Examples/include/muparserx #muparserx path
#CPPFLAGS += -I../../pacs-examples/Extras/muparserx/parser 
CPPFLAGS += -I../../pacs-examples/Examples/include #json path

EXEC     = main
LDFLAGS ?= -L../../pacs-examples/Examples/lib -L../../pacs-examples/Examples/lib64 #muparserx library paths
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