#### muparserx version (src-muparserx):
With respect to the base version, muparserx has been implemented using the MuParserXInterface.hpp header file (available in the pacs-examples). 
Consequently, to manage vectorial quantities std::array<double, N> has been used, with the additional template parameter "N" which accounts for the input dimension. 
Moreover it is possible to modify the other parameters by editing the data.json file. 
The function variables must be expressed as "x[0], x[1], ...".
The grad_f expression in "data.json" has to be in the format ["df/dx", "df/dy", ...], with a number of arguments equal to "N". Same for the "initial_guess" parameter.

To run the program it is necessary to modify the CPPFLAGS variable in the makefile, by putting the local muparserx and json directories . Then it is sufficient to type `make` and `./main`. 
