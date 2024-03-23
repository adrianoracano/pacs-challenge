# PACS Challenge:
### How to run the code:
The repository is divided in:
- src
- src-muparserx

To run the program it is necessary to type `make` and `./main`. In the muparserx version, you need to substitute the muparserx and the json directories with your directories.

### Brief explanation:
The above code gives the possibility to the user to compute the minimum of a given function. 
The user can choose among 4 different methods : 
- standard gradient method
- heavy ball method
- Nesterov iteration method
- ADAM

For the first three the step size can be adjusted by the inverse or exponential decay method (for the gradient method, the Armijo rule can also be chosen). 
