### PACS Challenge
The above code gives the possibility to the user to compute the minimum of a given function. 
The user can choose among 4 different methods : 
- standard gradient method
- Heavy Ball method
- Nesterov iteration method
- ADAM

For the first three the step size can be adjusted by the inverse or exponential decay method (for the gradient method, the Armijo rule can also be chosen). To modify the objective function or the method, it is necessary to edit the main file as follows:
- change the "test_function" and the "gradient_test_function" functions accordingly;
- change the template parameter "OptimizationMethod", the available choices are : "Gradient", "NesterovIteraton", "HeavyBall", "ADAM";
- change the template parameter "StepSizeStrategy", the available choices are : "InverseDecay", "ExponentialDecay", "ApproximateLineSearch".

Note : the last choice has no effect on the ADAM algorithm that uses a constant step size : params.alpha. Moreover, the choice of the "ApprximateLineSearch" strategy (Armijo rule) is effective only for the standard gradient method, which means that if NesterovIteration or HeavyBall are chosen as template parameter, the program will use the InverseDecay strategy to update alpha_k.

To run the program it is necessary to run `make` and `./main`.
