#### base version (src):
The base version is implemented through the use of std::vector, for which operations of elemnent-wise sum, difference and product with a scalar have been defined in "VectorOperations.hpp".
The parameters struct and the main methods are implemented in the "Optimization.hpp" file. The "gradient_method" function has two template parameters, "OptimizationMethod" and "StepSizeStrategy" and the choice between them is performed using the "if constexpr" syntax, in order to increase code efficency.
To modify the objective function or the used algorithm, it is necessary to edit manually the main file as follows:
- change the "test_function" and the "gradient_test_function" functions accordingly;
- change the template parameter "OptimizationMethod", the available choices are : "Gradient", "NesterovIteraton", "HeavyBall", "ADAM";
- change the template parameter "StepSizeStrategy", the available choices are : "InverseDecay", "ExponentialDecay", "ApproximateLineSearch".

Note : the last choice has no effect on the ADAM algorithm that uses a constant step size : params.alpha. Moreover, the choice of the "ApproximateLineSearch" strategy (Armijo rule) is effective only for the standard gradient method, which means that if NesterovIteration or HeavyBall are chosen as template parameter, the program will use the InverseDecay strategy by default to update alpha_k.
For the computation of the gradient, a finite-differences approximation can also be used.

To run the code type `make` and `./main`.
