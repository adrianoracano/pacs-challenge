#include <iostream>
#include "Optimization.hpp"

//===================================== CONSTANT PARAMETERS =======================================
//@note Fine the use of enumerators
constexpr OptimizationMethod M = OptimizationMethod::ADAM; 
//Optimization method, possible choices:
// - Gradient
// - NesterovIteration
// - HeavyBall
// - ADAM

constexpr StepSizeStrategy S = StepSizeStrategy::ExponentialDecay; 
//Step-size strategy, available choices (working if M != ADAM):
// - ExponentialDecay
// - InverseDecay
// - ApproximateLineSearch (working only if M == Gradient)

/* Stepsize strategy for the update of alpha: this choice has no effect on the ADAM algorithm that 
uses a constant step size : params.alpha.
Moreover, the choice of the ApproximateLineSearch strategy (Armijo rule) is possible only for the
standard gradient method, which means that if NesterovIteration or HeavyBall are chosen as templa-
-te parameter, the program will use by default the InverseDecay strategy to update alpha_k.
 */

//@note You could have put this in a separate file, since it is a common utility function
// that you may want to use also eslewhere
// Centered finite-differences gradient of the test function
auto const numeric_grad = [](const std::function<double(const std::vector<double> &)> f,  double const& h){
    return [f, h](std::vector<double> x){
        std::vector<double> result;
        //@note if you use push_back reserve the vector. More efficient
        // result.reserve(x.size());
        for(size_t i = 0 ; i < x.size() ; ++i){
            auto x_plus = x;
            auto x_minus = x;
            x_plus[i] = x[i] + h;
            x_minus[i] = x[i] - h;
            result.push_back((f(x_plus) - f(x_minus))/(2*h));
        }
        return result;
        };
};

// Test function
double test_function(const std::vector<double>& x) {
    return x[0] * x[1] + 4 * x[0] * x[0] * x[0] * x[0] + x[1] * x[1] + 3 * x[0];
}

// Gradient of the test function
std::vector<double> grad_test_function(const std::vector<double>& x) {
    return {x[1] + 16 * x[0] * x[0] * x[0] + 3, x[0] + x[1] * 2};
}

int main() {

    //Parameters struct
    OptimizationParams params;

    params.f = test_function;
    params.grad_f = grad_test_function;
    //params.grad_f = numeric_grad(test_function, 1e-4); //uncomment to use the finite-differences version
    //@note it would be better read this parameter from a file.
    
    params.initial_guess = {0.0, 0.0};
    params.alpha_0 = 0.1;
    params.mu = 0.2;
    params.sigma = 0.1;
    params.epsilon_r = 1e-6;
    params.epsilon_s = 1e-6;
    params.max_iterations = 1000;

    //Nesterov iteration and HeavyBall parameters
    params.eta = 0.9;
    //ADAM parameters
    params.alpha = 0.01;
    params.beta1 = 0.9;
    params.beta2 = 0.999;
    params.epsilon = 1e-8;

    // Perform optimization
    std::vector<double> result = gradient_method<S,M>(params);
    
    // Output result
    std::cout << "Minimum found at: ";
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "The value of f is "<< test_function(result) << std::endl;
    
    return 0;
}
