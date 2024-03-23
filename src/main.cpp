#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include "Optimization.hpp"
#include <fstream>

//Constant parameters to set at the beginning
constexpr OptimizationMethod M = OptimizationMethod::ADAM; //optimization method
constexpr StepSizeStrategy S = StepSizeStrategy::ApproximateLineSearch; //stepsize strategy for the update of alpha
//////////////////////

// Centered finite-differences gradient of the test function
auto const numeric_grad = [](const std::function<double(const std::vector<double> &)> f,  double const& h){
    return [f, h](std::vector<double> x){
        std::vector<double> result;
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
    // Define optimization parameters
    OptimizationParams params;
    params.f = test_function;
    //params.grad_f = grad_test_function;
    params.grad_f = numeric_grad(test_function, 1e-4);
    params.initial_guess = {0.0, 0.0};
    params.alpha_0 = 0.1;
    params.mu = 0.2;
    params.sigma = 0.1;
    params.epsilon_r = 1e-6;
    params.epsilon_s = 1e-6;
    params.max_iterations = 1000;
    //Nesterov and HeavyBall
    params.eta = 0.9;
    //ADAM
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
