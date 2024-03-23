#include <iostream>
#include <fstream>
#include "Optimization.hpp"
#include "muParserXInterface.hpp"
#include "json.hpp"

using json = nlohmann::json;
using namespace MuParserInterface;

//===================================== CONSTANT PARAMETERS =======================================
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

constexpr int N = 2; 
//Input dimension, by default = 2. The expressions of f and grad(f) must be written accordingly.
//################################################################################################


// Centered finite-differences gradient of the test function
template<int N>
auto const numeric_grad = [](const std::function<double(const std::array<double, N> &)> f,  double const& h){
    return [f, h](std::array<double, N> x){
        std::array<double, N> result;
        for(size_t i = 0 ; i < x.size() ; ++i){
            auto x_plus = x;
            auto x_minus = x;
            x_plus[i] = x[i] + h;
            x_minus[i] = x[i] - h;
            result[i] = ((f(x_plus) - f(x_minus))/(2*h));
        }
        return result;
        };
};

// Lambda expression to generate the gradient function
template<int N>
auto const lambda_grad = [](const std::array<muParserXInterface<N>, N> grad_f){
    return [grad_f](std::array<double, N> x){
        std::array<double, N> result;
        for(size_t i = 0 ; i < N; ++i){
            result[i] = grad_f[i](x);
        }
        return result;
    };
};

int main() {
    std::ifstream ifs("data.json");
    json data = json::parse(ifs);

    //Define the parameters struct
    OptimizationParams<N> params;

    //============================= Reading the expression of f: =================================
    std::string f_expr = data.value("f", "");
    muParserXInterface<N> f(f_expr);
    params.f = f;

    //============================ Reading the expression of grad_f: =============================
    
    //The user can choose the analytical expression of grad_f by setting the "use_finite_diff" pa-
    //rameter in the data file to 1. In this case an array of "muParserXInterface" is created and
    //converted into a function object by the "lambda_grad" function.
    //Otherwise the (centered) finite-differences version of grad_f is used and the user can also 
    //set the "h" parameter, the increment value. 

    const int finite_diff = data.value("use_finite_diff", 0);
    if( finite_diff == 0 ){
        std::array<std::string, N> grad_expr = data["grad_f"].get<std::array<std::string, N>>();
        std::array<muParserXInterface<N>, N> grad_f;
        for (int i = 0 ; i < N ; ++i)
            grad_f[i] = muParserXInterface<N>(grad_expr[i]);
        params.grad_f = lambda_grad<N>(grad_f);
    }else{
        const double h = data.value("h", 1e-4);
        params.grad_f = numeric_grad<N>(f, h); 
    }

    //Memory parameter for Nesterov iterations or heavy ball methods
    params.eta = data.value("eta", 0.9);

    //Parameters for ADAM method (the learning rate is fixed in this case)
    params.epsilon = data.value("epsilon", 1e-8);
    params.alpha = data.value("alpha", 0.01);
    params.beta1 = data.value("beta1", 0.9);
    params.beta2= data.value("beta2", 0.999);

    //Other parameters:
    //initial guess
    params.initial_guess = data["initial_guess"].get<std::array<double, N>>(); 

    //tolerances
    params.epsilon_r = data.value("epsilon_r", 1e-6);                          
    params.epsilon_s = data.value("epsilon_s", 1e-6);

    //parameters for the learning rate update
    params.mu = data.value("mu", 0.1);
    params.sigma = data.value("sigma", 0.1);

    //initial learning rate
    params.alpha_0 = data.value("alpha_0", 0.1);
    params.max_iterations = data.value("max_iterations", 1000);


    //Perform optimization
    auto result = gradient_method<S, M, N>(params);

    // Output result
    std::cout << "Minimum found at x = [ ";
    for (double val : result) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "The value of f is " << f(result) << std::endl;
    
    return 0;
}
