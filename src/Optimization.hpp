#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include "VectorOperations.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>

// Enumeration for different optimization methods
enum class OptimizationMethod
{
    Gradient,
    NesterovIteration,
    HeavyBall,
    ADAM
};

// Enumeration for different step size strategies
enum class StepSizeStrategy
{
    ExponentialDecay,
    InverseDecay,
    ApproximateLineSearch
};

// Struct to hold optimization parameters
struct OptimizationParams
{
    std::function<double(const std::vector<double> &)> f;                   // Objective function
    std::function<std::vector<double>(const std::vector<double> &)> grad_f; // Gradient of the objective function
    std::vector<double> initial_guess;   // Initial guess for optimization
    //@note if you initialize in-class you have also the defaults automatically                                   
    // for isntance double alpha_0=1.0;
    double alpha_0;                                                         // Initial step size
    double mu;                                                              // Decay parameter for step size
    double sigma;                                                           // Parameter for Armijo rule
    double epsilon_r;                                                       // Residual tolerance
    double epsilon_s;                                                       // Step length tolerance
    int max_iterations;                                                     // Maximum number of iterations
    //Nesterov or Heavy Ball
    double eta;                                                             // Memory parameter
    //ADAM
    double alpha; // Learning rate (fixed for ADAM)
    double beta1; // Exponential decay rate for first moment estimate
    double beta2; // Exponential decay rate for second moment estimate
    double epsilon; // Small constant to avoid division by zero
};

// Armijo rule for selecting step size
double armijo_rule(const std::vector<double>& x_k, const std::function<double(const std::vector<double>&)>& f,
                   const std::function<std::vector<double>(const std::vector<double>&)>& grad_f,
                   double alpha, double sigma) {
    double grad_norm = norm(grad_f(x_k));
    //@note set a maximumn number of iterations to avoid infinite loop
    // Armijio shoudl always converge if f is continuous and df has a Lipschitz continuity
    // but you never know if numerical errors can make it fail
    while (f(x_k) - f(x_k - alpha * grad_f(x_k)) < sigma * alpha * grad_norm * grad_norm) {
        alpha /= 2.0; // Reduce alpha
    }
    return alpha;
}

// Gradient method for optimization
template<StepSizeStrategy S, OptimizationMethod M>
std::vector<double> gradient_method(const OptimizationParams& params) {
    std::vector<double> x_k = params.initial_guess;
    double alpha_k = params.alpha_0;
    int iter = 0;
    
    if constexpr  (M == OptimizationMethod::Gradient) {   
        while (true) {
            std::vector<double> grad = params.grad_f(x_k);
            
            // Update step size based on strategy
            //@note Nice
            if constexpr (S == StepSizeStrategy::ExponentialDecay) {
                alpha_k = params.alpha_0 * std::exp(-params.mu * iter);
            } else if  constexpr (S == StepSizeStrategy::InverseDecay) {
                alpha_k = params.alpha_0 / (1 + params.mu * iter);
            } else if  constexpr (S == StepSizeStrategy::ApproximateLineSearch) {
                alpha_k = armijo_rule(x_k, params.f, params.grad_f, alpha_k, params.sigma);
            }
            
            // Update x_k
            x_k = x_k - alpha_k * grad;
            
            // Check termination conditions
            //@note you could have avoided the brak by setting properly the
            // while loop conditions, instead of using just "true" 
            if (norm(alpha_k * grad) < params.epsilon_s \
            || \
            std::abs(params.f(x_k) - params.f(x_k - alpha_k * grad)) < params.epsilon_r \
            || \
            iter >= params.max_iterations) {
                break;
            }

            iter++;
        } 

    } else if constexpr (M == OptimizationMethod::NesterovIteration){
        auto x_prev = x_k;
        x_k = x_k - alpha_k * params.grad_f(x_k);
        auto x_new = x_k;
        while (true) {
            // Update step size based on strategy
            if constexpr (S == StepSizeStrategy::ExponentialDecay) {
                alpha_k = params.alpha_0 * std::exp(-params.mu * iter);
            } else if  constexpr (S == StepSizeStrategy::InverseDecay) {
                alpha_k = params.alpha_0 / (1 + params.mu * iter);
            } 

            // Update x_k
            x_new = x_k + params.eta * (x_k - x_prev) - alpha_k * params.grad_f(x_k + params.eta * (x_k - x_prev));
            
            x_prev = x_k;
            x_k = x_new;

            // Check termination conditions
            if (norm(alpha_k * params.grad_f(x_k)) < params.epsilon_s \
            || \
            std::abs(params.f(x_k) - params.f(x_prev)) < params.epsilon_r \
            || \
            iter >= params.max_iterations) {
                break;
            }
            
            iter++;
        }

    }else if constexpr (M == OptimizationMethod::HeavyBall){
        auto x_prev = x_k;
        x_k = x_k - alpha_k * params.grad_f(x_k);
        auto x_new = x_k;
        while (true) {

            // Update step size based on strategy
            if constexpr (S == StepSizeStrategy::ExponentialDecay) {
                alpha_k = params.alpha_0 * std::exp(-params.mu * iter);
            } else if  constexpr (S == StepSizeStrategy::InverseDecay) {
                alpha_k = params.alpha_0 / (1 + params.mu * iter);
            } 

            // Update x_k
            x_new = x_k - alpha_k * params.grad_f(x_k) + params.eta * (x_k - x_prev);
            x_prev = x_k;
            x_k = x_new;

            // Check termination conditions
            if (norm(alpha_k * params.grad_f(x_k)) < params.epsilon_s 
                || \
                std::abs(params.f(x_k) - params.f(x_prev)) < params.epsilon_r 
                || \
                iter >= params.max_iterations) {
                break;
            }
            
            iter++;
        } 
    }else if constexpr (M == OptimizationMethod::ADAM){

        std::vector<double> m(x_k.size(), 0.0); // First moment estimate
        std::vector<double> v(x_k.size(), 0.0); // Second moment estimate

        //@note also here you could have avoided the break by setting properly the
        // while loop conditions, instead of using just "true"
        while (true) {
            iter++;
            std::vector<double> grad = params.grad_f(x_k);
            
            // Update biased first moment estimate
            for (size_t i = 0; i < x_k.size(); ++i) {
                m[i] = params.beta1 * m[i] + (1 - params.beta1) * grad[i];
            }
            
            // Update biased second moment estimate
            for (size_t i = 0; i < x_k.size(); ++i) {
                v[i] = params.beta2 * v[i] + (1 - params.beta2) * grad[i] * grad[i];
            }
            
            // Correct bias in first moment estimate
            std::vector<double> m_hat = 1/ (1 - std::pow(params.beta1, iter)) * m;
            
            // Correct bias in second moment estimate
            std::vector<double> v_hat = 1/ (1 - std::pow(params.beta2, iter)) * v;
            
            // Update parameters
            for (size_t i = 0; i < x_k.size(); ++i) {
                x_k[i] -= params.alpha * m_hat[i] / (std::sqrt(v_hat[i]) + params.epsilon);
            }
            
            // Check termination conditions
            if (norm(m_hat) < params.epsilon_s \
            || \
            norm(grad) < params.epsilon_s \
            || \
            iter >= params.max_iterations) {
                break;
            }
        }
    }
    
    return x_k;
}

#endif