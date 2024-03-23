#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

//#include "VectorOperations.hpp"
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
template<int N>
struct OptimizationParams
{
    std::function<double(const std::array<double, N> &)> f;                     // Objective function
    std::function<std::array<double, N>(const std::array<double, N> &)> grad_f; // Gradient of the objective function
    std::array<double, N> initial_guess;                                        // Initial guess for optimization
    double alpha_0;                                                             // Initial step size
    double mu;                                                                  // Decay parameter for step size
    double sigma;                                                               // Parameter for Armijo rule
    double epsilon_r;                                                           // Residual tolerance
    double epsilon_s;                                                           // Step length tolerance
    int max_iterations;                                                         // Maximum number of iterations
    
    //Nesterov or Heavy Ball
    double eta;                                                                 // Memory parameter
    
    //ADAM
    double alpha;                                                               // Learning rate (fixed for ADAM)
    double beta1;                                                               // Exponential decay rate for first moment estimate
    double beta2;                                                               // Exponential decay rate for second moment estimate
    double epsilon;                                                             // Small constant to avoid division by zero
};


// Armijo rule for selecting step size
template<int N>
double armijo_rule(const std::array<double, N>& x_k, const std::function<double(const std::array<double, N>&)>& f,
                   const std::function<std::array<double, N>(const std::array<double, N>&)>& grad_f,
                   double alpha, double sigma) {
    double grad_norm = norm<N>(grad_f(x_k));  

    //computing : alpha * grad(x_k)
    auto& vec = grad_f(x_k);
    for(int i = 0 ; i < vec.size() ; ++i)
        vec[i] *= alpha;

    //computing : x_k - alpha*grad(x_k)
    for(int i = 0 ; i < vec.size() ; ++i){
        vec[i] *= -1;
        vec[i] += x_k[i];
    }
    while (f(x_k) - f(vec) < sigma * alpha * grad_norm * grad_norm) {
        alpha /= 2.0; // Reduce alpha
    }
    return alpha;
}

template<int N>
double norm(const std::array<double, N>& vec) {
    double sum = 0.0;
    for (double val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Gradient method for optimization
template<StepSizeStrategy S, OptimizationMethod M, int N>
std::array<double, N> gradient_method(const OptimizationParams<N>& params) {
    std::array<double, N> x_k = params.initial_guess;
    double alpha_k = params.alpha_0;
    int iter = 0;
    
    if constexpr  (M == OptimizationMethod::Gradient) {   
        while (true) {
            std::array<double, N> grad = params.grad_f(x_k);
            
            // Update step size based on strategy
            if constexpr (S == StepSizeStrategy::ExponentialDecay) {
                alpha_k = params.alpha_0 * std::exp(-params.mu * iter);
            } else if  constexpr (S == StepSizeStrategy::InverseDecay) {
                alpha_k = params.alpha_0 / (1 + params.mu * iter);
            } else if  constexpr (S == StepSizeStrategy::ApproximateLineSearch) {
                alpha_k = armijo_rule(x_k, params.f, params.grad_f, alpha_k, params.sigma);
            }
            
            // Update x_k : x_k = x_k - alpha * grad_f(x_k)
            for (int i = 0; i < N ; ++i)
                x_k[i] -= alpha_k * grad[i];
            
            
            // computing : alpha_k * grad_f(x_k)
            auto vec = grad;
            for(int i = 0; i < N ; ++i)
                vec[i] *= alpha_k;

            // computing : x_k - alpha_k * grad_f(x_k)
            auto vec2 = x_k;
            for(int i = 0 ; i<N ; ++i)
                vec2[i] -= vec[i];
            
            // Check termination conditions:
            if (norm<N>(vec) < params.epsilon_s \
            || \
            std::abs(params.f(x_k) - params.f(vec2)) < params.epsilon_r \
            || \
            iter >= params.max_iterations) {
                break;
            }

            iter++;
        } 

    } else if constexpr (M == OptimizationMethod::NesterovIteration){
        auto x_prev = x_k;
        
        //x_1 = x_0 - alpha * grad_f(x_0)
        for(int i = 0 ; i < N ; ++i)
            x_k[i] -= alpha_k * params.grad_f(x_k)[i];

        auto x_new = x_k;
        while (true) {

            // Update step size based on strategy
            if constexpr (S == StepSizeStrategy::ExponentialDecay) {
                alpha_k = params.alpha_0 * std::exp(-params.mu * iter);
            } else  {
                alpha_k = params.alpha_0 / (1 + params.mu * iter);
            } 

            // Update x_k 
            // computing : x_k + eta * (x_k - x_k-1)
            auto prod = x_k;
            for (int i = 0 ; i < N ; ++i)
                prod[i] += params.eta * (x_k[i] - x_prev[i]);
            // computing : x_k+1 = x_k + eta * (x_k - x_k-1) - alpha_k * graf_f(x_k + eta * (x_k - x_k-1))
            for (int i = 0 ; i < N ; ++i)
                x_new[i] = x_k[i] + params.eta * (x_k[i] - x_prev[i]) - alpha_k * params.grad_f(prod)[i];
            
            x_prev = x_k;
            x_k = x_new;

            // Check termination conditions
            prod = params.grad_f(x_k);
            for(int i = 0 ; i< N ; ++i)
                prod[i] *= alpha_k;

            if (norm<N>(prod) < params.epsilon_s \
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
        for(int i = 0 ; i < N ; ++i)
            x_k[i] -= alpha_k * params.grad_f(x_k)[i];
        auto x_new = x_k;
        while (true) {

            // Update step size based on strategy
            if constexpr (S == StepSizeStrategy::ExponentialDecay) {
                alpha_k = params.alpha_0 * std::exp(-params.mu * iter);
            } else {
                alpha_k = params.alpha_0 / (1 + params.mu * iter);
            } 

            // Update x_k
            // compute : x_k+1 = x_k - alpha_k * grad_f(x_k) + eta * (x_k - x_k-1)
            for(int i = 0 ; i < N; i++)
                x_new[i] = x_k[i] - alpha_k * params.grad_f(x_k)[i] + params.eta * (x_k[i] - x_prev[i]);
            x_prev = x_k;
            x_k = x_new;

            // Check termination conditions
            auto prod = params.grad_f(x_k); // alpha_k * grad_f(x_k)
            for(int i = 0 ; i < N ; ++i)
                prod[i] *= alpha_k;
            if (norm(prod) < params.epsilon_s 
                || \
                std::abs(params.f(x_k) - params.f(x_prev)) < params.epsilon_r 
                || \
                iter >= params.max_iterations) {
                break;
            }
            
            iter++;
        } 
    }else if constexpr (M == OptimizationMethod::ADAM){

        std::array<double, N> m{}; // First moment estimate
        std::array<double, N> v{}; // Second moment estimate

        while (true) {
            iter++;
            std::array<double, N> grad = params.grad_f(x_k);
            
            // Update biased first moment estimate
            for (size_t i = 0; i < x_k.size(); ++i) {
                m[i] = params.beta1 * m[i] + (1 - params.beta1) * grad[i];
            }
            
            // Update biased second moment estimate
            for (size_t i = 0; i < x_k.size(); ++i) {
                v[i] = params.beta2 * v[i] + (1 - params.beta2) * grad[i] * grad[i];
            }
            
            // Correct bias in first moment estimate
            std::array<double, N> m_hat;
            for(int i = 0 ; i<N ; ++i)
                m_hat[i] = 1/ (1 - std::pow(params.beta1, iter)) * m[i];

            // Correct bias in second moment estimate
            std::array<double, N> v_hat;
            for(int i = 0 ; i<N ; ++i)
                v_hat[i] = 1/ (1 - std::pow(params.beta2, iter)) * v[i];
            
            // Update parameters
            for (size_t i = 0; i < x_k.size(); ++i) {
                x_k[i] -= params.alpha * m_hat[i] / (std::sqrt(v_hat[i]) + params.epsilon);
            }
            
            // Check termination conditions
            if (norm<N>(m_hat) < params.epsilon_s \
            || \
            norm<N>(grad) < params.epsilon_s \
            || \
            iter >= params.max_iterations) {
                break;
            }
        }
    }

    return x_k;
}

#endif