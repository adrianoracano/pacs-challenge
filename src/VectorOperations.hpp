#ifndef VECTOR_OP_HPP
#define VECTOR_OP_HPP

#include <iostream>
#include <vector>
#include <cmath>


//operators overloading
template<typename T>
std::vector<T> operator*( const double& scalar, const std::vector<T>& vec) {
    std::vector<T> result;
    for (const auto& element : vec) {
        result.push_back(element * scalar);
    }
    return result;
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& vec, const T& scalar) {
    std::vector<T> result;
    for (const auto& element : vec) {
        result.push_back(element * scalar);
    }
    return result;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for subtraction.");
    }

    std::vector<T> result;
    result.reserve(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] - vec2[i]);
    }
    return result;
}

template<typename T>
std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for addition.");
    }

    std::vector<T> result;
    result.reserve(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] + vec2[i]);
    }
    return result;
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for addition.");
    }

    std::vector<T> result;
    result.reserve(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] * vec2[i]);
    }
    return result;
}

// Function to compute the norm of a vector
double norm(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

#endif