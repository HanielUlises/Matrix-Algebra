#include <iostream>
#include <Eigen/Dense>
#include <print>

#include "utils_print.h"

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> F;
    F.resize(3, 3);
    F.setRandom();

    utils::print("F", F);
    std::println("Raw data of F {:p}", static_cast<const void*> (F.data()));

    Eigen::MatrixXd G(3,3);
    G.setRandom();

    utils::print("G", G);
    std::println("Raw data of F {:p}", static_cast<const void*> (G.data()));

    // Operations

    Eigen::Matrix4d A, B;
    A.setRandom(); B.setRandom();

    utils::print("A", A);
    utils::print("B", B);

    Eigen::Matrix4d C = A + B;
    utils::print("C = A + B", C);

    C = A * 10.0;
    utils::print("C = A * 10.0", C);

    // OPTION 1: Explicit constant (Slightly more overhead)
    C = A + Eigen::Matrix4d::Constant(10.0);
    utils::print("C = A + Eigen::Matrix4d::Constant(10.0)", C);

    // OPTION 2: Scalar addition via array (Preferred/Faster)
    // Performance Note: Using .array() + scalar is generally more efficient as it 
    // avoids the construction of a Constant expression object and maps 
    // directly to vectorized coefficient-wise addition.
    C = A.array() + 100.0; 
    utils::print("C = A + 100", C);

    C = A.cwiseAbs().cwiseSqrt();
    utils::print("A.cwiseAbs().cwiseSqrt()", C);

    C = A.array().abs().sqrt();
    utils::print("A.array().abs().sqrt()", C);

    // Vectors are matrices with one dimensionfixed to one, by default, column vectors
    Eigen::Vector3d s {0.0, 0.0, 1.0};
    Eigen::Vector3d t {0.0, 1.0, 0.0};

    auto dot_product = s.dot(t);
    std::println("Dot product = {}", dot_product);

    auto u = s.cross(t);
    utils::print("s x t", u);
    
    u = t.cross(s);
    utils::print("t x s", u);
}