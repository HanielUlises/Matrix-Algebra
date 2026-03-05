#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <print>

#include "utils_print.h"

int main() {
    // Linear system
    Eigen::Matrix3d A;
    A << 2, 1, -1,
        -3, -1, 2,
        -2, 1, 2;

    Eigen::Vector3d b;
    b << 8, -11, -3;

    // Solve Ax = b
    Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);

    utils::print("Soultion x", x);

    Eigen::Vector3d result = A * x;

    utils::print("A * x", result);
    utils::print("Original b", b);

    // Eigen values  and eigen vectors
    Eigen::Matrix3d M;
    M << 0, -3, 8,
        3, 0, -6,
        -8, 6, 0;

    utils::print("Symmetric matrix M", M);

    Eigen::EigenSolver<Eigen::Matrix3d> es(M);
    utils::print("Eigenvalues", es.eigenvalues());
    utils::print("Eigenvectors", es.eigenvectors());

    for (int i = 0; i < 3; i++) {
        auto lambda = es.eigenvalues()[i];
        auto v = es.eigenvectors().col(i);

        utils::print("Mv", M.cast<std::complex<double>>() * v);
        utils::print("λ*v", lambda * v);
    }

    return 0;
}