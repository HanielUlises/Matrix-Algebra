#include <iostream>
#include <Eigen/Dense>
#include <print>

#include "utils_print.h"

int main() {
    Eigen::RowVector<double, 5> v1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::cout << v1 << std::endl;

    Eigen::Matrix<double, 3, 3> m1; m1.setZero();
    Eigen::Matrix<double, 2, 4> m2 {
        {4, 3, 6, 8},
        {5, 7, 15, 10}
    };

    std::println("m2.sum() = {}", m2.sum());
    std::println("m2.mean() = {}", m2.mean());
    std::println("m2.maxCoeff() = {}", m2.maxCoeff());

    Eigen::RowVector<double, 11> v2 = {-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    utils::print("Original vector: ", v2);
    auto v2_mod = v2.cwiseAbs().cwiseSqrt().unaryExpr([] (double x) {return std::exp(x);});
    utils::print("Calculate the sum, mean, minimum, and maximum values of the vector.", v2_mod);

    Eigen::RowVector<double, 3> f_v = {3.0, 4.0, 6.0};
    Eigen::RowVector<double, 3> s_v = {4.0, 9.0, 7.0};

    utils::print("First vector: ", f_v);
    utils::print("Second vector: ", s_v);

    auto mult = f_v.array() * s_v.array();
    utils::print("Element wise multiplication: ", mult);

    auto div = f_v.array() / s_v.array();
    utils::print("Element wise division: ", div);

    auto pow = f_v.array().pow(s_v.array());
    utils::print("Element wise power: ", pow);
}