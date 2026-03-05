#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <print>

#include "utils_print.h"

int main() {
    Eigen::Matrix4d A;
    Eigen::VectorXd values(A.size());
    values.setLinSpaced(A.size(), 1, 16);
    A = values.reshaped(4, 4).transpose();

    utils::print("A", A);
    std::println("Raw data of A {:p}", static_cast<const void*> (A.data()));

    // auto B = A.reshaped(16, 1);

    // Slicing and row-column assignments
    Eigen::MatrixXd B (10,10);
    B.setZero();

    B.row(3) << 1,2,3,4,5,5,6,7,8,10;
    B.col(3) << 1,2,3,4,5,5,6,7,8,10;

    utils::print("B", B);

    // 
    Eigen::MatrixXd C (10,10);
    C.setZero();

    C(Eigen::seq(3,5), Eigen::seq(3,5)).setConstant(1);
    C(Eigen::seq(0,9,2), Eigen::seq(0,9,2)).setConstant(2);

    utils::print("C", C);

    C(Eigen::all, Eigen::last).setConstant(3);
    C(Eigen::all, Eigen::last - 1).setConstant(4);

    utils::print("C", C);

    Eigen::MatrixXd D(10,10);
    Eigen::VectorXd values_r(100);
    values_r.setLinSpaced(100, 1, 100);
    D = values_r.reshaped(10, 10);

    utils::print("D", D);

    // Extracting
    Eigen::Matrix<double, 4, 4> m1;
    m1.reshaped() = Eigen::VectorXd::LinSpaced(16, 1, 16);
    utils::print("m₁ ", m1);

    auto top_left = m1.block(0,0,2,2);
    auto bottom_right = m1.block(2,2,2,2);

    utils::print("m₁ top left: ", top_left);
    utils::print("m₁ bottom right: ", bottom_right);

    // Row and column operations
    Eigen::Matrix<double, 3, 4> m2;
    m2.setRandom();

    utils::print("m₂ ", m2);
    Eigen::VectorXd secondRow = m2.row(1);
    Eigen::VectorXd thirdCol = m2.col(2);

    m2.row(0) << 1.0, 2.0, 3.0, 4.0;

    m2.col(0).swap(m2.col(3));

    utils::print("Modified m₂:", m2);

    // Slicing
    Eigen::Matrix<double, 5, 5> m_t;
    A << 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25;
    
    utils::print("m_t", m_t);
    std::vector<Eigen::Index> rows = {1, 3, 4};
    std::vector<Eigen::Index> columns = {0, 2};

    Eigen::MatrixXd m_u = m_t(rows, columns);
    utils::print("mu", m_u);

    // Diagonal
    utils::print("m_t diagonal", m_t.diagonal());

    auto n = m_t.rows();
    Eigen::VectorXd anti_diagonal(n);

    for(int i = 0; i < n; i++)
        anti_diagonal(i) = m_t(i, n - i -1);

    utils::print("m_t antidiagonal", anti_diagonal);


    // Resizing and reshaping
    Eigen::Matrix<double, 2, 3> m3; m3.setRandom();
    utils::print("m₃ (2x3)", m3);
    auto m3_mod = m3.reshaped(3,2);
    utils::print("m₃ (3x2)", m3_mod);

    Eigen::Vector<double, 6> v1; v1.setRandom();
    utils::print("v₁", v1);
    auto m_v1 = v1.reshaped(2,3);
    utils::print("(2x3) matrix from v₁ ", m_v1);
    
    // Concatenation
    Eigen::Matrix<double, 2, 2> m_a, m_b;
    m_a.setZero(), m_b.setOnes();

    auto concat_h = [](const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        assert(A.rows() == B.rows());

        Eigen::MatrixXd C(A.rows(), A.cols() + B.cols());
        C << A, B;

        return C;
    };

    auto concat_v = [](const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        assert(A.cols() == B.cols());

        Eigen::MatrixXd C(A.cols() + B.rows(), A.cols());
        C << A,
            B;

        return C;
    };

    auto concat_a_b = concat_h(m_a, m_b);
    utils::print("Horizontal concatenation of A and B ", concat_a_b);

    concat_a_b = concat_v(m_a, m_b);
    utils::print("Vertical concatenation of A and B ", concat_a_b);

}