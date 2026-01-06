// solver.h
#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>

// =====================================================
//                      CONSTANTS
// =====================================================

constexpr int B = 2;  // Block dimension

// =====================================================
//                   DATA STRUCTURES
// =====================================================

struct SparseBlock {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> val;
};

using DenseBlock = std::array<std::array<double, B>, B>;
using VecBlock = std::array<double, B>;

// =====================================================
//                 BASIC LINEAR ALGEBRA
// =====================================================

DenseBlock to_dense(const SparseBlock& S);

void matvec(const DenseBlock& A, const double x[B], double y[B]);

void matmul(const DenseBlock& A, const DenseBlock& Bm, DenseBlock& C);

void subtract_inplace(DenseBlock& A, const DenseBlock& Bm);

// =====================================================
//                LU DECOMPOSITION (B×B)
// =====================================================

void lu_factor(DenseBlock& A, std::array<int, B>& piv);

void lu_solve_vec(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const double b_in[B],
    double x[B]
);

void lu_solve_mat(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const DenseBlock& Bm,
    DenseBlock& X
);

// =====================================================
//            BLOCK TRIDIAGONAL THOMAS SOLVER
// =====================================================

void solve_block_tridiag(
    const std::vector<SparseBlock>& L_pipe,
    const std::vector<SparseBlock>& D,
    const std::vector<SparseBlock>& R,
    const std::vector<VecBlock>& Q,
    std::vector<VecBlock>& X
);

// =====================================================
//               DEBUG / DIAGNOSTIC
// =====================================================

std::vector<std::vector<double>> build_dense(
    const std::vector<SparseBlock>& Lb,
    const std::vector<SparseBlock>& Db,
    const std::vector<SparseBlock>& Rb
);
