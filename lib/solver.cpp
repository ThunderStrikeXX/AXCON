// solver.cpp
#include "solver.h"

// =====================================================
//                 BASIC LINEAR ALGEBRA
// =====================================================

DenseBlock to_dense(const SparseBlock& S) {
    DenseBlock M{};
    for (std::size_t k = 0; k < S.val.size(); ++k) {
        M[S.row[k]][S.col[k]] = S.val[k];
    }
    return M;
}

void matvec(const DenseBlock& A, const double x[B], double y[B]) {
    for (int i = 0; i < B; ++i) {
        double s = 0.0;
        for (int j = 0; j < B; ++j)
            s += A[i][j] * x[j];
        y[i] = s;
    }
}

void matmul(const DenseBlock& A, const DenseBlock& Bm, DenseBlock& C) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j) {
            double s = 0.0;
            for (int k = 0; k < B; ++k)
                s += A[i][k] * Bm[k][j];
            C[i][j] = s;
        }
}

void subtract_inplace(DenseBlock& A, const DenseBlock& Bm) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j)
            A[i][j] -= Bm[i][j];
}

void add(SparseBlock& B, int p, int q, double v) {
    B.row.push_back(p);
    B.col.push_back(q);
    B.val.push_back(v);
}

// =====================================================
//                 LU DECOMPOSITION
// =====================================================

void lu_factor(DenseBlock& A, std::array<int, B>& piv) {
    for (int i = 0; i < B; ++i)
        piv[i] = i;

    for (int k = 0; k < B; ++k) {
        int p = k;
        double maxv = std::fabs(A[k][k]);

        for (int i = k + 1; i < B; ++i) {
            double v = std::fabs(A[i][k]);
            if (v > maxv) {
                maxv = v;
                p = i;
            }
        }

        if (maxv == 0.0)
            throw std::runtime_error("LU: singular matrix");

        if (p != k) {
            std::swap(piv[k], piv[p]);
            for (int j = 0; j < B; ++j)
                std::swap(A[k][j], A[p][j]);
        }

        for (int i = k + 1; i < B; ++i) {
            A[i][k] /= A[k][k];
            double lik = A[i][k];
            for (int j = k + 1; j < B; ++j)
                A[i][j] -= lik * A[k][j];
        }
    }
}

void lu_solve_vec(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const double b_in[B],
    double x[B]
) {
    double y[B];

    for (int i = 0; i < B; ++i)
        y[i] = b_in[piv[i]];

    for (int i = 0; i < B; ++i)
        for (int j = 0; j < i; ++j)
            y[i] -= LU[i][j] * y[j];

    for (int i = B - 1; i >= 0; --i) {
        for (int j = i + 1; j < B; ++j)
            y[i] -= LU[i][j] * x[j];
        x[i] = y[i] / LU[i][i];
    }
}

void lu_solve_mat(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const DenseBlock& Bm,
    DenseBlock& X
) {
    for (int col = 0; col < B; ++col) {
        double b[B], x[B];

        for (int i = 0; i < B; ++i)
            b[i] = Bm[i][col];

        lu_solve_vec(LU, piv, b, x);

        for (int i = 0; i < B; ++i)
            X[i][col] = x[i];
    }
}

// =====================================================
//             BLOCK TRIDIAGONAL SOLVER
// =====================================================

void solve_block_tridiag(
    const std::vector<SparseBlock>& L_pipe,
    const std::vector<SparseBlock>& D,
    const std::vector<SparseBlock>& R,
    const std::vector<VecBlock>& Q,
    std::vector<VecBlock>& X
) {
    const int Nx = static_cast<int>(D.size());
    if (Nx == 0) return;

    std::vector<DenseBlock> Dd(Nx), Ld(Nx), Rd(Nx);
    for (int i = 0; i < Nx; ++i) {
        Dd[i] = to_dense(D[i]);
        if (i > 0)      Ld[i] = to_dense(L_pipe[i]);
        if (i < Nx - 1) Rd[i] = to_dense(R[i]);
    }

    std::vector<VecBlock> Qm = Q;
    X.assign(Nx, VecBlock{});

    std::vector<std::array<int, B>> piv(Nx);
    std::vector<bool> factored(Nx, false);

    for (int i = 1; i < Nx; ++i) {
        int im1 = i - 1;

        if (!factored[im1]) {
            lu_factor(Dd[im1], piv[im1]);
            factored[im1] = true;
        }

        DenseBlock Xtemp, L_X;
        lu_solve_mat(Dd[im1], piv[im1], Rd[im1], Xtemp);
        matmul(Ld[i], Xtemp, L_X);
        subtract_inplace(Dd[i], L_X);

        double y[B], q_prev[B];
        for (int k = 0; k < B; ++k)
            q_prev[k] = Qm[im1][k];

        lu_solve_vec(Dd[im1], piv[im1], q_prev, y);

        double Ly[B];
        matvec(Ld[i], y, Ly);
        for (int k = 0; k < B; ++k)
            Qm[i][k] -= Ly[k];
    }

    if (!factored[Nx - 1]) {
        lu_factor(Dd[Nx - 1], piv[Nx - 1]);
        factored[Nx - 1] = true;
    }

    {
        double rhs[B], sol[B];
        for (int k = 0; k < B; ++k)
            rhs[k] = Qm[Nx - 1][k];
        lu_solve_vec(Dd[Nx - 1], piv[Nx - 1], rhs, sol);
        for (int k = 0; k < B; ++k)
            X[Nx - 1][k] = sol[k];
    }

    for (int i = Nx - 2; i >= 0; --i) {
        if (!factored[i]) {
            lu_factor(Dd[i], piv[i]);
            factored[i] = true;
        }

        double RX[B];
        matvec(Rd[i], X[i + 1].data(), RX);

        double rhs[B], sol[B];
        for (int k = 0; k < B; ++k)
            rhs[k] = Qm[i][k] - RX[k];

        lu_solve_vec(Dd[i], piv[i], rhs, sol);
        for (int k = 0; k < B; ++k)
            X[i][k] = sol[k];
    }
}

// =====================================================
//               DENSE ASSEMBLY (DEBUG)
// =====================================================

std::vector<std::vector<double>> build_dense(
    const std::vector<SparseBlock>& Lb,
    const std::vector<SparseBlock>& Db,
    const std::vector<SparseBlock>& Rb
) {
    const int Nblocks = Db.size();
    const int n = B * Nblocks;

    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));

    auto write_block = [&](int i, int j, const SparseBlock& Bk) {
        int r0 = B * i;
        int c0 = B * j;
        for (size_t k = 0; k < Bk.val.size(); ++k)
            M[r0 + Bk.row[k]][c0 + Bk.col[k]] += Bk.val[k];
    };

    for (int i = 0; i < Nblocks; ++i) {
        write_block(i, i, Db[i]);
        if (i > 0)           write_block(i, i - 1, Lb[i]);
        if (i < Nblocks - 1) write_block(i, i + 1, Rb[i]);
    }

    return M;
}
