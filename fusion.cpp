#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <array>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <cassert>
#include <string>

// =======================================================================
//
//                          [VARIOUS ALGORITHMS]
//
// =======================================================================

#pragma region solving_algorithms
/**
 * @brief Solves a tridiagonal system of linear equations A*x = d using the Thomas Algorithm (TDMA).
 *
 * The method consists of two main phases: forward elimination and back substitution,
 * which is optimized for the sparse tridiagonal structure.
 *
 * @param a The sub-diagonal vector (size N, with a[0] being zero/unused).
 * @param b The main diagonal vector (size N). Must contain non-zero elements.
 * @param c The super-diagonal vector (size N, with c[N-1] being zero/unused).
 * @param d The right-hand side vector (size N).
 * @return std::vector<double> The solution vector 'x' (size N).
 * * @note This implementation assumes the system is diagonally dominant or otherwise
 * stable, as it does not include pivoting. The vectors 'a', 'b', 'c', and 'd' must
 * all have the same size N, corresponding to the size of the system.
 */
std::vector<double> solveTridiagonal(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d) {

    int n = b.size();
    std::vector<double> c_star(n, 0.0);
    std::vector<double> d_star(n, 0.0);
    std::vector<double> x(n, 0.0);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {

        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];

    for (int i = n - 2; i >= 0; i--)
        x[i] = d_star[i] - c_star[i] * x[i + 1];

    return x;
}

#pragma endregion

#pragma region solving_functions

constexpr int B = 2;  // dimensione blocco

// Definition of data structures 
struct SparseBlock {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> val;
};

using DenseBlock = std::array<std::array<double, B>, B>;
using VecBlock = std::array<double, B>;

// ------------------------- Utility dense -------------------------

// Converts a sparse matrix S to a dense matrix M
DenseBlock to_dense(const SparseBlock& S) {
    DenseBlock M{};
    for (std::size_t k = 0; k < S.val.size(); ++k) {
        int i = S.row[k];
        int j = S.col[k];
        M[i][j] = S.val[k];
    }
    return M;
}

// Executes the application of a dense matrix A to a vector x to get a vector y
void matvec(const DenseBlock& A, const double x[B], double y[B]) {
    for (int i = 0; i < B; ++i) {
        double s = 0.0;
        for (int j = 0; j < B; ++j)
            s += A[i][j] * x[j];
        y[i] = s;
    }
}

// Executes the multiplication between a dense matrix A and a dense matrix B to get a dense matrix C
void matmul(const DenseBlock& A, const DenseBlock& Bm, DenseBlock& C) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j) {
            double s = 0.0;
            for (int k = 0; k < B; ++k)
                s += A[i][k] * Bm[k][j];
            C[i][j] = s;
        }
}

// Executes the subtraction of a matrix Bm from a matrix A
void subtract_inplace(DenseBlock& A, const DenseBlock& Bm) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j)
            A[i][j] -= Bm[i][j];
}

// ------------------------- LU with pivoting -------------------------

// In-place LU factorization with partial pivoting, storing L_pipe below and U on/above the diagonal.
void lu_factor(DenseBlock& A, std::array<int, B>& piv) {
    for (int i = 0; i < B; ++i)
        piv[i] = i;

    for (int k = 0; k < B; ++k) {

        // Pivot
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

        // Rows swapping
        if (p != k) {
            std::swap(piv[k], piv[p]);
            for (int j = 0; j < B; ++j)
                std::swap(A[k][j], A[p][j]);
        }

        // Elimination
        for (int i = k + 1; i < B; ++i) {
            A[i][k] /= A[k][k];
            double lik = A[i][k];
            for (int j = k + 1; j < B; ++j)
                A[i][j] -= lik * A[k][j];
        }
    }
}

// Solves Ax = b using the in-place LU factorization (with pivoting) via forward and backward substitution.
void lu_solve_vec(const DenseBlock& LU, const std::array<int, B>& piv,
    const double b_in[B], double x[B]) {

    // Applies pivot to b
    double y[B];
    for (int i = 0; i < B; ++i)
        y[i] = b_in[piv[i]];

    // Ly = Pb (forward)
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < i; ++j)
            y[i] -= LU[i][j] * y[j];
    }

    // Ux = y (backward)
    for (int i = B - 1; i >= 0; --i) {
        for (int j = i + 1; j < B; ++j)
            y[i] -= LU[i][j] * x[j];
        x[i] = y[i] / LU[i][i];
    }
}

// Solves LU·X = P·B column-wise by applying the LU-based vector solver to each column of B.
void lu_solve_mat(const DenseBlock& LU, const std::array<int, B>& piv,
    const DenseBlock& Bm, DenseBlock& X) {

    for (int col = 0; col < B; ++col) {
        double b_col[B];
        double x_col[B];

        for (int i = 0; i < B; ++i)
            b_col[i] = Bm[i][col];

        lu_solve_vec(LU, piv, b_col, x_col);

        for (int i = 0; i < B; ++i)
            X[i][col] = x_col[i];
    }
}

// ------------------------- Solve block-tridiagonal -------------------------


// Block Thomas solver: performs forward elimination and back substitution on a block-tridiagonal system using per-block LU factorizations.
void solve_block_tridiag(
    const std::vector<SparseBlock>& L_pipe,
    const std::vector<SparseBlock>& D,
    const std::vector<SparseBlock>& R,
    const std::vector<VecBlock>& Q,
    std::vector<VecBlock>& X) {
    const int Nx = static_cast<int>(D.size());
    if (Nx == 0)
        return;

    // Dense copy of the blocks
    std::vector<DenseBlock> Dd(Nx);
    std::vector<DenseBlock> Ld(Nx);
    std::vector<DenseBlock> Rd(Nx);

    for (int i = 0; i < Nx; ++i) {
        Dd[i] = to_dense(D[i]);
        if (i > 0)     Ld[i] = to_dense(L_pipe[i]);
        if (i < Nx - 1)  Rd[i] = to_dense(R[i]);
    }

    std::vector<VecBlock> Qm = Q;             // Q changed during forward
    X.assign(Nx, VecBlock{});                 // Solution

    std::vector<std::array<int, B>> piv(Nx);
    std::vector<bool> factored(Nx, false);

    // -------- Forward elimination --------
    for (int i = 1; i < Nx; ++i) {
        int im1 = i - 1;

        if (!factored[im1]) {
            lu_factor(Dd[im1], piv[im1]);
            factored[im1] = true;
        }

        // Solve D[im1] * Xtemp = R[im1]
        DenseBlock Xtemp;
        lu_solve_mat(Dd[im1], piv[im1], Rd[im1], Xtemp);

        // D[i] = D[i] - L_pipe[i] * Xtemp
        DenseBlock L_X;
        matmul(Ld[i], Xtemp, L_X);

        subtract_inplace(Dd[i], L_X);

        // Solve D[im1] * y = Qm[im1]
        double y[B], q_prev[B];
        for (int k = 0; k < B; ++k)
            q_prev[k] = Qm[im1][k];
        lu_solve_vec(Dd[im1], piv[im1], q_prev, y);

        // Qm[i] = Qm[i] - L_pipe[i] * y
        double Ly[B];
        matvec(Ld[i], y, Ly);
        for (int k = 0; k < B; ++k)
            Qm[i][k] -= Ly[k];
    }

    // -------- Backward substitution --------

    // Last block
    if (!factored[Nx - 1]) {
        lu_factor(Dd[Nx - 1], piv[Nx - 1]);
        factored[Nx - 1] = true;
    }
    {
        double rhs[B];
        double sol[B];
        for (int k = 0; k < B; ++k)
            rhs[k] = Qm[Nx - 1][k];
        lu_solve_vec(Dd[Nx - 1], piv[Nx - 1], rhs, sol);
        for (int k = 0; k < B; ++k)
            X[Nx - 1][k] = sol[k];
    }

    // Previous blocks
    for (int i = Nx - 2; i >= 0; --i) {
        if (!factored[i]) {
            lu_factor(Dd[i], piv[i]);
            factored[i] = true;
        }

        double RX[B];
        matvec(Rd[i], X[i + 1].data(), RX);

        double rhs[B];
        for (int k = 0; k < B; ++k)
            rhs[k] = Qm[i][k] - RX[k];

        double sol[B];
        lu_solve_vec(Dd[i], piv[i], rhs, sol);
        for (int k = 0; k < B; ++k)
            X[i][k] = sol[k];
    }
}

// Add triplet to sparse block
auto add = [&](SparseBlock& B, int p, int q, double v) {
    B.row.push_back(p);
    B.col.push_back(q);
    B.val.push_back(v);
    };

std::vector<std::vector<double>> build_dense(
    const std::vector<SparseBlock>& Lb,
    const std::vector<SparseBlock>& Db,
    const std::vector<SparseBlock>& Rb) {
    const int Nblocks = Db.size();
    const int n = 2 * Nblocks;

    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));

    auto write_block = [&](int i, int j, const SparseBlock& B) {
        int row0 = 2 * i;
        int col0 = 2 * j;

        for (size_t k = 0; k < B.val.size(); k++) {
            int r = B.row[k];
            int c = B.col[k];
            M[row0 + r][col0 + c] += B.val[k];
        }
        };

    for (int i = 0; i < Nblocks; i++) {
        write_block(i, i, Db[i]);
        if (i > 0)
            write_block(i, i - 1, Lb[i]);
        if (i < Nblocks - 1)
            write_block(i, i + 1, Rb[i]);
    }

    return M;
}


#pragma endregion

// =======================================================================
//
//                        [VARIABLES AND CONSTANTS]
//
// =======================================================================

#pragma region variables_constants

/// Mathematical constants
constexpr double M_PI = 3.14159265358979323846;

/// Simulation parameters
constexpr int N = 100;                  //// Cell number [-]
constexpr double L_pipe = 1.0;               //// Length of the domain [m]
constexpr double dz = L_pipe/ N;            //// Spatial step [m]
constexpr double dt = 1e-3;             //// Temporal step [s]

/// Geometric parameters
constexpr double r_o = 0.01335;         //// Outer wall radius [m]
constexpr double r_i = 0.0112;          //// Wall-wick interface radius [m]
constexpr double r_v = 0.01075;         //// Vapor-wick interface radius [m]

/// Temperatures and latent heat
constexpr double T_init = 280.0;        //// Initial temperature [K]
constexpr double T_s = 371.0;           //// Solidus Na
constexpr double T_l = 371.0;           //// Liquidus Na
constexpr double L_lat_Na = 1.13e6;     //// J/kg
constexpr double T_solidus = 370.5;     //// K
constexpr double T_liquidus = 371.5;    //// K

///  BCs
const double h_conv = 20.0;          /// Convective heat transfer coefficient [W/m2K]
const double T_env = 280.0;         /// Environmental temperature [K]
const double emissivity = 0.8;     /// Emissivity [-]
const double sigma = 5.67e-8;      /// Stefan-Boltzmann constant [W/m2K4]

/// Sodium properties
constexpr double rho_s_Na = 970.0;      /// kg/m3
constexpr double rho_l_Na = 930.0;      /// kg/m3
constexpr double cp_s_Na = 1200.0;      /// J/kgK
constexpr double cp_l_Na = 1300.0;      /// J/kgK
constexpr double k_s_Na = 140.0;        /// W/mK
constexpr double k_l_Na = 70.0;         /// W/mK

/// Steel properties
constexpr double rho_w = 8000.0;  /// kg/m3
constexpr double cp_w = 500.0;   /// J/kgK
constexpr double k_w = 16.0;    /// W/mK

///  Geometric sections and perimeters
constexpr double A_w = M_PI * (r_o * r_o - r_i * r_i);
constexpr double A_Na = M_PI * (r_i * r_i - r_v * r_v);
constexpr double P_ws = 2 * M_PI * r_i;

/// HTC sodium-wall
constexpr double h_ws = 2000.0;  /// W / m2K (placeholder)

struct Cell {

    double T_w;      /// Wall temperature [K]

    double T_Na;     /// Sodium effective temperature [K]
    double H_Na;     /// Volumetric enthalpy sodium [J/m3]
    double fl;       /// Sodium liquid fraction [0...1]

    double rho_Na;   /// kg/m3
    double cp_Na;    /// J/kgK
    double k_Na;     /// W/mK
};

constexpr double rho_s = 970.0;         /// kg/m3
constexpr double rho_l = 930.0;         /// kg/m3
constexpr double cp_s = 1200.0;         /// J/kgK
constexpr double cp_l = 1300.0;         /// J/kgK
constexpr double k_s = 140.0;           /// W/mK
constexpr double k_l = 70.0;            /// W/mK

///  Heat pipe parameters
const double evaporator_start = 0.1 * L_pipe;
const double evaporator_end = 0.2 * L_pipe;
const double condenser_length = 0.2 * L_pipe;
const double power = 1000.0;          /// Total power [W]

///  Solid sodium region
const double wick_start = 0.1 * L_pipe;
const double wick_end = 0.2 * L_pipe;

/// Evaporator
const double Lh = evaporator_end - evaporator_start;
const double delta_h = 0.01;
const double Lh_eff = Lh + delta_h;
const double q0 = power / (2.0 * M_PI * r_o * Lh_eff);      /// [W/m^2]

/// Condenser
const double delta_c = 0.05;
const double condenser_start = L_pipe - condenser_length;
const double condenser_end = L_pipe;

std::vector<double> a(N), b(N), c(N), d(N);

const double Cw = rho_w * cp_w * A_w * dz;
const double Kw = k_w * A_w;
const double aw = Kw / (dz * dz);

static std::vector<double> z(N);

std::vector<double> q_ow(N, 0.0);     /// Outer wall heat flux [W/m2]
std::vector<double> Q_ow(N, 0.0);     /// Outer wall heat source [W/m3]

std::vector<double> T_w(N);

// Blocks definition
std::vector<SparseBlock> L(N), D(N), R(N);
std::vector<VecBlock> Q(N), X(N);

std::vector<double> T_w_old(N, T_init);
std::vector<double> T_Na_old(N, T_init);

#pragma endregion

// =======================================================================
//
//                           [UPDATING FUNCTIONS]
//
// =======================================================================

#pragma region updating_functions

void update_cell(Cell& C) {

    const double T = C.T_Na;

    if (T <= T_solidus) {
        C.fl = 0.0;
        C.rho_Na = rho_s_Na;
        C.cp_Na = cp_s_Na;
        C.k_Na = k_s_Na;

        C.H_Na = C.rho_Na * C.cp_Na * T;
        return;
    }

    if (T >= T_liquidus) {
        C.fl = 1.0;
        C.rho_Na = rho_l_Na;
        C.cp_Na = cp_l_Na;
        C.k_Na = k_l_Na;

        const double Hs = rho_s_Na * cp_s_Na * T_solidus;
        C.H_Na = Hs + C.rho_Na * L_lat_Na + C.rho_Na * C.cp_Na * (T - T_liquidus);
        return;
    }

    // mushy zone
    const double w = (T - T_solidus) / (T_liquidus - T_solidus);
    C.fl = w;
    C.rho_Na = rho_s_Na + w * (rho_l_Na - rho_s_Na);
    C.cp_Na = cp_s_Na + w * (cp_l_Na - cp_s_Na);
    C.k_Na = k_s_Na + w * (k_l_Na - k_s_Na);

    const double Hs = rho_s_Na * cp_s_Na * T_solidus;
    const double Hl = Hs + rho_l_Na * L_lat_Na;
    C.H_Na = Hs + w * (Hl - Hs);
}

#pragma endregion

std::ofstream mesh_output(name + "/mesh.txt", std::ios::trunc);
std::ofstream time_output(name + "/time.txt", std::ios::trunc);

int main() {

    std::vector<Cell> cell(N);

	/// Cell center positions
    for (int j = 0; j < N; ++j) z[j] = (j + 0.5) * dz;

    int firstNa = -1;
    int lastNa = -1;

    for (int i = 0; i < N; ++i) {
        if (z[i] >= wick_start && z[i] <= wick_end) {
            if (firstNa < 0) firstNa = i;
            lastNa = i;
        }
    }

	/// Initial conditions
    for (int i = 0; i < N; ++i) {

        const bool hasNa = (i >= firstNa && i <= lastNa);

		cell[i].T_w = T_init;
        cell[i].T_Na = hasNa ? T_init : 0.0;
        cell[i].fl = 0.0;

        cell[i].rho_Na = hasNa ? rho_s_Na : 0.0;
        cell[i].cp_Na = hasNa ? cp_s_Na : 0.0;
        cell[i].k_Na = hasNa ? k_s_Na : 0.0;
        cell[i].H_Na = hasNa ? cell[i].rho_Na * cell[i].cp_Na * T_init : 0.0;
    }

	bool all_melted = false;

    while(all_melted == false){

		/// Power distribution along the wall
        for(int i = 0; i < N; ++i) {

            const double zi = z[i];

            if (zi >= (evaporator_start - delta_h) && zi < evaporator_start) {
                double x = (zi - (evaporator_start - delta_h)) / delta_h;
                q_ow[i] = 0.5 * q0 * (1.0 - std::cos(M_PI * x));
            }
            else if (zi >= evaporator_start && zi <= evaporator_end) {
                q_ow[i] = q0;
            }
            else if (zi > evaporator_end && zi <= (evaporator_end + delta_h)) {
                double x = (zi - evaporator_end) / delta_h;
                q_ow[i] = 0.5 * q0 * (1.0 + std::cos(M_PI * x));
            }

            double conv = h_conv * (cell[i].T_w - T_env);           /// [W/m^2]
            double irr = emissivity * sigma *
                (std::pow(cell[i].T_w, 4) - std::pow(T_env, 4));    /// [W/m^2]

            if (zi >= condenser_start && zi < condenser_start + delta_c) {
                double x = (zi - condenser_start) / delta_c;
                double w = 0.5 * (1.0 - std::cos(M_PI * x));
                q_ow[i] = -(conv + irr) * w;
            }
            else if (zi >= condenser_start + delta_c) {
                q_ow[i] = -(conv + irr);
            }
	    }

        std::vector<double> A;
        std::vector<double> rhs;

        const int nvar = B * N;

        for (int i = 1; i < N - 1; ++i) {

            const bool hasNa = (i >= firstNa && i <= lastNa);

            const double aw = Kw / (dz * dz);

            double aNa = 0.0;
            double S = 0.0;
            double CNa_eff = 0.0;

            if (hasNa) {

                const double kNa = cell[i].k_Na;
                aNa = kNa / (dz * dz);

                S = h_ws * P_ws;

                double dfdT = 0.0;
                if (cell[i].fl > 0.0 && cell[i].fl < 1.0)
                    dfdT = 1.0 / (T_liquidus - T_solidus);

                const double dH_dT = cell[i].rho_Na * (cell[i].cp_Na + L_lat_Na * dfdT);

                CNa_eff = dH_dT * A_Na * dz;
            }

            const double D11 = Cw / dt + 2.0 * aw + S;
            const double D12 = -S;
            const double D21 = -S;
            const double D22 = (hasNa ? (CNa_eff / dt + 2.0 * aNa + S) : 1.0);

            add(D[i], 0, 0, D11);
            add(D[i], 0, 1, D12);
            add(D[i], 1, 0, D21);
            add(D[i], 1, 1, D22);

            Q[i][0] = Cw / dt * cell[i].T_w + q_ow[i] * P_ws;
            Q[i][1] = hasNa ? (CNa_eff / dt * cell[i].T_Na) : 0.0;

            add(L[i], 0, 0, -aw);
            if (hasNa && i > firstNa) add(L[i], 1, 1, -aNa);

            add(R[i], 0, 0, -aw);
            if (hasNa && i < lastNa) add(R[i], 1, 1, -aNa);

            DenseBlock L_dense = to_dense(L[i]);
            DenseBlock D_dense = to_dense(D[i]);
            DenseBlock R_dense = to_dense(R[i]);

            printf("");
        }

        // BCs
        add(D[0], 0, 0, 1.0);
        add(R[0], 0, 0, -1.0);
        add(D[0], 1, 1, 1.0);
        Q[0][0] = 0.0;

        add(D[firstNa], 1, 0, 0.0);
        add(R[firstNa], 0, 1, 0.0);
        add(D[firstNa], 1, 1, 1.0);
        add(R[firstNa], 1, 1, -1.0);
        Q[firstNa][1] = 0.0;

        add(D[N - 1], 0, 0, 1.0);
        add(L[N - 1], 0, 0, -1.0);
        add(D[N - 1], 1, 1, 1.0);
        Q[N - 1][0] = 0.0;

        add(D[lastNa], 1, 0, 0.0);
        add(L[lastNa], 0, 1, 0.0);
        add(D[lastNa], 1, 1, 1.0);
        add(L[lastNa], 1, 1, -1.0);
        Q[lastNa][1] = 0.0;

        DenseBlock L_dense_f = to_dense(L[firstNa]);
        DenseBlock D_dense_f = to_dense(D[firstNa]);
        DenseBlock R_dense_f = to_dense(R[firstNa]);

        DenseBlock L_dense_l = to_dense(L[lastNa]);
        DenseBlock D_dense_l = to_dense(D[lastNa]);
        DenseBlock R_dense_l = to_dense(R[lastNa]);

		std::vector<std::vector<double>> dense = build_dense(L, D, R);

        solve_block_tridiag(L, D, R, Q, X);

        for (int i = 0; i < N; ++i){

            cell[i].T_w = X[i][0];
            cell[i].T_Na = X[i][1];

            update_cell(cell[i]);
        }

		/// Check if all sodium is melted
		all_melted = true;
        for (int i = 0; i < N; ++i) {

            const bool hasNa = (i >= firstNa && i <= lastNa);

            if (hasNa && cell[i].fl < 0.999)  all_melted = false;
        }
    }

    std::vector<double> T_wall(N), T_liq(N);

    for (int i = 0; i < N; ++i) {
        T_wall[i] = cell[i].T_w;
        T_liq[i] = cell[i].T_Na;
    }

    /// In  CFD: T_w(z)=T_wall, T_l(z)=T_liq, T_v(z)=T_liq, p(z)=p_sat(T_liq), u=0
}