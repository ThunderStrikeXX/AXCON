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

bool warnings = false;

#include "libs/steel.h"
#include "libs/liquid_sodium.h"
#include "libs/solid_sodium.h"

// =======================================================================
//
//                          [VARIOUS ALGORITHMS]
//
// =======================================================================

#pragma region solving_functions

constexpr int B = 2;  /// Block dimension

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

// Builds the full dense matrix from the sparse block-tridiagonal representation
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

#pragma region select_case

std::string select_case() {

    std::vector<std::string> cases;

    for (const auto& entry : std::filesystem::directory_iterator(".")) {
        if (entry.is_directory()) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("case_", 0) == 0) cases.push_back(name);
        }
    }

    if (cases.empty()) return "";

    std::cout << "Cases found:\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        std::cout << i << ": " << cases[i] << "\n";
    }

    std::cout << "Press ENTER for a new case. Input the number and press ENTER to load a case: ";

    std::string s;
    std::getline(std::cin, s);

    if (s.empty()) return "";

    int idx = std::stoi(s);
    if (idx < 0 || idx >= static_cast<int>(cases.size())) return "";

    return cases[idx];
}

#pragma endregion

int main() {

    // =======================================================================
    //
    //                        [VARIABLES AND CONSTANTS]
    //
    // =======================================================================

    #pragma region variables_constants

    /// Mathematical constants
    constexpr double M_PI = 3.14159265358979323846;

    /// Simulation parameters
    constexpr int N = 20;                  /// Cell number [-]
    constexpr double L_pipe = 1.0;          /// Length of the pipe domain [m]
    constexpr double dz = L_pipe / N;       /// Spatial step [m]
    constexpr double dt_user = 10;                       /// Temporal step [s]
	constexpr double maxPicard = 100;       /// Maximum Picard iterations
	constexpr double picTolerance = 1e-4;   /// Picard convergence tolerance
	constexpr int nvar = B * N;             /// Total number of variables
    constexpr int nSteps = 1e10;

    /// Geometric parameters
    constexpr double r_o = 0.01335;         /// Outer wall radius [m]
    constexpr double r_i = 0.0112;          /// Wall-wick interface radius [m]
    constexpr double r_v = 0.01075;         /// Vapor-wick interface radius [m]

    /// Heat pipe parameters
    const double evaporator_start = 0.1 * L_pipe;
    const double evaporator_end = 0.2 * L_pipe;
    const double condenser_length = 0.2 * L_pipe;

    ///  Solid sodium region
    const double wick_start = 0.0 * L_pipe;
    const double wick_end = 1.0 * L_pipe;

    /// Geometric sections and perimeters
    constexpr double A_w = M_PI * (r_o * r_o - r_i * r_i);
    constexpr double A_Na = M_PI * (r_i * r_i - r_v * r_v);
    constexpr double P_o = 2 * M_PI * r_o;

    ///  BCs
    constexpr double h_conv = 20.0;             /// Convective heat transfer coefficient [W/m2K]
    constexpr double T_env = 280.0;             /// Environmental temperature [K]
    constexpr double emissivity = 0.8;          /// Emissivity [-]
    constexpr double sigma = 5.67e-8;           /// Stefan-Boltzmann constant [W/m2K4]
    constexpr double power = 100.0;            /// Total power [W]
    constexpr double T_init = 280.0;            /// Initial temperature [K]

    /// Evaporator
    const double Lh = evaporator_end - evaporator_start;
    const double delta_h = 0.01;
    const double Lh_eff = Lh + delta_h;
    const double q0 = power / (2.0 * M_PI * r_o * Lh_eff);      /// [W/m^2]

    /// Condenser
    const double delta_c = 0.05;
    const double condenser_start = L_pipe - condenser_length;
    const double condenser_end = L_pipe;

    static std::vector<double> z(N);

	// New time step variables
    std::vector<double> T_w(N);
    std::vector<double> rho_w(N);
    std::vector<double> cp_w(N);
    std::vector<double> k_w(N);

	std::vector<double> T_Na(N);
    std::vector<double> fl(N);
    std::vector<double> rho_Na(N);
    std::vector<double> cp_Na(N);
    std::vector<double> k_Na(N);
    std::vector<double> H_Na(N);

    int firstNa = -1;
    int lastNa = -1;

    for (int i = 0; i < N; ++i) {
        if (z[i] >= wick_start && z[i] <= wick_end) {
            if (firstNa < 0) firstNa = i;
            lastNa = i;
        }
    }

    for (int i = 0; i < N; ++i) {

        const bool hasNa = (i >= firstNa && i <= lastNa);

        T_w[i] = T_init;
        rho_w[i] = steel::rho(T_init);
        cp_w[i] = steel::cp(T_init);
        k_w[i] = steel::k(T_init);

        T_Na[i] = hasNa ? T_init : 0.0;
        fl[i] = 0.0;
        rho_Na[i] = hasNa ? solid_sodium::rho(T_init) : 0.0;
        cp_Na[i] = hasNa ? solid_sodium::cp(T_init) : 0.0;
        k_Na[i] = hasNa ? solid_sodium::k(T_init) : 0.0;
        H_Na[i] = hasNa ? rho_Na[i] * cp_Na[i] * T_init : 0.0;
    }

    // Old time step variables
    std::vector<double> T_w_old = T_w;
    std::vector<double> rho_w_old = rho_w;
    std::vector<double> cp_w_old = cp_w;
    std::vector<double> k_w_old = k_w;

    std::vector<double> T_Na_old = T_Na;
    std::vector<double> fl_old = fl;
    std::vector<double> rho_Na_old = rho_Na;
    std::vector<double> cp_Na_old = cp_Na;
    std::vector<double> k_Na_old = k_Na;
    std::vector<double> H_Na_old = H_Na;

    // Picard values
    std::vector<double> T_w_iter = T_w;
    std::vector<double> rho_w_iter = rho_w;
    std::vector<double> cp_w_iter = cp_w;
    std::vector<double> k_w_iter = k_w;

    std::vector<double> T_Na_iter = T_Na;
    std::vector<double> fl_iter = fl;
    std::vector<double> rho_Na_iter = rho_Na;
    std::vector<double> cp_Na_iter = cp_Na;
    std::vector<double> k_Na_iter = k_Na;
    std::vector<double> H_Na_iter = H_Na;

    std::vector<double> q_ow(N, 0.0);     /// Outer wall heat flux [W/m2]

    // Blocks definition
    std::vector<SparseBlock> L(N), D(N), R(N);
    std::vector<VecBlock> Q(N), X(N);

    #pragma endregion

    std::string case_chosen = select_case();

    // Create result folder
    int new_case = 0;
    while (true) {
        case_chosen = "case_" + std::to_string(new_case);
        if (!std::filesystem::exists(case_chosen)) {
            std::filesystem::create_directory(case_chosen);
            break;
        }
        new_case++;
    }

    std::ofstream time_output(case_chosen + "/time.txt", std::ios::trunc);
    std::ofstream T_wall_output(case_chosen + "/T_wall.txt", std::ios::trunc);
    std::ofstream T_sodium_output(case_chosen + "/T_sodium.txt", std::ios::trunc);

	/// Cell center positions
    for (int j = 0; j < N; ++j) z[j] = (j + 0.5) * dz;

    std::ofstream mesh_output(case_chosen + "/mesh.txt", std::ios::app);
    mesh_output << std::setprecision(8);

    for (int i = 0; i < N; ++i) mesh_output << i * dz << " ";

    mesh_output.flush();
    mesh_output.close();

	bool all_melted = false;

    int halves = 0;
    double L1 = 0.0;
    double dt;
	double time_total = 0.0;

	// Temporal loop
    for (int n = 0; n < nSteps; ++n) {

        dt = dt_user;
        dt *= std::pow(0.5, halves);

        std::vector<double> A;
        std::vector<double> rhs;

		const double T_solidus = solid_sodium::T_solidus;
        const double T_liquidus = solid_sodium::T_liquidus;
        const double H_lat = solid_sodium::H_lat;

        int pic = 0;        /// Outside to check if convergence is reached

        /// Picard iterations
        for (pic = 0; pic < maxPicard; pic++) {



            // T_iter = T (new)
            T_w_iter = T_w;
            rho_w_iter = rho_w;
            cp_w_iter = cp_w;
            k_w_iter = k_w;

            T_Na_iter = T_Na;
            fl_iter = fl;
            rho_Na_iter = rho_Na;
            cp_Na_iter = cp_Na;
            k_Na_iter = k_Na;
            H_Na_iter = H_Na;

            /// Loop on nodes
            for (int i = 1; i < N - 1; ++i) {

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

                double conv = h_conv * (T_w_iter[i] - T_env);           /// [W/m^2]
                double irr = emissivity * sigma *
                    (std::pow(T_w_iter[i], 4) - std::pow(T_env, 4));    /// [W/m^2]

                if (zi >= condenser_start && zi < condenser_start + delta_c) {
                    double x = (zi - condenser_start) / delta_c;
                    double w = 0.5 * (1.0 - std::cos(M_PI * x));
                    q_ow[i] = -(conv + irr) * w;
                }
                else if (zi >= condenser_start + delta_c) {
                    q_ow[i] = -(conv + irr);
                }
                
                const bool hasNa = (i >= firstNa && i <= lastNa);

                const double k_w = steel::k(T_w_iter[i]);           /// W/mK   
                const double rho_w = steel::rho(T_w_iter[i]);       /// kg/m3
			    const double cp_w = steel::cp(T_w_iter[i]);         /// J/kgK

			    const double C_w = rho_w * cp_w * A_w;      /// J/mK
                const double K_w = k_w * A_w;               /// Wm/K
                const double a_w = K_w / (dz * dz);         /// W/mK

			    const double T_Na = T_Na_iter[i];           /// K
                double a_Na = 0.0;
                double K_Na = 0.0;
                double C_Na = 0.0;
                double C_Na_eff = 0.0;

                double h_ws = 0.0;                          

                if (hasNa) {

                    const double k_Na = k_Na_iter[i];        /// W/mK
				    const double rho_Na = rho_Na_iter[i];    /// kg/m3
                    const double cp_Na = cp_Na_iter[i];      /// J/kgK

                    C_Na = rho_Na * cp_Na * A_Na;      /// J/mK
                    K_Na = k_Na * A_Na;                /// Wm/K
                    a_Na = K_Na / (dz * dz);           /// W/mK

				    const double R_tot = 
                        + std::log(r_o / r_i) / (2 * M_PI * k_w) 
                        + std::log(r_i / r_v) / (2 * M_PI * k_Na);   /// mK/W

				    h_ws = 1.0 / R_tot;  /// W/mK

                    double dfdT = 0.0;
                    if (fl_iter[i] > 0.0 && fl_iter[i] < 1.0)
                        dfdT = 1.0 / (T_liquidus - T_solidus);

                    const double dH_dT = rho_Na * (cp_Na + H_lat * dfdT);

                    C_Na_eff = dH_dT * A_Na;    /// J/mK
                }

                const double D11 = C_w / dt + 2.0 * a_w + h_ws;     /// W/mK
                const double D12 = -h_ws;
                const double D21 = -h_ws;
                const double D22 = (hasNa ? (C_Na_eff / dt + 2.0 * a_Na + h_ws) : 1.0);

                add(D[i], 0, 0, D11);
                add(D[i], 0, 1, D12);
                add(D[i], 1, 0, D21);
                add(D[i], 1, 1, D22);

                add(L[i], 0, 0, -a_w);                                      /// W/mK
                if (hasNa && i > firstNa) add(L[i], 1, 1, -a_Na);           /// W/mK

                add(R[i], 0, 0, -a_w);                                      /// W/mK
                if (hasNa && i < lastNa) add(R[i], 1, 1, -a_Na);            /// W/mK

                Q[i][0] = C_w / dt * T_w_old[i] + q_ow[i] * P_o;            /// W/m
                Q[i][1] = hasNa ? (C_Na_eff / dt * T_Na_old[i]) : 0.0;      /// W/m
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

            solve_block_tridiag(L, D, R, Q, X);

            for (int i = 0; i < N; ++i){

                T_w[i] = X[i][0];
                T_Na[i] = X[i][1];
            }

            // Calculate Picard error
            L1 = 0.0;
            double Aold, Anew, denom, eps;

            for (int i = 0; i < N; ++i) {

                Aold = T_w_iter[i];
                Anew = T_w[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = T_Na_iter[i];
                Anew = T_Na[i];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;
            }

            if (L1 < picTolerance){
                halves = 0;             // Reset halves if Picard converged
                break;                  // Picard converged
            }

            for(int i = 0 ; i < N; ++i) {

				// T_iter = T (new)
                T_w_iter = T_w;
                T_Na_iter = T_Na;

                k_w[i] = steel::k(T_w_iter[i]);
                rho_w[i] = steel::rho(T_w_iter[i]);
                cp_w[i] = steel::cp(T_w_iter[i]);

                if (T_Na_iter[i] <= T_solidus) {

                    fl[i] = 0.0;
                    rho_Na[i] = solid_sodium::rho(T_Na_iter[i]);
                    cp_Na[i] = solid_sodium::cp(T_Na_iter[i]);
                    k_Na[i] = solid_sodium::k(T_Na_iter[i]);

                    H_Na[i] = rho_Na_iter[i] * cp_Na_iter[i] * T_Na_iter[i];

                } else if (T_Na_iter[i] >= T_liquidus) {

                    fl[i] = 1.0;
                    rho_Na[i] = liquid_sodium::rho(T_Na_iter[i]);
                    cp_Na[i] = liquid_sodium::cp(T_Na_iter[i]);
                    k_Na[i] = liquid_sodium::k(T_Na_iter[i]);

                    const double Hs = solid_sodium::rho(T_Na_iter[i]) * solid_sodium::cp(T_Na_iter[i]) * T_solidus;
                    H_Na[i] = Hs + rho_Na_iter[i] * H_lat + rho_Na_iter[i] * cp_Na_iter[i] * (T_Na_iter[i] - T_liquidus);

                } else {

                    // Linear interpolation in mushy region
                    const double w = (T_Na_iter[i] - T_solidus) / (T_liquidus - T_solidus);

                    fl[i] = w;
                    rho_Na[i] = solid_sodium::rho(T_Na_iter[i]) + w * (liquid_sodium::rho(T_Na_iter[i]) - solid_sodium::rho(T_Na_iter[i]));
                    cp_Na[i] = solid_sodium::cp(T_Na_iter[i]) + w * (liquid_sodium::cp(T_Na_iter[i]) - solid_sodium::cp(T_Na_iter[i]));
                    k_Na[i] = solid_sodium::k(T_Na_iter[i]) + w * (liquid_sodium::k(T_Na_iter[i]) - solid_sodium::k(T_Na_iter[i]));

                    const double Hs = solid_sodium::rho(T_Na_iter[i]) * solid_sodium::cp(T_Na_iter[i]) * T_solidus;
                    const double Hl = Hs + liquid_sodium::rho(T_Na_iter[i]) * H_lat;
                    H_Na[i] = Hs + w * (Hl - Hs);
                }
            }
        }

        // Picard converged
        if (pic != maxPicard) {

            // T_old = T_new
            T_w_old = T_w;
            rho_w_old = rho_w;
            cp_w_old = cp_w;
            k_w_old = k_w;

            T_Na_old = T_Na;
            fl_old = fl;
            rho_Na_old = rho_Na;
            cp_Na_old = cp_Na;
            k_Na_old = k_Na;
            H_Na_old = H_Na;

            time_total += dt;
        }
        else {

            // Rollback to previous time step
            // T_new = T_old
            T_w = T_w_old;
            rho_w = rho_w_old;
            cp_w = cp_w_old;
            k_w = k_w_old;

            T_Na = T_Na_old;
            fl = fl_old;
            rho_Na = rho_Na_old;
            cp_Na = cp_Na_old;
            k_Na = k_Na_old;
            H_Na = H_Na_old;

            halves += 1;      // Reduce time step if max Picard iterations reached
        }

		/// Check if all sodium is melted
		all_melted = true;
        for (int i = 0; i < N; ++i) {

            const bool hasNa = (i >= firstNa && i <= lastNa);

            if (hasNa && fl[i] < 0.999)  all_melted = false;
        }

        const int output_every = 10;

        if (n % output_every == 0) {

            for (int i = 0; i < N; ++i) {

                T_wall_output << T_w[i] << " ";
                T_sodium_output << T_Na[i] << " ";
            }

            time_output << time_total << " ";

            T_wall_output << "\n";
            T_sodium_output << "\n";

            time_output.flush();
            T_wall_output.flush();
            T_sodium_output.flush();
        }
    }

    time_output.close();
    T_wall_output.close();
    T_sodium_output.close();
}