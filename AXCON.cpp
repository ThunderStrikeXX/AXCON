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

#include "solver.h"
#include "steel.h"
#include "liquid_sodium.h"
#include "solid_sodium.h"
#include "vapor_sodium.h"

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

    // Mathematical constants
    constexpr double M_PI = 3.14159265358979323846;

    // Simulation parameters
    constexpr int N = 20;                   // Cell number [-]
    constexpr double L_pipe = 1.0;          // Length of the pipe domain [m]
    constexpr double dz = L_pipe / N;       // Spatial step [m]
    constexpr double dt_user = 1e-1;        // Temporal step [s]
	constexpr double max_picard = 100;      // Maximum Picard iterations
	constexpr double pic_tolerance = 1e-4;  // Picard convergence tolerance
	constexpr int nvar = B * N;             // Total number of variables
    constexpr int tot_iter = 1e6;

    int halves = 0;                         // Number of times the time step has been halved
    double L1 = 0.0;                        // L1 error for picard convergence
    double dt = dt_user;                              // Actual time step
    double time_total = 0.0;                // Total time elapsed
    bool all_melted = false;                // Flag to indicate if all sodium has melted
    int pic = 0;

    // Geometric parameters
    constexpr double r_o = 0.01335;         // Outer wall radius [m]
    constexpr double r_i = 0.0112;          // Wall-wick interface radius [m]
    constexpr double r_v = 0.01075;         // Vapor-wick interface radius [m]

	// Vapor core variables (0D model)
    constexpr double M_Na = 0.02298977;                  // [kg/mol]
    constexpr double R_univ = 8.314462618;               // [J/mol/K]
    constexpr double R_Na = R_univ / M_Na;               // [J/kg/K]

    constexpr double dP_on = 2.0;   // [K] isteresi evaporazione
    constexpr double dP_off = 2.0;   // [K] isteresi condensazione
    constexpr double f_conn = 0.9;   // soglia liquido "connesso"

    constexpr double V_vapor = M_PI * r_v * r_v * L_pipe;   // [m3] (volume core)
    constexpr double p_min = 1.0;                           // [Pa] quasi-vuoto iniziale

    // Heat pipe parameters
    const double evaporator_start = 0.1 * L_pipe;
    const double evaporator_end = 0.2 * L_pipe;
    const double condenser_length = 0.2 * L_pipe;

    //  Solid sodium region
    const double wick_start = 0.0 * L_pipe;
    const double wick_end = 1.0 * L_pipe;

    // Geometric sections and perimeters
    constexpr double A_w = M_PI * (r_o * r_o - r_i * r_i);
    constexpr double A_Na = M_PI * (r_i * r_i - r_v * r_v);
    constexpr double P_o = 2 * M_PI * r_o;

    //  BCs
    constexpr double h_conv = 20.0;             // Convective heat transfer coefficient [W/m2K]
    constexpr double T_env = 280.0;             // Environmental temperature [K]
    constexpr double emissivity = 0.8;          // Emissivity [-]
    constexpr double sigma = 5.67e-8;           // Stefan-Boltzmann constant [W/m2K4]
    constexpr double power = 100.0;            // Total power [W]
    constexpr double T_init = 280.0;            // Initial temperature [K]

    double p_v = p_min;                                  // pressione vapore [Pa]
    double m_v = p_v * V_vapor / (R_Na * T_init);        // massa vapore [kg]
    double T_v = T_init;                                 // temperatura vapore [K]

    // Evaporator
    const double Lh = evaporator_end - evaporator_start;
    const double delta_h = 0.01;
    const double Lh_eff = Lh + delta_h;
    const double q0 = power / (2.0 * M_PI * r_o * Lh_eff);      // [W/m^2]

    // Condenser
    const double delta_c = 0.05;
    const double condenser_start = L_pipe - condenser_length;
    const double condenser_end = L_pipe;

    static std::vector<double> z(N);
    for (int j = 0; j < N; ++j) z[j] = (j + 0.5) * dz;

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

	std::vector<double> P_sat(N, vapor_sodium::P_sat(T_v));

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

    std::vector<double> q_ow(N, 0.0);     // Outer wall heat flux [W/m2]

    // Blocks definition
    std::vector<SparseBlock> L(N), D(N), R(N);
    std::vector<VecBlock> Q(N), X(N);

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

    std::ofstream mesh_output(case_chosen + "/mesh.txt", std::ios::app);
    std::ofstream time_output(case_chosen + "/time.txt", std::ios::trunc);
    std::ofstream T_wall_output(case_chosen + "/T_wall.txt", std::ios::trunc);
    std::ofstream T_sodium_output(case_chosen + "/T_sodium.txt", std::ios::trunc);
    std::ofstream f_sodium_output(case_chosen + "/f_sodium.txt", std::ios::trunc);
	std::ofstream p_sat_output(case_chosen + "/p_sat.txt", std::ios::trunc);

    mesh_output << std::setprecision(8);

    // Cell center positions
    for (int i = 0; i < N; ++i) mesh_output << i * dz << " ";

    mesh_output.flush();
    mesh_output.close();

    double h_ws = 0.0;

    double m_dot_ev = 0.0;   // [kg/s]
    double m_dot_co = 0.0;   // [kg/s]

    #pragma endregion

	// Temporal loop
    for (int n = 0; n < tot_iter; ++n) {

        dt = dt_user;
        dt *= std::pow(0.5, halves);

        std::vector<double> A;
        std::vector<double> rhs;

		const double T_solidus = solid_sodium::T_solidus;
        const double T_liquidus = solid_sodium::T_liquidus;
        const double H_lat = solid_sodium::H_lat;

        // Picard iterations
        for (pic = 0; pic < max_picard; pic++) {

            // Cleaning blocks
            for (int i = 0; i < N; i++) {
                L[i].row.clear(); L[i].col.clear(); L[i].val.clear();
                D[i].row.clear(); D[i].col.clear(); D[i].val.clear();
                R[i].row.clear(); R[i].col.clear(); R[i].val.clear();
            }

            // Iter variables = new variables
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

            // Loop on nodes
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

                double conv = h_conv * (T_w_iter[i] - T_env);           // [W/m^2]
                double irr = emissivity * sigma *
                    (std::pow(T_w_iter[i], 4) - std::pow(T_env, 4));    // [W/m^2]

                if (zi >= condenser_start && zi < condenser_start + delta_c) {
                    double x = (zi - condenser_start) / delta_c;
                    double w = 0.5 * (1.0 - std::cos(M_PI * x));
                    q_ow[i] = -(conv + irr) * w;
                }
                else if (zi >= condenser_start + delta_c) {
                    q_ow[i] = -(conv + irr);
                }
                
                const bool hasNa = (i >= firstNa && i <= lastNa);

                const double k_w = steel::k(T_w_iter[i]);           // [W/mK]   
                const double rho_w = steel::rho(T_w_iter[i]);       // [kg/m3]
			    const double cp_w = steel::cp(T_w_iter[i]);         // [J/kgK]

			    const double C_w = rho_w * cp_w * A_w;              // [J/mK]
                const double K_w = k_w * A_w;                       // [Wm/K]
                const double a_w = K_w / (dz * dz);                 // [W/mK]

			    const double T_Na = T_Na_iter[i];                   // [K]
                double a_Na = 0.0;
                double K_Na = 0.0;
                double C_Na = 0.0;
                double C_Na_eff = 0.0;

                h_ws = 0.0;   
				P_sat[i] = vapor_sodium::P_sat(T_Na_iter[i]);

                if (hasNa) {

                    const double k_Na = k_Na_iter[i];               // [W/mK]
				    const double rho_Na = rho_Na_iter[i];           // [kg/m3]
                    const double cp_Na = cp_Na_iter[i];             // [J/kgK]

                    C_Na = rho_Na * cp_Na * A_Na;                   // [J/mK]
                    K_Na = k_Na * A_Na;                             // [Wm/K]
                    a_Na = K_Na / (dz * dz);                        // [W/mK]

				    const double R_tot = 
                        + std::log(r_o / r_i) / (2 * M_PI * k_w) 
                        + std::log(r_i / r_v) / (2 * M_PI * k_Na);  // [mK/W]

				    h_ws = 1.0 / R_tot;                             // [W/mK]

                    double dfdT = 0.0;
                    if (fl_iter[i] > 0.0 && fl_iter[i] < 1.0)
                        dfdT = 1.0 / (T_liquidus - T_solidus);

                    const double dH_dT = rho_Na * (cp_Na + H_lat * dfdT);

                    C_Na_eff = dH_dT * A_Na;                        // [J/mK]
                }

                const double D11 = C_w / dt + 2.0 * a_w + h_ws;     // [W/mK]
                const double D12 = -h_ws;
                const double D21 = -h_ws;
                const double D22 = (hasNa ? (C_Na_eff / dt + 2.0 * a_Na + h_ws) : 1.0);

                add(D[i], 0, 0, D11);
                add(D[i], 0, 1, D12);
                add(D[i], 1, 0, D21);
                add(D[i], 1, 1, D22);

                add(L[i], 0, 0, -a_w);                                      // W/mK
                if (hasNa && i > firstNa) add(L[i], 1, 1, -a_Na);           // W/mK

                add(R[i], 0, 0, -a_w);                                      // W/mK
                if (hasNa && i < lastNa) add(R[i], 1, 1, -a_Na);            // W/mK

                Q[i][0] = C_w / dt * T_w_old[i] + q_ow[i] * P_o;            // W/m
                Q[i][1] = hasNa ? (C_Na_eff / dt * T_Na_old[i]) : 0.0;      // W/m

                double Q_phase = 0.0;

                // se evapora localmente
                if (fl[i] > f_conn && p_v > P_sat[i] + dP_on) {
                    Q_phase -= h_ws * (T_w_iter[i] - T_Na_iter[i]); // sink
                }

                // se condensa
                if (fl[i] > f_conn && p_v < P_sat[i] - dP_off) {
                    Q_phase += h_ws * (T_w_iter[i] - T_Na_iter[i]); // source
                }

                // aggiunta alla RHS sodio
                Q[i][1] += Q_phase * dz;
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

            // Update Picard values (iter = new)
            for(int i = 0 ; i < N; ++i) {

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

            m_dot_ev = 0.0;   // [kg/s]
            m_dot_co = 0.0;   // [kg/s]

            // Temperatura di riferimento vapore (media evaporatore)
            double T_ref = 0.0;
            int nref = 0;
            for (int i = 0; i < N; ++i) {
                if (z[i] >= evaporator_start && z[i] <= evaporator_end && fl[i] > f_conn) {
                    T_ref += T_Na[i];
                    nref++;
                }
            }

            if (nref > 0) T_ref /= nref;
            else T_ref = T_v;

            // Saturazione
            const double p_sat = vapor_sodium::P_sat(T_ref);

            // Loop celle: calcolo flussi di massa
            for (int i = 0; i < N; ++i) {

                if (fl[i] < f_conn) continue; // niente liquido, niente fase

                const double T_int = T_Na[i];
                const double h_fg = vapor_sodium::h_vap_sodium(T_int);

                // Potenza disponibile parete->sodio (per cella)
                const double Q_ws = h_ws * (T_w[i] - T_Na[i]) * dz; // [W]

                // Evaporazione
                if (p_v > p_sat + dP_on && Q_ws > 0.0) {
                    const double md = Q_ws / h_fg;
                    m_dot_ev += md;
                }

                // Condensazione
                if (p_v < p_sat - dP_off && Q_ws < 0.0) {
                    const double md = (-Q_ws) / h_fg;
                    m_dot_co += md;
                }
            }

            // Aggiornamento massa vapore
            m_v += dt * (m_dot_ev - m_dot_co);
            m_v = std::max(m_v, 0.0);

            // Aggiorna temperatura vapore (scelta semplice)
            T_v = T_ref;

            // EOS gas ideale
            p_v = std::max(p_min, m_v * R_Na * T_v / V_vapor);

            if (L1 < pic_tolerance) {
                halves = 0;             // Reset halves if Picard converged
                break;                  // Picard converged
            }
        }

        // Picard converged or max iterations reached
        if (pic != max_picard) {

            // Update n values (old = new)
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

            bool HP_active =
                (m_dot_ev > 0.0) &&
                (m_dot_co > 0.0) &&
                (p_v > p_min * 10.0);

            time_total += dt;

        } else {

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

            halves += 1;        // Reduce time step if max Picard iterations reached
            n -= 1;             // Repeat time step to maintain output frequency
        }

        /*
		// Check if all sodium is melted
		all_melted = true;
        for (int i = 0; i < N; ++i) {

            const bool hasNa = (i >= firstNa && i <= lastNa);

            if (hasNa && fl[i] < 0.999)  all_melted = false;
        }
        */

        const int output_every = 10;

        if (n % output_every == 0) {

            for (int i = 0; i < N; ++i) {

                T_wall_output << T_w[i] << " ";
                T_sodium_output << T_Na[i] << " ";
				f_sodium_output << fl[i] << " ";
            }

            time_output << time_total << " ";
            p_sat_output << p_v << " ";

            T_wall_output << "\n";
            T_sodium_output << "\n";
			f_sodium_output << "\n";

            time_output.flush();

            T_wall_output.flush();
            T_sodium_output.flush();
			f_sodium_output.flush();
        }
    }

    time_output.close();
    T_wall_output.close();
    T_sodium_output.close();
	f_sodium_output.close();

    return 0;
}