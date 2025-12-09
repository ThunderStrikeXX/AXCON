/**
 * @brief Provides thermophysical properties for Solid Sodium (Na).
 *
 * This namespace contains constant data and functions to calculate key
 * temperature-dependent properties of solid sodium.
 * All functions accept temperature T in Kelvin [K] and return values
 * in standard SI units unless otherwise specified.
 * The functions give warnings if the input temperature is above the
 * (constant) melting temperature.
 */
namespace solid_sodium {

    /// Melting temperatures
    constexpr double T_solidus = 370.5;     /// Solidus temperature [K]
    constexpr double T_liquidus = 371.5;    /// Liquidus temperature [K]
    constexpr double H_lat = 113e3;         /// Latent heat [J/(kg)]

    /**
    * @brief Density [kg/m3] as a function of temperature T
    *   Keenan–Keyes / Vargaftik
    */
    inline double rho(double T) {

        if (T > T_liquidus && warnings == true) std::cout << "Warning, temperature " << T << " is above melting temperature!";
        return 972.70 - 0.2154 * (T - 273.15);
    }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T
    *   Vargaftik
    */
    inline double k(double T) {

        if (T > T_liquidus && warnings == true) std::cout << "Warning, temperature " << T << " is above melting temperature!";
        return 135.6 - 0.167 * (T - 273.15);
    }

    /**
    * @brief Specific heat at constant pressure [J/(kg·K)] as a function of temperature
    *   Vargaftik / Fink & Leibowitz
    */
    inline double cp(double T) {

        if (T > T_liquidus && warnings == true) std::cout << "Warning, temperature " << T << " is above melting temperature!";
        return 1199 + 0.649 * (T - 273.15) + 1052.9e-5 * (T - 273.15) * (T - 273.15);
    }
}