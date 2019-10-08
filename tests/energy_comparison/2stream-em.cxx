#include "src/input/deck.h"
// TODO: reaching into this path is a bit odd..
#include "tests/energy_comparison/compare_energies.h"

class Custom_Finalizer : public Run_Finalizer {
    public:
        using real_ = real_t;

        // This *has* to be virtual, as we store the object as a pointer to the
        // base class
        virtual void finalize()
        {
           // Try and validate the final answers

            std::string energy_file_name = "energies";
            //std::string energy_gold_file_name = EXPAND_AND_STRINGIFY( GOLD_ENERGY_FILE );
            std::string energy_gold_file_name = "energies_gold";

            // TODO: VPIC has a newer verson of compare_energies I should pull over
            // TODO: add propper calls to test functions REQUIRE

            // Mask which fields to sum
            const unsigned short e_mask = 0b0000001110;
            test_utils::compare_energies(energy_file_name, energy_gold_file_name,
                    0.3, e_mask, test_utils::FIELD_ENUM::Sum, 1, "Weibel.e.out");
        }
};

Input_Deck::Input_Deck()
{
    // User puts initialization code here
    // Example: EM 2 Stream in 1d?

    run_finalizer = new Custom_Finalizer();

    nx = 1;
    ny = 32;
    nz = 1;

    num_steps = 3000;
    nppc = 100;

    v0 = 0.2;

    // Can also create temporaries
    real_ gam = 1.0 / sqrt(1.0 - v0*v0);

    const real_ default_grid_len = 1.0;

    len_x_global = default_grid_len;
    len_y_global = 3.14159265358979*0.5; // TODO: use proper PI?
    len_z_global = default_grid_len;

    dt = 0.99*courant_length(
            len_x_global, len_y_global, len_z_global,
            nx, ny, nz
            ) / c;

    n0 = 2.0; //for 2stream, for 2 species, making sure omega_p of each species is 1
}
