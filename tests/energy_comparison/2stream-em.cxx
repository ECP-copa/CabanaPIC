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

#ifndef GOLD_ENERGY_FILE
            std::cerr << "Cannot find energy gold file, exiting" << std::endl;
            std::exit(EXIT_FAILURE);
#endif

#ifdef CUSTOM_ERROR_MARGIN
            double error_margin = CUSTOM_ERROR_MARGIN
#else
            double error_margin = 0.10; // 10%
            // TODO: add constexpr if for if real_T is double to decrease the toll
#endif

            // TODO: we might need to clear out local energy file first for
            // this to be sensible? Else we could end up reading old stale
            // files locally

            std::string energy_file_name = "energies.txt";
            std::string energy_gold_file_name = EXPAND( GOLD_ENERGY_FILE );

            // TODO: port this to a testing framework instead of relying on
            // error codes?

            // We want to measure 18-50 in science time, so 371 to 1030 in
            // timestep for the given configuration

            // This does 2 passes through the file, but it's OK for now..

            // Test steps 3581..5081 as it covers approx 60..100 sim time

            // Mask which fields to sum, read only 3rd val
            const unsigned short e_mask = 0b0000000100;
            bool e_correct = test_utils::compare_energies(
                    energy_gold_file_name,
                    energy_file_name,
                    error_margin,  // margin for error
                    e_mask,
                    test_utils::FIELD_ENUM::Sum,
                    1,  // if should diagnostic out
                    "e.out", // diagnostic output file
                    3581, //371 // num to skip, reads 3581
                    1300 // check 1500 lines
            );
            std::cout << "E Test Pass: " << e_correct << std::endl;

            const unsigned short b_mask = 0b0000001000;
            bool b_correct = test_utils::compare_energies(
                    energy_gold_file_name,
                    energy_file_name,
                    error_margin,  // margin for error
                    b_mask,
                    test_utils::FIELD_ENUM::Sum,
                    1,  // if should diagnostic out
                    "b.out", // diagnostic output file
                    3581, //371 // num to skip, reads 3581 to EOF
                    1300 // check 1500 lines
            );
            std::cout << "B Test Pass: " << b_correct << std::endl;

            // Throw error code if either failed
            if ((!b_correct) || (!e_correct)) {
                std::exit(1);
            }
        }
};

// This relies on the default particle init, changing that will break this..
Input_Deck::Input_Deck()
{
    // User puts initialization code here
    // Example: EM 2 Stream in 1d?

    run_finalizer = new Custom_Finalizer();

    nx = 1;
    ny = 32;
    nz = 1;

    num_steps = 6000;
    nppc = 100;

    //v0 = 0.2;
    v0 = 0.0866025403784439;

    // Can also create temporaries
    real_ gam = 1.0 / sqrt(1.0 - v0*v0);

    const real_ default_grid_len = 1.0;

    len_x_global = default_grid_len;
    //len_y_global = 3.14159265358979*0.5; // TODO: use proper PI?
    len_y_global = 0.628318530717959*(gam*sqrt(gam));
    len_z_global = default_grid_len;

    dt = 0.99*courant_length(
            len_x_global, len_y_global, len_z_global,
            nx, ny, nz
            ) / c;

    n0 = 2.0; //for 2stream, for 2 species, making sure omega_p of each species is 1
}
