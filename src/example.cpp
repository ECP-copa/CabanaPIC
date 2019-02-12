#include <Cabana_AoSoA.hpp>
#include <Cabana_Core.hpp>

#include <cstdlib>
#include <iostream>

#include "types.h"
#include "helpers.h"
#include "simulation_parameters.h"
#include "initializer.h"
#include "visualization.h"
#include "push.h"

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Cabana::initialize( argc, argv );

    // Cabana scoping block
    {

    // Initialize input deck params.
    Initializer::initialize_params(2, 1);
    const size_t num_steps = Parameters::instance().num_steps;
    const size_t num_cells = Parameters::instance().num_cells;
    const size_t num_particles = Parameters::instance().num_particles;

    logger << "nx " << Parameters::instance().nx << std::endl;
    logger << "num_particles " << num_particles << std::endl;
    logger << "num_cells " << num_cells << std::endl;

    // Create the particle list.
    particle_list_t particles( num_particles );
    logger << "numSoA particles.numSoA() << std::endl;

    // Initialize particles.
    Initializer::initialize_particles( particles );

    grid_t* g = new grid();

    // Define some consts
    real_t qdt_2mc = 1.0f;
    real_t cdt_dx = 1.0f;
    real_t cdt_dy = 1.0f;
    real_t cdt_dz = 1.0f;
    real_t qsp = 1.0f;

    // Print initial particle positions
    loggerer << "Initial:" << std::endl;
    print_particles( particles );
    loggerer << std::endl;

    // OLD WAY TO CREATE DATA
    //interpolator_array_t* f = new interpolator_array_t(num_cells);
    //accumulator_array_t* a = new accumulator_array_t(num_cells);

    // NEW CABANA STYLE
    interpolator_array_t f(num_cells);
    accumulator_array_t a(num_cells);

    Initializer::initialize_interpolator(f);

    Visualizer vis;

    for (size_t step = 0; step < num_steps; step++)
    {
        std::cout << "Step " << step << std::endl;

        // Sort TODO
        // sort_particles();

        // Move
        push(
            particles,
            f,
            qdt_2mc,
            cdt_dx,
            cdt_dy,
            cdt_dz,
            qsp,
            a,
            g
        );

        // boundary_p TODO
        // boundary_p(); // Implies Parallel!

        // unload_accumulator_array TODO

        // Half advance the magnetic field from B_0 to B_{1/2}
        // advance_b(); TODO

        // Advance the electric field from E_0 to E_1
        // advance_e(); TODO

        // Half advance the magnetic field from B_{1/2} to B_1
        // advance_b(); TODO

        // Print particles.
        print_particles( particles );
        std::cout << std::endl;

        // Output vis
        vis.write_vis(particles, step);


    }
    } // End Scoping block

    // Finalize.
    Cabana::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
