#include <Cabana_AoSoA.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Sort.hpp> // is this needed if we already have core?

#include <cstdlib>
#include <iostream>

#include "types.h"
#include "helpers.h"
#include "simulation_parameters.h"

#include "initializer.h"

#include "fields.h"
#include "accumulator.h"
#include "interpolators.h"

#include "push.h"

#include "visualization.h"

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Cabana::initialize( argc, argv );

    // Cabana scoping block
    {

    Visualizer vis;

    // Initialize input deck params.

    // num_cells (without ghosts), num_particles_per_cell
    Initializer::initialize_params(4, 2);

    // Cache some values locally for printing
    const size_t num_cells = Parameters::instance().num_cells;
    const size_t num_particles = Parameters::instance().num_particles;

    // Create the particle list.
    particle_list_t particles( num_particles );
    logger << "size " << particles.size() << std::endl;
    logger << "numSoA " << particles.numSoA() << std::endl;

    // Initialize particles.
    Initializer::initialize_particles( particles );

    grid_t* grid = new grid_t();

    // Define some consts
    real_t qdt_2mc = 1.0f;
    real_t cdt_dx = 1.0f;
    real_t cdt_dy = 1.0f;
    real_t cdt_dz = 1.0f;
    real_t qsp = 1.0f;

    // Print initial particle positions
    logger << "Initial:" << std::endl;
    print_particles( particles );

    // Allocate Cabana Data
    interpolator_array_t interpolators(num_cells);
    accumulator_array_t accumulators(num_cells); // TODO: this should become a kokkos scatter add
    field_array_t fields(num_cells);

    Initializer::initialize_interpolator(interpolators);

    // Can obviously supply solver type at compile time
    //Field_Solver<EM_Field_Solver> field_solver;
    Field_Solver<ES_Field_Solver> field_solver;

    // Grab some global values for use later
    const size_t nx = Parameters::instance().nx;
    const size_t ny = Parameters::instance().ny;
    const size_t nz = Parameters::instance().nz;

    logger << "nx " << Parameters::instance().nx << std::endl;
    logger << "num_particles " << num_particles << std::endl;
    logger << "num_cells " << num_cells << std::endl;
    logger << "Actual NPPC " << Parameters::instance().NPPC << std::endl;

    // TODO: give these a real value
    const real_t px = 0.9; // (nx>1) ? frac*g->cvac*g->dt*g->rdx : 0;
    const real_t py = 0.9; // (ny>1) ? frac*g->cvac*g->dt*g->rdy : 0;
    const real_t pz = 0.9; // (nz>1) ? frac*g->cvac*g->dt*g->rdz : 0;

    // simulation loop
    const size_t num_steps = Parameters::instance().num_steps;
    for (size_t step = 0; step < num_steps; step++)
    {
        std::cout << "Step " << step << std::endl;

        // Convert fields to interpolators
        load_interpolator_array(fields, interpolators, nx, ny, nz);

        // TODO: Make the frequency of this configurable (every step is not
        // required for this incarnation)
        // Sort by cell index
        auto keys = particles.slice<Cell_Index>();
        auto bin_data = Cabana::sortByKey( keys );

        // Move
        push(
            particles,
            interpolators,
            qdt_2mc,
            cdt_dx,
            cdt_dy,
            cdt_dz,
            qsp,
            accumulators,
            grid
        );

        // TODO: boundaries?
        // boundary_p(); // Implies Parallel?

        // Map accumulator current back onto the fields
        unload_accumulator_array(fields, accumulators, nx, ny, nz);

        // Half advance the magnetic field from B_0 to B_{1/2}
        field_solver.advance_b(fields, px, py, pz, nx, ny, nz);

        // Advance the electric field from E_0 to E_1
        field_solver.advance_e(fields, px, py, pz, nx, ny, nz);

        // Half advance the magnetic field from B_{1/2} to B_1
        field_solver.advance_b(fields, px, py, pz, nx, ny, nz);

        // Print particles.
        print_particles( particles );

        // Output vis
        vis.write_vis(particles, step);

    }
    } // End Scoping block

    // Finalize.
    Cabana::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
//

////// Known Possible Improvements /////
// I pass nx/ny/nz round a lot more than I could

