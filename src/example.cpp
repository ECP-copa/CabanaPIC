#include <Cabana_AoSoA.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Sort.hpp> // is this needed if we already have core?

#include <cstdlib>
#include <iostream>

#include "types.h"
#include "helpers.h"

#include "fields.h"
#include "accumulator.h"
#include "interpolator.h"

#include "push.h"

#include "visualization.h"

#include "input/deck.h"

// Global variable to hold paramters
//Parameters params;
Input_Deck deck;

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Cabana::initialize( argc, argv );

    printf ("#On Kokkos execution space %s\n",
            typeid (Kokkos::DefaultExecutionSpace).name ());
    // Cabana scoping block
    {

        //Visualizer vis;

        // Initialize input deck params.

        // num_cells (without ghosts), num_particles_per_cell
        size_t npc = 100;

        deck.derive_params();
        deck.print_run_details();

        // Cache some values locally for printing
        const int nx = deck.nx;
        const int ny = deck.ny;
        const int nz = deck.nz;
        const int num_ghosts = deck.num_ghosts;
        const size_t num_cells = deck.num_cells;
        const size_t num_particles = deck.num_particles;
        real_t dxp = 2.f/(npc);

        // Define some consts
        const real_t dx = deck.dx;
        const real_t dy = deck.dy;
        const real_t dz = deck.dz;
        real_t dt   = deck.dt;
        real_t c    = deck.c;
        real_t me   = deck.me;
        real_t n0   = deck.n0;
        real_t ec   = deck.ec;
        real_t Lx   = deck.len_x;
        real_t Ly   = deck.len_y;
        real_t Lz   = deck.len_z;
        int nppc = deck.nppc;
        real_t eps0 = deck.eps;
        real_t Npe  = n0*Lx*Ly*Lz;
        size_t Ne=  (nppc*nx*ny*nz);
        real_t qsp = -ec;
        real_t qdt_2mc = qsp*dt/(2*me*c);

        real_t cdt_dx = c*dt/dx;
        real_t cdt_dy = c*dt/dy;
        real_t cdt_dz = c*dt/dz;
        real_t dt_eps0 = dt/eps0;
        real_t frac = 1.0f;
        real_t we = (real_t) Npe/(real_t) Ne;

        printf("c %e dt %e dx %e cdt_dx %e \n", c, dt,dx,cdt_dx);

        // Create the particle list.
        particle_list_t particles( num_particles );
        //logger << "size " << particles.size() << std::endl;
        //logger << "numSoA " << particles.numSoA() << std::endl;

        // Initialize particles.
        deck.initialize_particles( particles, nx, ny, nz, dxp, npc, we );

        grid_t* grid = new grid_t();

        // Print initial particle positions
        //logger << "Initial:" << std::endl;
        //print_particles( particles );

        // Allocate Cabana Data
        interpolator_array_t interpolators(num_cells);

        accumulator_array_t accumulators("Accumulator View", num_cells);

        auto scatter_add = Kokkos::Experimental::create_scatter_view(accumulators);
        //<Kokkos::Experimental::ScatterSum,
        //KOKKOS_SCATTER_DUPLICATED,
        //KOKKOS_SCATTER_ATOMIC>(accumulators);

        field_array_t fields(num_cells);

        // Zero out the interpolator
        // Techincally this is optional?
        initialize_interpolator(interpolators);

        // Can obviously supply solver type at compile time
        Field_Solver<EM_Field_Solver> field_solver(fields);
        //Field_Solver<ES_Field_Solver_1D> field_solver(fields);

        // Grab some global values for use later
        const Boundary boundary = deck.BOUNDARY_TYPE;

        //logger << "nx " << params.nx << std::endl;
        //logger << "num_particles " << num_particles << std::endl;
        //logger << "num_cells " << num_cells << std::endl;
        //logger << "Actual NPPC " << params.NPPC << std::endl;

        // TODO: give these a real value
        const real_t px =  (nx>1) ? frac*c*dt/dx : 0;
        const real_t py =  (ny>1) ? frac*c*dt/dy : 0;
        const real_t pz =  (nz>1) ? frac*c*dt/dz : 0;

        // simulation loop
        const size_t num_steps = deck.num_steps;

        printf( "#***********************************************\n" );
        printf( "#num_step = %ld\n" , num_steps );
        printf( "#Lx/de = %f\n" , Lx );
        printf( "#Ly/de = %f\n" , Ly );
        printf( "#Lz/de = %f\n" , Lz );
        printf( "#nx = %d\n" , nx );
        printf( "#ny = %d\n" , ny );
        printf( "#nz = %d\n" , nz );
        printf( "#nppc = %ld\n" , nppc );
        printf( "# Ne = %d\n" , Ne );
        printf( "#dt*wpe = %f\n" , dt );
        printf( "#dx/de = %f\n" , Lx/(nx) );
        printf( "#dy/de = %f\n" , Ly/(ny) );
        printf( "#dz/de = %f\n" , Lz/(nz) );
        printf( "#n0 = %f\n" , n0 );
        printf( "*****" );

        for (size_t step = 0; step < num_steps; step++)
        {
            //     //std::cout << "Step " << step << std::endl;
            // Convert fields to interpolators

            load_interpolator_array(fields, interpolators, nx, ny, nz, num_ghosts);

            clear_accumulator_array(fields, accumulators, nx, ny, nz);
            //     // TODO: Make the frequency of this configurable (every step is not
            //     // required for this incarnation)
            //     // Sort by cell index
            //     auto keys = particles.slice<Cell_Index>();
            //     auto bin_data = Cabana::sortByKey( keys );

            // Move
            push(
                    particles,
                    interpolators,
                    qdt_2mc,
                    cdt_dx,
                    cdt_dy,
                    cdt_dz,
                    qsp,
                    scatter_add,
                    grid,
                    nx,
                    ny,
                    nz,
                    num_ghosts,
                    boundary
                );

            Kokkos::Experimental::contribute(accumulators, scatter_add);
            //for ( int zz = 0; zz < num_cells; zz++)
            //{
                //std::cout << "post accum " << zz << " = " << accumulators(zz, 0, 0) << std::endl;
            //}

            // Only reset the data if these two are not the same arrays
            scatter_add.reset_except(accumulators);

            // TODO: boundaries? MPI
            // boundary_p(); // Implies Parallel?

            // Map accumulator current back onto the fields
            unload_accumulator_array(fields, accumulators, nx, ny, nz, num_ghosts, dx, dy, dz, dt);

            //     // Half advance the magnetic field from B_0 to B_{1/2}
	    field_solver.advance_b(fields, real_t(0.5)*px, real_t(0.5)*py, real_t(0.5)*pz, nx, ny, nz, num_ghosts);

            // Advance the electric field from E_0 to E_1
            field_solver.advance_e(fields, px, py, pz, nx, ny, nz, num_ghosts, dt_eps0);

	    // Half advance the magnetic field from B_{1/2} to B_1
	    field_solver.advance_b(fields, real_t(0.5)*px, real_t(0.5)*py, real_t(0.5)*pz, nx, ny, nz, num_ghosts);

            //     // Print particles.
            //     print_particles( particles );

            //     // Output vis
            //     vis.write_vis(particles, step);
            printf("%d  %f  %e  %e\n",step, step*dt,field_solver.e_energy(fields, px, py, pz, nx, ny, nz),field_solver.b_energy(fields, px, py, pz, nx, ny, nz));
        }


    } // End Scoping block

    printf("#Good!\n");
    // Finalize.
    Cabana::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
//

////// Known Possible Improvements /////
// I pass nx/ny/nz round a lot more than I could

