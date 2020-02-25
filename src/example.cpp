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
#include "interpolator.h"

#include "push.h"

#include "visualization.h"

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
      FILE *fptr = fopen("partloc","w");
      FILE *fpfd = fopen("ex1d","w");
      
        Visualizer vis;

        // Initialize input deck params.

        // num_cells (without ghosts), num_particles_per_cell
        size_t npc = 40;
        Initializer::initialize_params(64, npc);

        // Cache some values locally for printing
        const size_t nx = Parameters::instance().nx;
        const size_t ny = Parameters::instance().ny;
        const size_t nz = Parameters::instance().nz;
        const size_t num_ghosts = Parameters::instance().num_ghosts;
        const size_t num_cells = Parameters::instance().num_cells;
        const size_t num_particles = Parameters::instance().num_particles;
        real_t dxp = 2.f/(npc);

        // Define some consts
        const real_t dx = Parameters::instance().dx;
        const real_t dy = Parameters::instance().dy;
        const real_t dz = Parameters::instance().dz;
        real_t dt   = Parameters::instance().dt;
        real_t c    = Parameters::instance().c;
        real_t me   = Parameters::instance().me;
        real_t n0   = Parameters::instance().n0;
        real_t ec   = Parameters::instance().ec;
        real_t Lx   = Parameters::instance().len_x;
        real_t Ly   = Parameters::instance().len_y;
        real_t Lz   = Parameters::instance().len_z;
        size_t nppc = Parameters::instance().NPPC;
	real_t eps0 = Parameters::instance().eps;
        real_t Npe  = n0*Lx*0.2*Lz;
        size_t Ne=  (nppc*nx*ny*nz)/8;
        real_t qsp = -ec;
        real_t qdt_2mc = qsp*dt/(2*me*c);
        real_t cdt_dx = c*dt/dx;
        real_t cdt_dy = c*dt/dy;
        real_t cdt_dz = c*dt/dz;
	real_t dt_eps0 = dt/eps0;
        real_t frac = 1.0f;
        real_t we = (real_t) Npe/(real_t) Ne;
	
        // Create the particle list.
        particle_list_t particles( num_particles );
        //logger << "size " << particles.size() << std::endl;
        //logger << "numSoA " << particles.numSoA() << std::endl;

        // Initialize particles.
        Initializer::initialize_particles( particles, nx, ny, nz, num_ghosts, dxp, npc, we, Lx,Ly,Lz );
	
        grid_t* grid = new grid_t();

        // Print initial particle positions
        //logger << "Initial:" << std::endl;
        //print_particles( fptr, particles, -Lx/2.0, -Ly/2.0, -Lz/2.0, dx,dy,dz,nx,ny,nz,num_ghosts );
	fprintf(fptr,"#step=0\n");	
        print_particles( fptr, particles, -Lx/2.0, -Ly/2.0, -Lz/2.0, dx,dy,dz,nx,ny,nz,num_ghosts );

        // Allocate Cabana Data
        interpolator_array_t interpolators(num_cells);

        accumulator_array_t accumulators("Accumulator View", num_cells);

        auto scatter_add = Kokkos::Experimental::create_scatter_view(accumulators);
            //<Kokkos::Experimental::ScatterSum,
             //KOKKOS_SCATTER_DUPLICATED,
             //KOKKOS_SCATTER_ATOMIC>(accumulators);

        field_array_t fields(num_cells);

        Initializer::initialize_interpolator(interpolators);

        // Can obviously supply solver type at compile time
        Field_Solver<EM_Field_Solver> field_solver(fields, -Lx/2.0, -Ly/2.0, -Lz/2.0, dx,dy,dz,nx,ny,nz,num_ghosts );
        //Field_Solver<ES_Field_Solver_1D> field_solver(fields);

	field_solver.print_fields(fpfd,fields, -Lx/2.0, -Ly/2.0, -Lz/2.0, dx,dy,dz,nx,ny,nz,num_ghosts );

        // Grab some global values for use later
        const Boundary boundary = Parameters::instance().BOUNDARY_TYPE;

        //logger << "nx " << Parameters::instance().nx << std::endl;
        //logger << "num_particles " << num_particles << std::endl;
        //logger << "num_cells " << num_cells << std::endl;
        //logger << "Actual NPPC " << Parameters::instance().NPPC << std::endl;

        // TODO: give these a real value
        const real_t px =  (nx>1) ? frac*c*dt/dx : 0;
        const real_t py =  (ny>1) ? frac*c*dt/dy : 0;
        const real_t pz =  (nz>1) ? frac*c*dt/dz : 0;

        // simulation loop
        const size_t num_steps = Parameters::instance().num_steps;

        printf( "#***********************************************\n" );
        printf ( "#num_step = %d\n" , num_steps );
        printf ( "#Lx/de = %f\n" , Lx );
        printf ( "#Ly/de = %f\n" , Ly );
        printf ( "#Lz/de = %f\n" , Lz );
        printf ( "#nx = %d\n" , nx );
        printf ( "#ny = %d\n" , ny );
        printf ( "#nz = %d\n" , nz );
        printf ( "#nppc = %d\n" , nppc );
        printf ( "# Ne = %d\n" , Ne );
        printf ( "#dt*wpe = %f\n" , dt );
        printf ( "#dx/de = %f\n" , Lx/(nx) );
        printf ( "#dy/de = %f\n" , Ly/(ny) );
        printf ( "#dz/de = %f\n" , Lz/(nz) );
        printf ( "#n0 = %f\n" , n0 );
        printf ( "#we = %f\n" , we );
        printf( "" );

	for (size_t step = 1; step < num_steps+1; step++)
        //for (size_t step = 1; step < 11; step++)
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

	    //
	    //	    exit(1);
	    // Print particles.
	    if((step)%100==0){
	      fprintf(fpfd,"#step=%d\n",step);
	      field_solver.print_fields(fpfd,fields, -Lx/2.0, -Ly/2.0, -Lz/2.0, dx,dy,dz,nx,ny,nz,num_ghosts );
	      fprintf(fptr,"#step=%d\n",step);
	      print_particles( fptr, particles, -Lx/2.0, -Ly/2.0, -Lz/2.0, dx,dy,dz,nx,ny,nz,num_ghosts );
	      printf("%d  %f  %e  %e\n",step, step*dt,field_solver.e_energy(fields, px, py, pz, nx, ny, nz, num_ghosts),field_solver.b_energy(fields, px, py, pz, nx, ny, nz, num_ghosts));	      
	    }

            //     // Output vis
            //     vis.write_vis(particles, step);

        }
    fclose(fptr);

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

