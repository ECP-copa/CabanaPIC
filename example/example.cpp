#include <Cabana_Core.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Sort.hpp> // is this needed if we already have core?

#include <cstdlib>
#include <iostream>

#include "types.h"
#include "helpers.h"

#include "fields.h"
#include "accumulator.h"
#include "interpolator.h"

#include "uncenter_p.h"

#include "push.h"

#include "visualization.h"

#include "input/deck.h"
//#include "../decks/2stream-short.cxx"

// Global variable to hold paramters
//Parameters params;
Input_Deck deck;

Visualizer visualize;

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Kokkos::ScopeGuard scope_guard( argc, argv );

    printf("#Running On Kokkos execution space %s\n",
            typeid (Kokkos::DefaultExecutionSpace).name ());


#ifndef ENERGY_DUMP_INTERVAL
#define ENERGY_DUMP_INTERVAL 1
#endif

    // Cabana scoping block
    {
        FILE *fptr = fopen("partloc","w");
        FILE *fpfd = fopen("ex1d","w");
        deck.derive_params(); 			// compute derived parameters from user-specified parameters in input deck
        deck.print_run_details();		// print out various run parameters

        // Cache some values locally for printing
        const int npc = deck.nppc;
        const int nx = deck.nx;
        const int ny = deck.ny;
        const int nz = deck.nz;
        const int num_ghosts = deck.num_ghosts;
        const size_t num_cells = deck.num_cells;
        real_t dxp = 2.f / (npc);

		  // TODO: Move these parameters into input deck
		  // implicitness,  SG, verbosity
		  bool implicit = false;
		  int maxits = 5;
		  bool SG = true;
		  bool verbose = false;

        // Define some consts
        const real_t dx = deck.dx;
        const real_t dy = deck.dy;
        const real_t dz = deck.dz;
	const real_t dV = dx*dy*dz;

        real_t dt = deck.dt;
        real_t c = deck.c;
        real_t n0 = deck.n0;
        //real_t ec = deck.ec;
        real_t Lx = deck.len_x;
        real_t Ly = deck.len_y;
        real_t Lz = deck.len_z;
        real_t v0 = deck.v0;

        int nppc = deck.nppc;
        real_t eps0 = deck.eps;

        real_t Npe = deck.Npe;
        size_t Ne = deck.Ne; // (nppc*nx*ny*nz)
        printf("nppc %d nx %d ny %d nz %d \n", nppc, nx, ny, nz);
             printf("n0 %e lx %e nly %e lz %e \n", n0, Lx, Ly, Lz);
        printf("ne %ld npe %e \n", Ne, Npe);

        real_t qsp = deck.qsp;
        printf("qsp %e \n", qsp);
        real_t me = deck.me;

        real_t qdt_2mc = qsp*dt/(2*me*c);

        real_t cdt_dx = c*dt/dx;
        real_t cdt_dy = c*dt/dy;
        real_t cdt_dz = c*dt/dz;
        real_t dt_eps0 = dt/eps0;
        real_t frac = 1.0f;
        real_t we = (real_t) Npe/(real_t) Ne;
        printf("we %e \n", we);

        const size_t num_particles = deck.num_particles;

        printf("c %e dt %e dx %e cdt_dx %e \n", c, dt,dx,cdt_dx);

        // Create the particle list.
        particle_list_t particles( "particles", num_particles );

		  // TODO: Wrap this in an ifdef statement... only needed for implicit version
		  particle_list_t old_particles( "old_particles", num_particles );
       
		  // Initialize particles.
        deck.initialize_particles( particles, nx, ny, nz, num_ghosts, dxp, npc, we, v0 );

        grid_t* grid = new grid_t();

        // Print initial particle positions
        fprintf(fptr,"#step=0\n0 ");
        dump_particles( fptr, particles, 0, 0, 0, dx,dy,dz,nx,ny,nz,num_ghosts );

        // Allocate Cabana Data
        interpolator_array_t interpolators("interpolator", num_cells);

        accumulator_array_t accumulators("accumulator", num_cells);
	
        auto scatter_add = Kokkos::Experimental::create_scatter_view(accumulators);
        //<Kokkos::Experimental::ScatterSum,
        //KOKKOS_SCATTER_DUPLICATED,
        //KOKKOS_SCATTER_ATOMIC>(accumulators);

		  // Create array to hold fields
		  rho_array_t rho_accumulator("rho_accumulator",num_cells);

		  //get the initial charge density
		  accumulate_rho_p_1D(particles,rho_accumulator,nx,ny,nz,num_ghosts,dx,dy,dz,qsp);
	
        field_array_t fields("fields", num_cells);
		  // TODO: wrap this in an ifdef... only needed for implicit version
		  field_array_t old_fields( "old_fields", num_cells);

        // Zero out the interpolator
        // Techincally this is optional?
        initialize_interpolator(interpolators);

        // Can obviously supply solver type at compile time
        //Field_Solver<EM_Field_Solver> field_solver(fields);
        //Field_Solver<ES_Field_Solver_1D> field_solver(fields);
        // This is able to deduce solver type from compile options
        auto field_solver = make_field_solver(fields);

        deck.initialize_fields(
            fields,
            nx,
            ny,
            nz,
            num_ghosts,
            Lx,
            Ly,
            Lz,
            dx,
            dy,
            dz
        );
		  std::cout << "Completed field initialization" << std::endl;

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
        const int num_steps = deck.num_steps;

        printf( "#***********************************************\n" );
        printf( "#num_step = %d\n" , num_steps );
        printf( "#Lx/de = %f\n" , Lx );
        printf( "#Ly/de = %f\n" , Ly );
        printf( "#Lz/de = %f\n" , Lz );
        printf( "#nx = %d\n" , nx );
        printf( "#ny = %d\n" , ny );
        printf( "#nz = %d\n" , nz );
        printf( "#nppc = %d\n" , nppc );
        printf( "# Ne = %ld\n" , Ne );
        printf( "#dt*wpe = %f\n" , dt );
        printf( "#dx/de = %f\n" , Lx/(nx) );
        printf( "#dy/de = %f\n" , Ly/(ny) );
        printf( "#dz/de = %f\n" , Lz/(nz) );
        printf( "#n0 = %f\n" , n0 );
        printf( "#we = %f\n" , we );
        printf( "*****\n" );

		  // An initial backward half-step in velocity to initialize the standard Boris/leapfrog scheme
        if (deck.perform_uncenter)
        {
            load_interpolator_array(fields, interpolators, nx, ny, nz, num_ghosts);

            uncenter_particles(
                particles,
                interpolators,
                qdt_2mc
            );
        }

		  int itcount;

		  Binomial_Filters SGfilt(nx,ny,nz,num_ghosts);
		  int minres = 8;

		  real_t dt_frac = 1.;
		  bool converged, last_iteration, deposit_current;  // Deposit current tells you whether to do current deposition during a particle push.  For implicit, you only want to do this if it's not the last iteration.  
		  																	 // For explicit, you always want to do it.
		  int step = 0;
		  const real_t tot_en0 = dump_energies(particles, field_solver, fields, step, step*dt, px, py, pz, nx, ny, nz, num_ghosts,dV);
        // Main loop //
		  std::cout << "Starting main loop" << std::endl;
        for ( step = 1; step <= num_steps; step++)
        {
            //printf("Step %d \n", step);

            // Convert fields to interpolators
            load_interpolator_array(fields, interpolators, nx, ny, nz, num_ghosts);

            clear_accumulator_array(fields, accumulators, nx, ny, nz);
            // TODO: Make the frequency of this configurable (every step is not
            // required for this incarnation)
            // Sort by cell index
            //auto keys = particles.slice<Cell_Index>();
            //auto bin_data = Cabana::sortByKey( keys );
				//
				
				// TODO: Need to wrap this in an ifdef statement
				if ( implicit ) {
					Cabana::deep_copy( old_particles, particles ); // record particle states at start of time-step for implicit run
					Cabana::deep_copy( old_fields, fields ); // do the same for fields
				}
			   
				converged = false;
				last_iteration = !implicit; // a flag to see if we're on the last iteration.  If we are, do a full time-step instead of a half.  If we're not implicit, first iteration is the last
				itcount = 0; 
				while ( !converged )
				{
					if ( verbose ) { std::cout << "Started iteration number " << itcount << std::endl; }
					if ( last_iteration )
						  {
						  		dt_frac = 1.; 
						  }
						  else { dt_frac = 0.5; }
           			  
						  deposit_current = !last_iteration || !implicit;

			  			  if ( implicit ) {
						  		Cabana::deep_copy( particles, old_particles ); // reset particle states to beginning of time-step
						  }
						  if ( verbose ) { std::cout << "Starting particle push" << std::endl; }
						  // Particle push for half a time-step (if implicit) or full step (if explicit)
						  push(
									 particles,
									 interpolators,
									 dt_frac*qdt_2mc,
									 dt_frac*cdt_dx,
									 dt_frac*cdt_dy,
									 dt_frac*cdt_dz,
									 qsp,
									 scatter_add,
									 grid,
									 nx,
									 ny,
									 nz,
									 num_ghosts,
									 boundary,
									 deposit_current
								);
						  if ( verbose ) { std::cout << "Pushed particles" << std::endl; }
						  if ( !last_iteration && implicit )
						  {
						  		Cabana::deep_copy( fields, old_fields ); // Reset fields back to beginning of time-step
						  }
						  Kokkos::Experimental::contribute(accumulators, scatter_add); 
						  // Only reset the data if these two are not the same arrays
						  scatter_add.reset_except(accumulators);
						  // TODO: boundaries? MPI
						  //boundary_p(); // Implies Parallel?

						  // Map accumulator current back onto the fields
						  unload_accumulator_array(fields, accumulators, nx, ny, nz, num_ghosts, dx, dy, dz, dt);  // this is where the current gets put into the fields array

						  // Fill ghosts cells

						  auto jfx = Cabana::slice<FIELD_JFX>(fields);
						  auto jfy = Cabana::slice<FIELD_JFY>(fields);
						  auto jfz = Cabana::slice<FIELD_JFZ>(fields);

						  serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, num_ghosts); // Add current deposited to last ghost cell into first valid cell
						  serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, num_ghosts); // Apply periodic BCs


						  // SG filtering 
						  if ( SG ) {
						  		for (int filts=0; filts<5; filts++) {
						  			SGfilt.SGfilter(fields, nx, ny, nz, num_ghosts, minres);
								}
						  }
						  // Half advance the magnetic field from B_0 to B_{1/2}
						  field_solver.advance_b(fields, dt_frac*real_t(0.5)*px, dt_frac*real_t(0.5)*py, dt_frac*real_t(0.5)*pz, nx, ny, nz, num_ghosts);

						  // Advance the electric field from E_0 to E_1
						  if ( !last_iteration || !implicit) {
						  		field_solver.advance_e(fields, dt_frac*px, dt_frac*py, dt_frac*pz, nx, ny, nz, num_ghosts, dt_eps0);
						  }
						  else {
								if ( verbose ) { std::cout << "Extending E" << std::endl; }
								field_solver.extend_e(fields, old_fields); // get E_1 given E_{1/2} and E_0
						  }
						  // Half advance the magnetic field from B_{1/2} to B_1
						  field_solver.advance_b(fields, dt_frac*real_t(0.5)*px, dt_frac*real_t(0.5)*py, dt_frac*real_t(0.5)*pz, nx, ny, nz, num_ghosts);
						  if ( verbose ) { std::cout << "Advanced fields" << std::endl; }

						  if ( !last_iteration ) { // only need to reload interpolator array if we're going to do another iteration
									 // Convert fields to interpolators for next iteration
									 load_interpolator_array(fields, interpolators, nx, ny, nz, num_ghosts);
									 clear_accumulator_array(fields, accumulators, nx, ny, nz);
						  }

						  itcount++;
						  if ( last_iteration ) { converged = true; }
						  if ( itcount >= maxits-1 ) { last_iteration = true; } // Currently, don't check for convergence, just do a fixed number of iterations

						  if ( verbose ) { std::cout << "Iteration number " << itcount << " -  Step number " << step <<  std::endl; }
				}

				if ( verbose ) { std::cout << "Converged step number " << step << std::endl; }

            if( step % ENERGY_DUMP_INTERVAL == 0 )
            {
                real_t tot_en = dump_energies(particles, field_solver, fields, step, step*dt, px, py, pz, nx, ny, nz, num_ghosts,dV,tot_en0);
		//dump_kinetic_energy( particles, step, step*dt );
            }

            // TODO: abstract this out
            fprintf(fpfd,"#step=%d\n",step);
            //field_solver.dump_fields(fpfd,fields, 0, 0, 0, dx,dy,dz,nx,ny,nz,num_ghosts );
            fprintf(fptr,"#step=%d\n%e ",step,step*dt);
            //dump_particles( fptr, particles, 0, 0, 0, dx,dy,dz,nx,ny,nz,num_ghosts );

				// Write out visualization files
        		visualize.write_vis(particles, fields, step, nx, ny, nz, num_ghosts, dx, dy, dz);

        }

        fclose(fptr);
        fclose(fpfd);

    } // End Scoping block

    // Let the user perform any needed finalization
    deck.finalize();

    return 0;
}

//---------------------------------------------------------------------------//
//

////// Known Possible Improvements /////
// I pass nx/ny/nz round a lot more than I could

