#ifndef paritcleManger_h
#define particleManger_h
#include "src/input/deck.h"
#include "interpolator.h"
#include "accumulator.h"
#include "fields.h"
#include "push.h"

#ifndef ENERGY_DUMP_INTERVAL
#define ENERGY_DUMP_INTERVAL 1
#endif

template <class Particle,
	  class Field,
	  class DeckPolicy,
	  template <class U> class PrintPolicy
	  >
class ParticleManager
    : public DeckPolicy
    , public PrintPolicy<Particle>
{
public:
    ParticleManager() {}
    ~ParticleManager()
    {
	delete d_field_solver;
    }

    void timeStepping(Input_Deck * deck)
    {
	const int nx = deck->nx;
	const int ny = deck->ny;
	const int nz = deck->nz;
	const int num_ghosts = deck->num_ghosts;	
        const real_t dx = deck->dx;
        const real_t dy = deck->dy;
        const real_t dz = deck->dz;
	
	real_t dt = deck->dt;
	real_t c = deck->c;
	real_t eps0 = deck->eps;
        real_t qsp = deck->qsp;
        real_t me = deck->me;
        real_t qdt_2mc = qsp*dt/(2*me*c);

        real_t cdt_dx = c*dt/dx;
        real_t cdt_dy = c*dt/dy;
        real_t cdt_dz = c*dt/dz;
        real_t dt_eps0 = dt/eps0;
        real_t frac = 1.0f;
        const real_t px =  (nx>1) ? frac*c*dt/dx : 0;
        const real_t py =  (ny>1) ? frac*c*dt/dy : 0;
        const real_t pz =  (nz>1) ? frac*c*dt/dz : 0;

	const int num_steps = deck->num_steps;
	
        auto scatter_add = Kokkos::Experimental::create_scatter_view(d_accumulators);
	
	for(size_t step = 1; step <= num_steps; ++step)
	{

	    load_interpolator_array(d_fields, d_interpolators, nx, ny, nz, num_ghosts);

	    clear_accumulator_array(d_fields, d_accumulators, nx, ny, nz);

            // Move
	    for(int is=0; is<d_numSpecies; ++is){
		push(
		     d_particles_k[is],
		     d_interpolators,
		     qdt_2mc,
		     cdt_dx,
		     cdt_dy,
		     cdt_dz,
		     qsp,
		     scatter_add,
		     d_grid,
		     nx,
		     ny,
		     nz,
		     num_ghosts,
		     d_boundary
		     );
	    }
	    
	    Kokkos::Experimental::contribute(d_accumulators, scatter_add);
	    // Only reset the data if these two are not the same arrays
            scatter_add.reset_except(d_accumulators);
	    // Map accumulator current back onto the fields
            unload_accumulator_array(d_fields, d_accumulators, nx, ny, nz, num_ghosts, dx, dy, dz, dt);
	    
	    //TODO: wrap it to be one call instead of three
	    // Half advance the magnetic field from B_0 to B_{1/2}
            d_field_solver->advance_b(d_fields, real_t(0.5)*px, real_t(0.5)*py, real_t(0.5)*pz, nx, ny, nz, num_ghosts);
	    
            // Advance the electric field from E_0 to E_1
            d_field_solver->advance_e(d_fields, px, py, pz, nx, ny, nz, num_ghosts, dt_eps0);

            // Half advance the magnetic field from B_{1/2} to B_1
            d_field_solver->advance_b(d_fields, real_t(0.5)*px, real_t(0.5)*py, real_t(0.5)*pz, nx, ny, nz, num_ghosts);


            if( step % ENERGY_DUMP_INTERVAL == 0 )
            {
                dump_energies(*d_field_solver, d_fields, step, step*dt, px, py, pz, nx, ny, nz, num_ghosts);
            }

	}

    }
    
    void setBoundaryType(Input_Deck * deck)
    {
	d_boundary = deck->BOUNDARY_TYPE;
    }
    
    void createFieldSolver(Input_Deck * deck)
    {
	const size_t num_cells = deck->num_cells;
	d_fields = field_array_t("fields", num_cells);

	d_field_solver = new Field_Solver<Field> (d_fields);

	//initialize fields
	const int nx = deck->nx;
	const int ny = deck->ny;
	const int nz = deck->nz;
	const int num_ghosts = deck->num_ghosts;	
        const real_t Lx = deck->len_x;
        const real_t Ly = deck->len_y;
        const real_t Lz = deck->len_z;
        const real_t dx = deck->dx;
        const real_t dy = deck->dy;
        const real_t dz = deck->dz;
	
        deck->initialize_fields(
			       d_fields,
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
	
    }
    
    void createAccumulator(Input_Deck * deck)
    {
	const size_t num_cells = deck->num_cells;
	d_accumulators = accumulator_array_t("accumulator", num_cells);	
    }

    
    void createInterpolator(Input_Deck * deck)
    {
	const size_t num_cells = deck->num_cells;
	d_interpolators = interpolator_array_t("interpolator", num_cells);
	initialize_interpolator(d_interpolators); //calling an extenal function
    }
    
    void createGrid(Input_Deck * deck)
    {
	d_grid = new grid_t();
	
    }
    
    void createParticles(Input_Deck * deck)
    {
	d_numSpecies = deck->num_species;
	std::cout<<"number of species="<<d_numSpecies<<"\n";
	for(int is=0; is<d_numSpecies; ++is){
	    d_nip.push_back( 0 );
	    auto num_particles = deck->num_particles + d_nip[is];
	    d_particles_k.push_back( particle_list_t( "particleSpecies" + std::to_string( is ),
	     					      num_particles ) );
	    std::cout << "###In particleData: is=" << is << ",n_p.size()=" << d_particles_k[is].size()
	     	      << ",nip=" << d_nip[is] << ", particle type "
	     	      << typeid( d_particles_k[is] ).name() << "\n";
	    // Initialize particles.
	    const int npc = deck->nppc;
	    const int nx = deck->nx;
	    const int ny = deck->ny;
	    const int nz = deck->nz;
	    const int num_ghosts = deck->num_ghosts;
	    real_t dxp = 2.f / (npc);
	    real_t Npe = deck->Npe;
	    size_t Ne = deck->Ne; // (nppc*nx*ny*nz)
	    real_t we = (real_t) Npe/(real_t) Ne;
	    real_t v0 = deck->v0;
	    
	    deck->initialize_particles( d_particles_k[is], nx, ny, nz, num_ghosts, dxp, npc, we, v0 );
	    
	}
	
    }

    std::vector<particle_list_t> &getCabanaParticles() { return d_particles_k; }


private:
    std::vector<particle_list_t> d_particles_k;
    size_t d_numSpecies;
    std::vector<size_t> d_nip; // number of inactive particles
    grid_t* d_grid;
    interpolator_array_t d_interpolators;
    accumulator_array_t d_accumulators;
    field_array_t d_fields;
    Field_Solver<Field> *d_field_solver;
    Boundary d_boundary;
    
};

#endif
