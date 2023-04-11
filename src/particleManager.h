#ifndef paritcleManger_h
#define particleManger_h
#include "src/input/deck.h"
#include "interpolator.h"
#include "fields.h"

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

    void setBoundaryType(Input_Deck * deck)
    {
	d_boundary = deck->BOUNDARY_TYPE;
    }
    
    void createFieldSolver(Input_Deck * deck)
    {
	const size_t num_cells = deck->num_cells;
	d_fields = field_array_t("fields", num_cells);

	d_field_solver = new Field_Solver<Field> (d_fields);
	
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
