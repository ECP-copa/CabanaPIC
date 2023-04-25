#include <Cabana_Core.hpp>
#include <Cabana_AoSoA.hpp>

#include "particleManager.h"
#include "src/input/deck.h"
#include "src/input/defaultDeck.h"
#include "decks/landau_damping.h"
#include "particleDiagnostics.h"

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{    
    // Initialize the kokkos runtime.
    Kokkos::ScopeGuard scope_guard( argc, argv );
    printf("#Running On Kokkos execution space %s\n",
            typeid (Kokkos::DefaultExecutionSpace).name ());

    // Cabana scoping block
    {

	//        typedef ParticleManager< Particle, LandauDampingDeck, PrintFile > MyParticleMgr;
#ifdef ES_FIELD_SOLVER
	std::cout<<"Use ES_FIELD_SOLVER\n";
        typedef ParticleManager< Particle, ES_Field_Solver_1D, defaultDeck, PrintFile > MyParticleMgr;
#else // EM
	std::cout<<"Use EM_FIELD_SOLVER\n";
        typedef ParticleManager< Particle, EM_Field_Solver, defaultDeck, PrintFile > MyParticleMgr;
#endif
	
	MyParticleMgr aParticleMgr;
	aParticleMgr.Create();
	Input_Deck *deck = aParticleMgr.getDeck();
	aParticleMgr.createParticles(deck);
	aParticleMgr.createGrid(deck);
	aParticleMgr.createInterpolator(deck);
	aParticleMgr.createAccumulator(deck);
	aParticleMgr.createFieldSolver(deck);
	aParticleMgr.setBoundaryType(deck);
	aParticleMgr.timeStepping(deck);
	
	aParticleMgr.Print();
	aParticleMgr.Print();
	
    } // End Scoping block

    return 0;
}
