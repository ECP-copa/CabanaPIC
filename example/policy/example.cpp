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
        typedef ParticleManager< Particle, Field, defaultDeck, PrintFile > MyParticleMgr;

	MyParticleMgr aParticleMgr;
	aParticleMgr.Create();
	aParticleMgr.Print();
	aParticleMgr.Print();
	
    } // End Scoping block

    return 0;
}
