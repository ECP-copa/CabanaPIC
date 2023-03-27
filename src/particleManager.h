#ifndef paritcleManger_h
#define particleManger_h

template <class Particle,
	  class Field,
	  class CreationPolicy,
	  template <class U> class PrintPolicy
	  >
class ParticleManager
    : public CreationPolicy
    , public PrintPolicy<Particle>
{
};

#endif
