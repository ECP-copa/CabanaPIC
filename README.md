# Overview 

This document is to be used to collect a wish list of features before
development starts

## Feature Wishlist

1. Configurable to run in different precisions (real_t to configure float/double)
2. The particle data store layout should be Configurable (AoS/SoA/AoSoA)
3. The particle shape function used should be configurable 

## Technical Details 

1. It should exclusively use OMP for threading
2. It should be written with OMP4.5 in mind, as well as being CUDA extensible 

## Considerations 

One of the more interesting things about PIC for advanced architecrues is
that the different aspects of the push require different (contradictory)
optimizations.

The main `particle_push` has three main parts:

### The particle move

**Particle Properties Used (memory streams)**:
    all  
**Data layouts**: 
i) AoS => Good, not great vectorized 
ii) SoA => Good, but many memory streams
iii) AoSoA => Ideal

- If we vectorize AoS we need to do a transpose 
- SoA doesn't really buy us anything in terms of memory streams here as we
    need all properties.
- Particle order isn't a concern here
- There isn't much burden on cache here as long as the next particle(s) are pre-fetched in time

### The field stencil (read) and particle velocity 

**Particle Properties Used (memory streams)**:
    Most  
**Data layouts**: 
i) AoS => Not great, doesn't need to use all the streams
ii) SoA =>  Good, lets you split the streams out
iii) AoSoA =>  Great

- The field stencil benefits significantly from having well ordered (on a cell
    basis) particles
- Cache use here is crucial because of the large, semi-nonobvious
    (prefetching), stencil
- If we can explicitly tell the compiler that groups of particle share
    properties, we get better re-use than good "accidental" cache reuse 


### The current accumulation stencil (write)

**Particle Properties Used (memory streams)**:
    Momentum  
**Data layouts**: 
i) AoS => Not great, doesn't need to use all the streams
ii) SoA =>  Good, lets you split the streams out
iii) AoSoA =>  Great

- If particles all write to the same cell, that gives good assumptions and
    makes it safe to do some writes
- If particles are highly disordered, this can cause big problems for the
    safety of the writes and often leads to atomics
