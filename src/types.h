#ifndef pic_types_h
#define pic_types_h

#include <interpolator.h>
#define real_t float

//---------------------------------------------------------------------------//
// Define particle data.
//---------------------------------------------------------------------------//
// Inner array size (the size of the arrays in the structs-of-arrays).
#ifndef VLEN
#define VLEN 16
#endif
const std::size_t array_size = VLEN;

using MemorySpace = Cabana::HostSpace;
using ExecutionSpace = Kokkos::Serial;
using parallel_algorithm_tag = Cabana::StructParallelTag;

// User field enumeration. These will be used to index into the data set. Must
// start at 0 and increment contiguously.
//
// NOTE: Users don't have to make this enum (or some other set of integral
// constants) but it is a nice way to provide meaning to the different data
// types and values assigned to the particles.
//
// NOTE: These enums are also ordered in the same way as the data in the
// template parameters below.
enum UserParticleFields
{
    PositionX = 0,
    PositionY,
    PositionZ,
    VelocityX,
    VelocityY,
    VelocityZ,
    Charge,
    Cell_Index,
};

// Designate the types that the particles will hold.
using ParticleDataTypes =
Cabana::MemberTypes<
    float,                        // (0) x-position
    float,                        // (1) y-position
    float,                        // (2) z-position
    float,                        // (3) x-velocity
    float,                        // (4) y-velocity
    float,                        // (5) z-velocity
    float,                        // (6) charge
    int                           // (7) Cell index
>;

// Set the type for the particle AoSoA.
using particle_list_t =
    Cabana::AoSoA<ParticleDataTypes,MemorySpace,array_size>;

#endif // pic_types_h
