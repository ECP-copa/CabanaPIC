#ifndef pic_types_h
#define pic_types_h

#define real_t float

// Inner array size (the size of the arrays in the structs-of-arrays).
#ifndef VLEN
#define VLEN 16
#endif
const std::size_t array_size = VLEN;

#ifndef CELL_BLOCK_FACTOR
#define CELL_BLOCK_FACTOR 16
#endif
// Cell blocking factor in memory
const size_t cell_blocking = CELL_BLOCK_FACTOR;

using MemorySpace = Kokkos::CudaUVMSpace;
using ExecutionSpace = Kokkos::Cuda;
//using MemorySpace = Cabana::HostSpace;
//using ExecutionSpace = Kokkos::Serial;
//using parallel_algorithm_tag = Cabana::StructParallelTag;

///// END ESSENTIALS ///

#include "simulation_parameters.h"

using Parameters = Parameters_<real_t>;

enum UserParticleFields
{
    PositionX = 0,
    PositionY,
    PositionZ,
    VelocityX,
    VelocityY,
    VelocityZ,
    Charge,
    Cell_Index, // This is stored as per VPIC, such that it includes ghost_offsets
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

/////////////// START VPIC TYPE ////////////

#include <grid.h>

#ifdef USE_NON_KOKKOS_TYPES // UNTESTED!
#include <interpolator.h>
#include <accumulator.h>
#else

enum InterpolatorFields
{ // TODO: things in here like EXYZ and CBXYZ are ambigious
    EX = 0,
    DEXDY,
    DEXDZ,
    D2EXDYDZ,
    EY,
    DEYDZ,
    DEYDX,
    D2EYDZDX,
    EZ,
    DEZDX,
    DEZDY,
    D2EZDXDY,
    CBX,
    DCBXDX,
    CBY,
    DCBYDY,
    CBZ,
    DCBZDZ
};

    using InterpolatorDataTypes =
        Cabana::MemberTypes<
        float, //  ex,
        float , // dexdy,
        float , // dexdz,
        float , // d2exdydz,
        float , // ey,
        float , // deydz,
        float , // deydx,
        float , // d2eydzdx,
        float , // ez,
        float , // dezdx,
        float , // dezdy,
        float , // d2ezdxdy,
        // Below here is not need for ES? EM only?
        float , // cbx,
        float , // dcbxdx,
        float , // cby,
        float , // dcbydy,
        float , // cbz,
        float // dcbzdz,
        >;
    using interpolator_array_t = Cabana::AoSoA<InterpolatorDataTypes,MemorySpace,cell_blocking>;
using AccumulatorDataTypes =
    Cabana::MemberTypes<
    float[12] // jx[4] jy[4] jz[4]
>;
using accumulator_array_t = Cabana::AoSoA<AccumulatorDataTypes,MemorySpace,cell_blocking>;
#endif

enum FieldFields
{
    FIELD_EX = 0,
    FIELD_EY,
    FIELD_EZ,
    FIELD_CBX,
    FIELD_CBY,
    FIELD_CBZ,
    FIELD_JFX,
    FIELD_JFY,
    FIELD_JFZ
};
using FieldDataTypes = Cabana::MemberTypes<
/*
  float ex,   ey,   ez,   div_e_err;     // Electric field and div E error
  float cbx,  cby,  cbz,  div_b_err;     // Magnetic field and div B error
  float tcax, tcay, tcaz, rhob;          // TCA fields and bound charge density
  float jfx,  jfy,  jfz,  rhof;          // Free current and charge density
  material_id ematx, ematy, ematz, nmat; // Material at edge centers and nodes
  material_id fmatx, fmaty, fmatz, cmat; // Material at face and cell centers
  */

  float, // ex
  float, // ey
  float, // ez
  float, // cbx
  float, // cby
  float, // cbz
  float, // jfx
  float, // jfy
  float // jfz
>;

using field_array_t = Cabana::AoSoA<FieldDataTypes,MemorySpace,cell_blocking>;

// TODO: should this be in it's own file?
class particle_mover_t {
    public:
  float dispx, dispy, dispz; // Displacement of particle
  int32_t i;                 // Index of the particle to move
};

/////////////// END VPIC TYPE ////////////

#endif // pic_types_h
