#ifndef pic_types_h
#define pic_types_h

#define real_t float

#include <Kokkos_ScatterView.hpp>

// Inner array size (the size of the arrays in the structs-of-arrays).
#ifndef VLEN
#define VLEN 16 //32
#endif
const std::size_t array_size = VLEN;

#ifndef CELL_BLOCK_FACTOR
#define CELL_BLOCK_FACTOR 32
#endif
// Cell blocking factor in memory
const size_t cell_blocking = CELL_BLOCK_FACTOR;

#ifdef USE_GPU
using MemorySpace = Kokkos::CudaUVMSpace;
using ExecutionSpace = Kokkos::Cuda;
#else
  #ifdef USE_SERIAL_CPU
    //cpu
    using MemorySpace = Cabana::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
  #else // CPU Parallel
    using MemorySpace = Cabana::HostSpace;
    using ExecutionSpace = Kokkos::OpenMP;
  #endif
#endif


#ifdef USE_GPU
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterNonDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterAtomic
  //#define KOKKOS_LAYOUT Kokkos::LayoutLeft
  #define KOKKOS_LAYOUT Kokkos::LayoutRight
#else
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterNonAtomic
  #define KOKKOS_LAYOUT Kokkos::LayoutRight
#endif


// Defaults
//using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
//using ExecutionSpace = Kokkos::DefaultExecutionSpace;

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
    Weight,
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
    float,                        // (6) weight
    int                           // (7) Cell index
>;

// Set the type for the particle AoSoA.
using particle_list_t =
    Cabana::AoSoA<ParticleDataTypes,MemorySpace,array_size>;

/////////////// START VPIC TYPE ////////////

#include <grid.h>

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

//using accumulator_array_t = Cabana::AoSoA<AccumulatorDataTypes,MemorySpace,cell_blocking>;

#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4

// TODO: should we flatten this out to 1D 12 big?
using accumulator_array_t = Kokkos::View<float* [ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

using accumulator_array_sa_t = Kokkos::Experimental::ScatterView<
    float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], KOKKOS_LAYOUT,
    Kokkos::DefaultExecutionSpace, Kokkos::Experimental::ScatterSum,
    KOKKOS_SCATTER_DUPLICATED, KOKKOS_SCATTER_ATOMIC
>;

namespace accumulator_var {
  enum a_v { \
    jx = 0, \
    jy = 1, \
    jz = 2, \
  };
}



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
//
// TODO: this may be a bad name?
# define RANK_TO_INDEX(rank,ix,iy,iz,_x,_y) \
    int _ix, _iy, _iz;                                                    \
    _ix  = (rank);                        /* ix = ix+gpx*( iy+gpy*iz ) */ \
    _iy  = _ix/int(_x);   /* iy = iy+gpy*iz */            \
    _ix -= _iy*int(_x);   /* ix = ix */                   \
    _iz  = _iy/int(_y);   /* iz = iz */                   \
    _iy -= _iz*int(_y);   /* iy = iy */                   \
    (ix) = _ix;                                                           \
    (iy) = _iy;                                                           \
    (iz) = _iz;                                                           \

#define VOXEL(x,y,z, nx,ny,nz, NG) ((x) + ((nx)+(NG*2))*((y) + ((ny)+(NG*2))*(z)))

#endif // pic_types_h
