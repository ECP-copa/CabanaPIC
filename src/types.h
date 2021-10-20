#ifndef pic_types_h
#define pic_types_h

#ifndef REAL_TYPE
#define real_t float
#else
#define real_t REAL_TYPE
#endif


#include <Kokkos_ScatterView.hpp>
#include <Cabana_Types.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Parallel.hpp>

// Inner array size (the size of the arrays in the structs-of-arrays).

#ifndef CELL_BLOCK_FACTOR
#define CELL_BLOCK_FACTOR 32
#endif
// Cell blocking factor in memory
const size_t cell_blocking = CELL_BLOCK_FACTOR;


// TODO: do we even need to explicitly specify these? We only use the default
// space..
#ifdef USE_GPU
using MemorySpace = Kokkos::CudaSpace;
using ExecutionSpace = Kokkos::Cuda;
#else
  #ifdef USE_SERIAL_CPU
    //cpu
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
  #else // CPU Parallel
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace; //Kokkos::OpenMP;
  #endif
#endif

typedef Kokkos::View<Kokkos::complex<real_t>*,  MemorySpace>   ViewVecComplex;

// Defaults
//using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
//using ExecutionSpace = Kokkos::DefaultExecutionSpace;

///// END ESSENTIALS ///

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
    real_t,                        // (0) x-position
    real_t,                        // (1) y-position
    real_t,                        // (2) z-position
    real_t,                        // (3) x-velocity
    real_t,                        // (4) y-velocity
    real_t,                        // (5) z-velocity
    real_t,                        // (6) weight
    int                           // (7) Cell index
>;

// Set the type for the particle AoSoA.
using particle_list_t =
    Cabana::AoSoA<ParticleDataTypes,MemorySpace>;

/////////////// START VPIC TYPE ////////////

#include "grid.h"

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
        real_t, //  ex,
        real_t , // dexdy,
        real_t , // dexdz,
        real_t , // d2exdydz,
        real_t , // ey,
        real_t , // deydz,
        real_t , // deydx,
        real_t , // d2eydzdx,
        real_t , // ez,
        real_t , // dezdx,
        real_t , // dezdy,
        real_t , // d2ezdxdy,
        // Below here is not need for ES? EM only?
        real_t , // cbx,
        real_t , // dcbxdx,
        real_t , // cby,
        real_t , // dcbydy,
        real_t , // cbz,
        real_t // dcbzdz,
        >;
    using interpolator_array_t = Cabana::AoSoA<InterpolatorDataTypes,MemorySpace,cell_blocking>;
using AccumulatorDataTypes =
    Cabana::MemberTypes<
    real_t[12] // jx[4] jy[4] jz[4]
>;

//using accumulator_array_t = Cabana::AoSoA<AccumulatorDataTypes,MemorySpace,cell_blocking>;

#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4

// TODO: should we flatten this out to 1D 12 big?
using accumulator_array_t = Kokkos::View<real_t* [ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

using accumulator_array_sa_t = Kokkos::Experimental::ScatterView<
    real_t *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>; //, KOKKOS_LAYOUT,
    //Kokkos::DefaultExecutionSpace, Kokkos::Experimental::ScatterSum,
    //KOKKOS_SCATTER_DUPLICATED, KOKKOS_SCATTER_ATOMIC
//>;

namespace accumulator_var {
  enum a_v { \
    jx = 0, \
    jy = 1, \
    jz = 2, \
  };
}

// for charge deposition
using rho_array_t =
    Kokkos::View<real_t *, Kokkos::LayoutRight, MemorySpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;

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
  ex,   ey,   ez,   div_e_err;     // Electric field and div E error
  cbx,  cby,  cbz,  div_b_err;     // Magnetic field and div B error
  tcax, tcay, tcaz, rhob;          // TCA fields and bound charge density
  jfx,  jfy,  jfz,  rhof;          // Free current and charge density
  material_id ematx, ematy, ematz, nmat; // Material at edge centers and nodes
  material_id fmatx, fmaty, fmatz, cmat; // Material at face and cell centers
  */

  real_t, // ex
  real_t, // ey
  real_t, // ez
  real_t, // cbx
  real_t, // cby
  real_t, // cbz
  real_t, // jfx
  real_t, // jfy
  real_t // jfz
>;

using field_array_t = Cabana::AoSoA<FieldDataTypes,MemorySpace,cell_blocking>;

// TODO: should this be in it's own file?
class particle_mover_t {
    public:
  real_t dispx, dispy, dispz; // Displacement of particle
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
