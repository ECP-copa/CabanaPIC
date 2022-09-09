#ifndef particle_diagnostics_h
#define particle_diagnostics_h

#include "types.h"
struct mom_type {
    real_t value[7]; // one zero moment + three 1st moments + three 2nd moments
};

template<class SliceType>
struct tottemp {
    SliceType d_upx, d_upy, d_upz, d_wp;

    tottemp( SliceType upx, SliceType upy, SliceType upz, SliceType wp )
        : d_upx( upx ), d_upy( upy ), d_upz( upz ), d_wp( wp )
    {
    }

    // May need nested loops to be more efficient
    KOKKOS_INLINE_FUNCTION
    void operator()( const int i, real_t &lsum ) const
    {
        lsum += d_wp( i ) *
                ( d_upx( i ) * d_upx( i ) + d_upy( i ) * d_upy( i ) + d_upz( i ) * d_upz( i ) );
    }
};

template<class SliceType>
struct moms {
    SliceType d_upx, d_upy, d_upz, d_wp;

    moms( SliceType upx, SliceType upy, SliceType upz, SliceType wp )
        : d_upx( upx ), d_upy( upy ), d_upz( upz ), d_wp( wp )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile mom_type &dst, const volatile mom_type &src ) const
    {
        dst.value[0] += src.value[0];
        dst.value[1] += src.value[1];
        dst.value[2] += src.value[2];
        dst.value[3] += src.value[3];
        dst.value[4] += src.value[4];
        dst.value[5] += src.value[5];
        dst.value[6] += src.value[6];
    }


    KOKKOS_INLINE_FUNCTION
    void operator()( const int i, mom_type &lsum ) const
    {
        lsum.value[0] += d_wp( i );
        lsum.value[1] += d_wp( i ) * d_upx( i );
        lsum.value[2] += d_wp( i ) * d_upy( i );
        lsum.value[3] += d_wp( i ) * d_upz( i );
        lsum.value[4] += d_wp( i ) * d_upx( i ) * d_upx( i );
        lsum.value[5] += d_wp( i ) * d_upy( i ) * d_upy( i );
        lsum.value[6] += d_wp( i ) * d_upz( i ) * d_upz( i );
    }
};

void particle_moment( particle_list_t &particles,
                      const real_t m_i,
                      size_t nx,
		      size_t ny,
		      size_t nz,
                      size_t ng,
                      const std::vector<size_t> &npc_scan_i,
                      const moment_array_t &moment_i,
                      const int debug );

void print_2pcle( size_t step,
                  real_t dt,
                  particle_list_t &particles,
                  size_t nx,
                  size_t ny,
                  size_t ng,
                  real_t xmin,
                  real_t dx );

#endif
