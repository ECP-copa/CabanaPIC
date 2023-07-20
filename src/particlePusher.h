#ifndef particle_pusher_h
#define particle_pusher_h

#include "particleBC.h"

class Particle_Pusher_Leapfrog
{
public:
template<typename T1, typename T2, typename T3, typename T4, typename T5> KOKKOS_INLINE_FUNCTION int move_p(
        //particle_list_t particles,
        T1& position_x,
        T2& position_y,
        T3& position_z,
        T4& cell,
        T5& a0, // TODO: does this need to be const
        real_t q,
        particle_mover_t& pm,
        const grid_t* ,
        const size_t s,
        const size_t i,
        const size_t nx,
        const size_t ny,
        const size_t nz,
        const size_t num_ghosts,
        const Boundary boundary
    )
{

    auto _asa = a0.access();

    /* // Kernel variables */
    real_t s_dir[3];
    real_t v0, v1, v2, v3; //, v4, v5;
    size_t axis, face;

    for(;;)
    {
        real_t s_midx = position_x.access(s, i);
        real_t s_midy = position_y.access(s, i);
        real_t s_midz = position_z.access(s, i);

        real_t s_dispx = pm.dispx;
        real_t s_dispy = pm.dispy;
        real_t s_dispz = pm.dispz;

        s_dir[0] = (s_dispx>0) ? 1 : -1;
        s_dir[1] = (s_dispy>0) ? 1 : -1;
        s_dir[2] = (s_dispz>0) ? 1 : -1;

        // Compute the twice the fractional distance to each potential
        // streak/cell face intersection.
        v0 = (s_dispx==0) ? 3.4e38 : (s_dir[0]-s_midx)/s_dispx;
        v1 = (s_dispy==0) ? 3.4e38 : (s_dir[1]-s_midy)/s_dispy;
        v2 = (s_dispz==0) ? 3.4e38 : (s_dir[2]-s_midz)/s_dispz;

        // Determine the fractional length and axis of current streak. The
        // streak ends on either the first face intersected by the
        // particle track or at the end of the particle track.
        //
        //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
        //   axis 3        ... streak ends at end of the particle track
        /**/      v3=2,  axis=3;
        if(v0<v3) v3=v0, axis=0;
        if(v1<v3) v3=v1, axis=1;
        if(v2<v3) v3=v2, axis=2;
        v3 *= 0.5;

        // Compute the midpoint and the normalized displacement of the streak
        s_dispx *= v3;
        s_dispy *= v3;
        s_dispz *= v3;
        s_midx += s_dispx;
        s_midy += s_dispy;
        s_midz += s_dispz;

        // Accumulate the streak.  Note: accumulator values are 4 times
        // the total physical charge that passed through the appropriate
        // current quadrant in a time-step
        //v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);

        int ii = cell.access(s, i);

        //a = (real_t *)(a0 + ii);

        //1D only
        // _asa(ii,accumulator_var::jx, 0) += 4.0f*q*s_dispx;
        // _asa(ii,accumulator_var::jx, 1) += 0.0;
        // _asa(ii,accumulator_var::jx, 2) += 0.0;
        // _asa(ii,accumulator_var::jx, 3) += 0.0;
        real_t v4;
        real_t v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);

#define accumulate_j(X,Y,Z)						\
        v4  = q*s_disp##X;    /* v2 = q ux                            */  \
        v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
        v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
        v1 += v4;             /* v1 = q ux (1+dy)                     */  \
        v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
        v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
        v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
        v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
        v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
        v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
        v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
        v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
        v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
        v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \

       accumulate_j(x,y,z);
       // printf("move_p deposit v0 %e to %d \n",
       //         v0, ii);
       _asa(ii, accumulator_var::jx, 0) += v0; // q*ux;
       _asa(ii, accumulator_var::jx, 1) += v1; // 0.0;
       _asa(ii, accumulator_var::jx, 2) += v2; // 0.0;
       _asa(ii, accumulator_var::jx, 3) += v3; // 0.0;

       accumulate_j(y,z,x);
       _asa(ii, accumulator_var::jy, 0) += v0; // q*ux;
       _asa(ii, accumulator_var::jy, 1) += v1; // 0.0;
       _asa(ii, accumulator_var::jy, 2) += v2; // 0.0;
       _asa(ii, accumulator_var::jy, 3) += v3; // 0.0;

       accumulate_j(z,x,y);
       _asa(ii, accumulator_var::jz, 0) += v0; // q*ux;
       _asa(ii, accumulator_var::jz, 1) += v1; // 0.0;
       _asa(ii, accumulator_var::jz, 2) += v2; // 0.0;
       _asa(ii, accumulator_var::jz, 3) += v3; // 0.0;

#   undef accumulate_j

        // Compute the remaining particle displacment
        pm.dispx -= s_dispx;
        pm.dispy -= s_dispy;
        pm.dispz -= s_dispz;

        //printf("%d %d, %d, %f %f",s, i, ii, position_x.access(s, i),position_x.access(s, i));
        // Compute the new particle offset
        position_x.access(s, i) += s_dispx+s_dispx;
        position_y.access(s, i) += s_dispy+s_dispy;
        position_z.access(s, i) += s_dispz+s_dispz;

        //printf(" %f\n",position_x.access(s, i));

        // If an end streak, return success (should be ~50% of the time)

        if( axis==3 ) break;

        // Determine if the particle crossed into a local cell or if it
        // hit a boundary and convert the coordinate system accordingly.
        // Note: Crossing into a local cell should happen ~50% of the
        // time; hitting a boundary is usually a rare event.  Note: the
        // entry / exit coordinate for the particle is guaranteed to be
        // +/-1 _exactly_ for the particle.

        v0 = s_dir[axis];

        // TODO: do branching based on axis

        //(&(p->dx))[axis] = v0; // Avoid roundoff fiascos--put the particle

        // TODO: this conditional could be better
        if (axis == 0) position_x.access(s, i) = v0;
        if (axis == 1) position_y.access(s, i) = v0;
        if (axis == 2) position_z.access(s, i) = v0;

        // _exactly_ on the boundary.
        face = axis;
        if( v0>0 ) face += 3;

        size_t ix, iy, iz;
        RANK_TO_INDEX(ii, ix, iy, iz, (nx+(2*num_ghosts)), (ny+(2*num_ghosts)));
        // ix = ii-((nx+2)*(ny+2)+(nx+2)); //ii-12;
        // iy = 1;
        // iz = 1;

        if (face == 0) { ix--; }
        if (face == 1) { iy--; }
        if (face == 2) { iz--; }
        if (face == 3) { ix++; }
        if (face == 4) { iy++; }
        if (face == 5) { iz++; }

        int is_leaving_domain = detect_leaving_domain(face, nx, ny, nz, ix, iy, iz, num_ghosts);
        if (is_leaving_domain >= 0) {
            /*     //std::cout << s << ", " << i << " leaving on " << face << std::endl; */

            /*     //std::cout << */
            /*         //" x " << position_x.access(s,i) << */
            /*         //" y " << position_y.access(s,i) << */
            /*         //" z " << position_z.access(s,i) << */
            /*         //" cell " << cell.access(s,i) << */
            /*         //std::endl; */

            if ( boundary == Boundary::Periodic)
            {
                //std::cout << "face" << std::endl;
                // If we hit the periodic boundary, try and put the article in the right place

                // TODO: we can do this in 1d just fine

                //size_t ix, iy, iz;

                //RANK_TO_INDEX(ii, ix, iy, iz, (nx-1+(2*num_ghosts)), (ny-1+(2*num_ghosts)));
                /* ix = ii-12; */
                /* iy = 1; */
                /* iz = 1; */

                if (is_leaving_domain == 0) { // -1 on x face
                    ix = (nx-1) + num_ghosts;
                }
                else if (is_leaving_domain == 1) { // -1 on y face
                    iy = (ny-1) + num_ghosts;
                }
                else if (is_leaving_domain == 2) { // -1 on z face
                    iz = (nz-1) + num_ghosts;
                }
                else if (is_leaving_domain == 3) { // 1 on x face
                    ix = num_ghosts;
                }
                else if (is_leaving_domain == 4) { // 1 on y face
                    iy = num_ghosts;
                }
                else if (is_leaving_domain == 5) { // 1 on z face
                    iz = num_ghosts;
                }
                /* int updated_ii = VOXEL(ix, iy, iz, */
                /*         nx, */
                /*         ny, */
                /*         nz, */
                /*         num_ghosts); */

            }


        }


        //1D only
        //int updated_ii = ix+(nx+2)*(ny+2) + (nx+2);
        size_t updated_ii = VOXEL(ix, iy, iz, nx, ny, nz, num_ghosts);
        cell.access(s, i) = updated_ii;

        /**/                         // Note: neighbor - g->rangel < 2^31 / 6
        //(&(p->dx))[axis] = -v0;      // Convert coordinate system
        // TODO: this conditional/branching could be better
        if (axis == 0) position_x.access(s, i) = -v0;
        if (axis == 1) position_y.access(s, i) = -v0;
        if (axis == 2) position_z.access(s, i) = -v0;
    }

    return 0; // Return "mover not in use"
}
    
    template <class _accumulator>
    void push(
	      particle_list_t& particles,
	      interpolator_array_t& f0,
	      real_t qdt_2mc,
	      real_t cdt_dx,
	      real_t cdt_dy,
	      real_t cdt_dz,
	      real_t qsp,
	      _accumulator& a0,
	      grid_t* g,
	      const size_t nx,
	      const size_t ny,
	      const size_t nz,
	      const size_t num_ghosts,
	      Boundary boundary, 
	      bool deposit_current = true	      
	      )
    {
    auto position_x = Cabana::slice<PositionX>(particles);
    auto position_y = Cabana::slice<PositionY>(particles);
    auto position_z = Cabana::slice<PositionZ>(particles);

    auto velocity_x = Cabana::slice<VelocityX>(particles);
    auto velocity_y = Cabana::slice<VelocityY>(particles);
    auto velocity_z = Cabana::slice<VelocityZ>(particles);

    auto weight = Cabana::slice<Weight>(particles);
    auto cell = Cabana::slice<Cell_Index>(particles);

    //const real_t qdt_4mc        = -0.5*qdt_2mc; // For backward half rotate
    const real_t one            = 1.;
    const real_t one_third      = 1./3.;
    const real_t two_fifteenths = 2./15.;

    // We prefer making slices out side of the llambda
    auto _ex = Cabana::slice<EX>(f0);
    auto _dexdy = Cabana::slice<DEXDY>(f0);
    auto _dexdz = Cabana::slice<DEXDZ>(f0);
    auto _d2exdydz = Cabana::slice<D2EXDYDZ>(f0);
    auto _ey = Cabana::slice<EY>(f0);
    auto _deydz = Cabana::slice<DEYDZ>(f0);
    auto _deydx = Cabana::slice<DEYDX>(f0);
    auto _d2eydzdx = Cabana::slice<D2EYDZDX>(f0);
    auto _ez = Cabana::slice<EZ>(f0);
    auto _dezdx = Cabana::slice<DEZDX>(f0);
    auto _dezdy = Cabana::slice<DEZDY>(f0);
    auto _d2ezdxdy = Cabana::slice<D2EZDXDY>(f0);
    auto _cbx = Cabana::slice<CBX>(f0);
    auto _dcbxdx = Cabana::slice<DCBXDX>(f0);
    auto _cby = Cabana::slice<CBY>(f0);
    auto _dcbydy = Cabana::slice<DCBYDY>(f0);
    auto _cbz = Cabana::slice<CBZ>(f0);
    auto _dcbzdz = Cabana::slice<DCBZDZ>(f0);

    auto _push =
        KOKKOS_LAMBDA( const int s, const int i )
        {
            auto accumulators_scatter_access = a0.access();

            //for ( int i = 0; i < particle_list_t::vector_length; ++i )
            //{
            // Setup data accessors
            // This may be cleaner if we hoisted it?
            int ii = cell.access(s,i);

            auto ex = _ex(ii);
            auto dexdy = _dexdy(ii);
            auto dexdz = _dexdz(ii);
            auto d2exdydz = _d2exdydz(ii);
            auto ey = _ey(ii);
            auto deydz = _deydz(ii);
            auto deydx = _deydx(ii);
            auto d2eydzdx = _d2eydzdx(ii);
            auto ez = _ez(ii);
            auto dezdx = _dezdx(ii);
            auto dezdy = _dezdy(ii);
            auto d2ezdxdy = _d2ezdxdy(ii);
            auto cbx = _cbx(ii);
            auto dcbxdx = _dcbxdx(ii);
            auto cby = _cby(ii);
            auto dcbydy = _dcbydy(ii);
            auto cbz = _cbz(ii);
            auto dcbzdz = _dcbzdz(ii);

            // Perform push

            // TODO: deal with pm's
            particle_mover_t local_pm = particle_mover_t();

            real_t dx = position_x.access(s,i);   // Load position
            real_t dy = position_y.access(s,i);   // Load position
            real_t dz = position_z.access(s,i);   // Load position

            real_t hax  = qdt_2mc*(    ( ex    + dy*dexdy    ) +
                    dz*( dexdz + dy*d2exdydz ) );
            real_t hay  = qdt_2mc*(    ( ey    + dz*deydz    ) +
                    dx*( deydx + dz*d2eydzdx ) );
            real_t haz  = qdt_2mc*(    ( ez    + dx*dezdx    ) +
                    dy*( dezdy + dx*d2ezdxdy ) );

            //1D only
            //real_t hax = qdt_2mc*ex;
            // real_t hay = 0;
            // real_t haz = 0;

            cbx  = cbx + dx*dcbxdx;             // Interpolate B
            cby  = cby + dy*dcbydy;
            cbz  = cbz + dz*dcbzdz;

            real_t ux = velocity_x.access(s,i);   // Load velocity
            real_t uy = velocity_y.access(s,i);   // Load velocity
            real_t uz = velocity_z.access(s,i);   // Load velocity

            ux  += hax;                               // Half advance E
            uy  += hay;
            uz  += haz;

            real_t v0   = qdt_2mc/sqrtf(one + (ux*ux + (uy*uy + uz*uz)));
            /**/                                      // Boris - scalars
            real_t v1   = cbx*cbx + (cby*cby + cbz*cbz);
            real_t v2   = (v0*v0)*v1;
            real_t v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
            real_t v4   = v3/(one+v1*(v3*v3));
            v4  += v4;
            v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
            v1   = uy + v3*( uz*cbx - ux*cbz );
            v2   = uz + v3*( ux*cby - uy*cbx );
            ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
            uy  += v4*( v2*cbx - v0*cbz );
            uz  += v4*( v0*cby - v1*cbx );
            ux  += hax;                               // Half advance E
            uy  += hay;
            uz  += haz;

            velocity_x.access(s,i) = ux;
            velocity_y.access(s,i) = uy;
            velocity_z.access(s,i) = uz;

            v0   = one/sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));
            /**/                                      // Get norm displacement
            ux  *= cdt_dx;
            uy  *= cdt_dy;
            uz  *= cdt_dz;
            ux  *= v0;
            uy  *= v0;
            uz  *= v0;
            v0   = dx + ux;                           // Streak midpoint (inbnds)
            v1   = dy + uy;
            v2   = dz + uz;
            v3   = v0 + ux;                           // New position
            v4   = v1 + uy;
            real_t v5   = v2 + uz;

            real_t q = weight.access(s,i)*qsp;   // Load charge

            // Check if inbnds
            if(  v3<=one &&  v4<=one &&  v5<=one && -v3<=one && -v4<=one && -v5<=one )
            {

                // Common case (inbnds).  Note: accumulator values are 4 times
                // the total physical charge that passed through the appropriate
                // current quadrant in a time-step


                // Store new position
                position_x.access(s,i) = v3;
                position_y.access(s,i) = v4;
                position_z.access(s,i) = v5;

                dx = v0;                                // Streak midpoint
                dy = v1;
                dz = v2;
                v5 = q*ux*uy*uz*one_third;              // Compute correction

                //real_t* a  = (real_t *)( a0[ii].a );              // Get accumulator

                //1D only
                //_a(ii,0) += q*ux;
                //_a(ii,1) = 0;
                //_a(ii,2) = 0;
                //_a(ii,3) = 0;

                // accumulators_scatter_access(ii, accumulator_var::jx, 0) += 4.0f*q*ux;
                // accumulators_scatter_access(ii, accumulator_var::jx, 1) += 0.0;
                // accumulators_scatter_access(ii, accumulator_var::jx, 2) += 0.0;
                // accumulators_scatter_access(ii, accumulator_var::jx, 3) += 0.0;

                #define CALC_J(X,Y,Z)                                        \
                v4  = q*u##X;   /* v2 = q ux                            */   \
                v1  = v4*d##Y;  /* v1 = q ux dy                         */   \
                v0  = v4-v1;    /* v0 = q ux (1-dy)                     */   \
                v1 += v4;       /* v1 = q ux (1+dy)                     */   \
                v4  = one+d##Z; /* v4 = 1+dz                            */   \
                v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */   \
                v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */   \
                v4  = one-d##Z; /* v4 = 1-dz                            */   \
                v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */   \
                v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */   \
                v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */   \
                v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */   \
                v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */   \
                v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

                CALC_J( x,y,z );
                //std::cout << "Contributing " << v0 << ", " << v1 << ", " << v2 << ", " << v3 << std::endl;
                accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0; // q*ux*(1-dy)*(1-dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1; // q*ux*(1+dy)*(1-dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2; // q*ux*(1-dy)*(1+dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3; // q*ux*(1+dy)*(1+dz);

                // printf("push deposit v0 %e to %d where ux = %e uy = %e and uz = %e \n",
                //         v0, ii, ux, uy, uz);

                CALC_J( y,z,x );
                accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0; // q*ux;
                accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3; // 0.0;

                CALC_J( z,x,y );
                accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0; // q*ux;
                accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3; // 0.0;

                #undef CALC_J

            }
            else
            {                                    // Unlikely
                local_pm.dispx = ux;
                local_pm.dispy = uy;
                local_pm.dispz = uz;

                local_pm.i = s*particle_list_t::vector_length + i; //i + itmp; //p_ - p0;

                // Handle particles that cross cells
                //move_p( position_x, position_y, position_z, cell, _a, q, local_pm,  g,  s, i, nx, ny, nz, num_ghosts, boundary );
                move_p( position_x, position_y, position_z, cell, a0, q, local_pm,  g,  s, i, nx, ny, nz, num_ghosts, boundary );

            }

            //} // end VLEN loop
        };

        Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
            vec_policy( 0, particles.size() );
        Cabana::simd_parallel_for( vec_policy, _push, "push()" );
	
    }
};

    
class Particle_Pusher_Leapfrog_EC
{
public:
template<typename T1, typename T2, typename T3, typename T4, typename T5> KOKKOS_INLINE_FUNCTION int move_p(
        //particle_list_t particles,
        T1& position_x,
        T2& position_y,
        T3& position_z,
        T4& cell,
        T5& a0, // TODO: does this need to be const
        real_t q,
        particle_mover_t& pm,
        const grid_t* g,
        const size_t s,
        const size_t i,
        const size_t nx,
        const size_t ny,
        const size_t nz,
        const size_t num_ghosts,
        const Boundary boundary
    )
{

    auto _asa = a0.access();

    /* // Kernel variables */
    real_t s_dir[3];
    real_t v0, v1, v2, v3; //, v4, v5;
    size_t axis, face;

    for(;;)
    {

        real_t s_midx = position_x.access(s, i);
        real_t s_midy = position_y.access(s, i);
        real_t s_midz = position_z.access(s, i);

        real_t s_dispx = pm.dispx;
        real_t s_dispy = pm.dispy;
        real_t s_dispz = pm.dispz;

        s_dir[0] = (s_dispx>0) ? 1 : -1;
        s_dir[1] = (s_dispy>0) ? 1 : -1;
        s_dir[2] = (s_dispz>0) ? 1 : -1;

        // Compute the twice the fractional distance to each potential
        // streak/cell face intersection.
        v0 = (s_dispx==0) ? 3.4e38 : (s_dir[0]-s_midx)/s_dispx;
        v1 = (s_dispy==0) ? 3.4e38 : (s_dir[1]-s_midy)/s_dispy;
        v2 = (s_dispz==0) ? 3.4e38 : (s_dir[2]-s_midz)/s_dispz;

        // Determine the fractional length and axis of current streak. The
        // streak ends on either the first face intersected by the
        // particle track or at the end of the particle track.
        //
        //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
        //   axis 3        ... streak ends at end of the particle track
        /**/      v3=2,  axis=3;
        if(v0<v3) v3=v0, axis=0;
        if(v1<v3) v3=v1, axis=1;
        if(v2<v3) v3=v2, axis=2;
        v3 *= 0.5;

        // Compute the midpoint and the normalized displacement of the streak
        s_dispx *= v3;
        s_dispy *= v3;
        s_dispz *= v3;
        s_midx += s_dispx;
        s_midy += s_dispy;
        s_midz += s_dispz;

        // Accumulate the streak.  Note: accumulator values are 4 times
        // the total physical charge that passed through the appropriate
        // current quadrant in a time-step
        //v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);

        int ii = cell.access(s, i);

        //a = (real_t *)(a0 + ii);

        //1D only
        // _asa(ii,accumulator_var::jx, 0) += 4.0f*q*s_dispx;
        // _asa(ii,accumulator_var::jx, 1) += 0.0;
        // _asa(ii,accumulator_var::jx, 2) += 0.0;
        // _asa(ii,accumulator_var::jx, 3) += 0.0;
        real_t v4;
        real_t v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);

#define accumulate_j(X,Y,Z)						\
        v4  = q*s_disp##X;    /* v2 = q ux                            */  \
        v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
        v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
        v1 += v4;             /* v1 = q ux (1+dy)                     */  \
        v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
        v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
        v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
        v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
        v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
        v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
        v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
        v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
        v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
        v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \

       /*accumulate_j(x,y,z);
       // printf("move_p deposit v0 %e to %d \n",
       //         v0, ii);
       _asa(ii, accumulator_var::jx, 0) += v0; // q*ux;
       _asa(ii, accumulator_var::jx, 1) += v1; // 0.0;
       _asa(ii, accumulator_var::jx, 2) += v2; // 0.0;
       _asa(ii, accumulator_var::jx, 3) += v3; // 0.0;

       accumulate_j(y,z,x);
       _asa(ii, accumulator_var::jy, 0) += v0; // q*ux;
       _asa(ii, accumulator_var::jy, 1) += v1; // 0.0;
       _asa(ii, accumulator_var::jy, 2) += v2; // 0.0;
       _asa(ii, accumulator_var::jy, 3) += v3; // 0.0;

       accumulate_j(z,x,y);
       _asa(ii, accumulator_var::jz, 0) += v0; // q*ux;
       _asa(ii, accumulator_var::jz, 1) += v1; // 0.0;
       _asa(ii, accumulator_var::jz, 2) += v2; // 0.0;
       _asa(ii, accumulator_var::jz, 3) += v3; // 0.0;
		 */

#   undef accumulate_j

        // Compute the remaining particle displacment
        pm.dispx -= s_dispx;
        pm.dispy -= s_dispy;
        pm.dispz -= s_dispz;

        //printf("%d %d, %d, %f %f",s, i, ii, position_x.access(s, i),position_x.access(s, i));
        // Compute the new particle offset
        position_x.access(s, i) += s_dispx+s_dispx;
        position_y.access(s, i) += s_dispy+s_dispy;
        position_z.access(s, i) += s_dispz+s_dispz;

        //printf(" %f\n",position_x.access(s, i));

        // If an end streak, return success (should be ~50% of the time)

        if( axis==3 ) break;

        // Determine if the particle crossed into a local cell or if it
        // hit a boundary and convert the coordinate system accordingly.
        // Note: Crossing into a local cell should happen ~50% of the
        // time; hitting a boundary is usually a rare event.  Note: the
        // entry / exit coordinate for the particle is guaranteed to be
        // +/-1 _exactly_ for the particle.

        v0 = s_dir[axis];

        // TODO: do branching based on axis

        //(&(p->dx))[axis] = v0; // Avoid roundoff fiascos--put the particle

        // TODO: this conditional could be better
        if (axis == 0) position_x.access(s, i) = v0;
        if (axis == 1) position_y.access(s, i) = v0;
        if (axis == 2) position_z.access(s, i) = v0;

        // _exactly_ on the boundary.
        face = axis;
        if( v0>0 ) face += 3;

        size_t ix, iy, iz;
        RANK_TO_INDEX(ii, ix, iy, iz, (nx+(2*num_ghosts)), (ny+(2*num_ghosts)));
        // ix = ii-((nx+2)*(ny+2)+(nx+2)); //ii-12;
        // iy = 1;
        // iz = 1;

        if (face == 0) { ix--; }
        if (face == 1) { iy--; }
        if (face == 2) { iz--; }
        if (face == 3) { ix++; }
        if (face == 4) { iy++; }
        if (face == 5) { iz++; }

        int is_leaving_domain = detect_leaving_domain(face, nx, ny, nz, ix, iy, iz, num_ghosts);
        if (is_leaving_domain >= 0) {
            /*     //std::cout << s << ", " << i << " leaving on " << face << std::endl; */

            /*     //std::cout << */
            /*         //" x " << position_x.access(s,i) << */
            /*         //" y " << position_y.access(s,i) << */
            /*         //" z " << position_z.access(s,i) << */
            /*         //" cell " << cell.access(s,i) << */
            /*         //std::endl; */

            if ( boundary == Boundary::Periodic)
            {
                //std::cout << "face" << std::endl;
                // If we hit the periodic boundary, try and put the article in the right place

                // TODO: we can do this in 1d just fine

                //size_t ix, iy, iz;

                //RANK_TO_INDEX(ii, ix, iy, iz, (nx-1+(2*num_ghosts)), (ny-1+(2*num_ghosts)));
                /* ix = ii-12; */
                /* iy = 1; */
                /* iz = 1; */

                if (is_leaving_domain == 0) { // -1 on x face
                    ix = (nx-1) + num_ghosts;
                }
                else if (is_leaving_domain == 1) { // -1 on y face
                    iy = (ny-1) + num_ghosts;
                }
                else if (is_leaving_domain == 2) { // -1 on z face
                    iz = (nz-1) + num_ghosts;
                }
                else if (is_leaving_domain == 3) { // 1 on x face
                    ix = num_ghosts;
                }
                else if (is_leaving_domain == 4) { // 1 on y face
                    iy = num_ghosts;
                }
                else if (is_leaving_domain == 5) { // 1 on z face
                    iz = num_ghosts;
                }
                /* int updated_ii = VOXEL(ix, iy, iz, */
                /*         nx, */
                /*         ny, */
                /*         nz, */
                /*         num_ghosts); */

            }


            /*         if ( Parameters::instance().BOUNDARY_TYPE == Boundary::Reflect) */
            /*         { */
            /*             // Hit a reflecting boundary condition.  Reflect the particle */
            /*             // momentum and remaining displacement and keep moving the */
            /*             // particle. */

            /*             //logger << "Reflecting " << s << " " << i << " on axis " << axis << std::endl; */

            /*             //(&(p->ux    ))[axis] = -(&(p->ux    ))[axis]; */
            /*             //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis]; */
            /*             if (axis == 0) */
            /*             { */
            /*                 velocity_x.access(s, i) = -1.0f * velocity_x.access(s, i); */
            /*                 pm.dispx = -1.0f * s_dispx; */
            /*             } */
            /*             if (axis == 1) */
            /*             { */
            /*                 velocity_y.access(s, i) = -1.0f * velocity_y.access(s, i); */
            /*                 pm.dispy = -1.0f * s_dispy; */
            /*             } */
            /*             if (axis == 2) */
            /*             { */
            /*                 velocity_z.access(s, i) = -1.0f * velocity_z.access(s, i); */
            /*                 pm.dispz = -1.0f * s_dispz; */
            /*             } */
            /*             continue; */
            /*         } */
        }

        /*     // TODO: this nieghbor stuff can be removed by going to more simple */
        /*     // boundaries */
        /*     /\* */
        /*     if ( neighbor<g->rangel || neighbor>g->rangeh ) { */
        /*         // Cannot handle the boundary condition here.  Save the updated */
        /*         // particle position, face it hit and update the remaining */
        /*         // displacement in the particle mover. */
        /*         //p->i = 8*p->i + face; */
        /*         cell.access(s, i) = 8 * ii + face; */

        /*         return 1; // Return "mover still in use" */
        /*     } */
        /*     *\/ */
        /*     else { */

        /*     // Crossed into a normal voxel.  Update the voxel index, convert the */
        /*     // particle coordinate system and keep moving the particle. */

        /*     //p->i = neighbor - g->rangel; // Compute local index of neighbor */
        /*     //cell.access(s, i) = neighbor - g->rangel; */
        /*     // TODO: I still need to update the cell we're in */

        //1D only
        //int updated_ii = ix+(nx+2)*(ny+2) + (nx+2);
        size_t updated_ii = VOXEL(ix, iy, iz, nx, ny, nz, num_ghosts);
        cell.access(s, i) = updated_ii;


        /* int updated_ii = VOXEL(ix, iy, iz, */
        /*         nx, */
        /*         ny, */
        /*         nz, */
        /*         num_ghosts); */

        /* cell.access(s, i) = updated_ii; */
        /*     //std::cout << "Moving from cell " << ii << " to " << updated_ii << std::endl; */
        /* } */

        /**/                         // Note: neighbor - g->rangel < 2^31 / 6
        //(&(p->dx))[axis] = -v0;      // Convert coordinate system
        // TODO: this conditional/branching could be better
        if (axis == 0) position_x.access(s, i) = -v0;
        if (axis == 1) position_y.access(s, i) = -v0;
        if (axis == 2) position_z.access(s, i) = -v0;
    }

    return 0; // Return "mover not in use"
}
    
    template <class _accumulator>
    void push(
	      particle_list_t& particles,
	      interpolator_array_t& f0,
	      real_t qdt_2mc,
	      real_t cdt_dx,
	      real_t cdt_dy,
	      real_t cdt_dz,
	      real_t qsp,
	      _accumulator& a0,
	      grid_t* g,
	      const size_t nx,
	      const size_t ny,
	      const size_t nz,
	      const size_t num_ghosts,
	      Boundary boundary, 
	      bool deposit_current = true	      
	      )
    {
    auto position_x = Cabana::slice<PositionX>(particles);
    auto position_y = Cabana::slice<PositionY>(particles);
    auto position_z = Cabana::slice<PositionZ>(particles);

    auto velocity_x = Cabana::slice<VelocityX>(particles);
    auto velocity_y = Cabana::slice<VelocityY>(particles);
    auto velocity_z = Cabana::slice<VelocityZ>(particles);

    auto weight = Cabana::slice<Weight>(particles);
    auto cell = Cabana::slice<Cell_Index>(particles);

    //const real_t qdt_4mc        = -0.5*qdt_2mc; // For backward half rotate
    const real_t one            = 1.;
    const real_t one_third      = 1./3.;
    const real_t two_fifteenths = 2./15.;
	 const real_t one_half		  = 1./2.;

    // We prefer making slices out side of the llambda
    auto _ex = Cabana::slice<EX>(f0);
    auto _dexdy = Cabana::slice<DEXDY>(f0);
    auto _dexdz = Cabana::slice<DEXDZ>(f0);
    auto _d2exdydz = Cabana::slice<D2EXDYDZ>(f0);
    auto _ey = Cabana::slice<EY>(f0);
    auto _deydz = Cabana::slice<DEYDZ>(f0);
    auto _deydx = Cabana::slice<DEYDX>(f0);
    auto _d2eydzdx = Cabana::slice<D2EYDZDX>(f0);
    auto _ez = Cabana::slice<EZ>(f0);
    auto _dezdx = Cabana::slice<DEZDX>(f0);
    auto _dezdy = Cabana::slice<DEZDY>(f0);
    auto _d2ezdxdy = Cabana::slice<D2EZDXDY>(f0);
    auto _cbx = Cabana::slice<CBX>(f0);
    auto _dcbxdx = Cabana::slice<DCBXDX>(f0);
    auto _cby = Cabana::slice<CBY>(f0);
    auto _dcbydy = Cabana::slice<DCBYDY>(f0);
    auto _cbz = Cabana::slice<CBZ>(f0);
    auto _dcbzdz = Cabana::slice<DCBZDZ>(f0);

    auto _push =
        KOKKOS_LAMBDA( const int s, const int i )
        {
            auto accumulators_scatter_access = a0.access();

            //for ( int i = 0; i < particle_list_t::vector_length; ++i )
            //{
            // Setup data accessors
            // This may be cleaner if we hoisted it?
            int ii = cell.access(s,i);

            auto ex = _ex(ii);
            auto dexdy = _dexdy(ii);
            auto dexdz = _dexdz(ii);
            auto d2exdydz = _d2exdydz(ii);
            auto ey = _ey(ii);
            auto deydz = _deydz(ii);
            auto deydx = _deydx(ii);
            auto d2eydzdx = _d2eydzdx(ii);
            auto ez = _ez(ii);
            auto dezdx = _dezdx(ii);
            auto dezdy = _dezdy(ii);
            auto d2ezdxdy = _d2ezdxdy(ii);
            auto cbx = _cbx(ii);
            auto dcbxdx = _dcbxdx(ii);
            auto cby = _cby(ii);
            auto dcbydy = _dcbydy(ii);
            auto cbz = _cbz(ii);
            auto dcbzdz = _dcbzdz(ii);
            /*
               auto ex  = f0.get<EX>(ii);
               auto dexdy  = f0.get<DEXDY>(ii);
               auto dexdz  = f0.get<DEXDZ>(ii);
               auto d2exdydz  = f0.get<D2EXDYDZ>(ii);
               auto ey  = f0.get<EY>(ii);
               auto deydz  = f0.get<DEYDZ>(ii);
               auto deydx  = f0.get<DEYDX>(ii);
               auto d2eydzdx  = f0.get<D2EYDZDX>(ii);
               auto ez  = f0.get<EZ>(ii);
               auto dezdx  = f0.get<DEZDX>(ii);
               auto dezdy  = f0.get<DEZDY>(ii);
               auto d2ezdxdy  = f0.get<D2EZDXDY>(ii);
               auto cbx  = f0.get<CBX>(ii);
               auto dcbxdx   = f0.get<DCBXDX>(ii);
               auto cby  = f0.get<CBY>(ii);
               auto dcbydy  = f0.get<DCBYDY>(ii);
               auto cbz  = f0.get<CBZ>(ii);
               auto dcbzdz  = f0.get<DCBZDZ>(ii);
               */

            // Perform push

            // TODO: deal with pm's
            particle_mover_t local_pm = particle_mover_t();

            real_t dx = position_x.access(s,i);   // Load position
            real_t dy = position_y.access(s,i);   // Load position
            real_t dz = position_z.access(s,i);   // Load position

            real_t hax  = qdt_2mc*(    ( ex    + dy*dexdy    ) +  // interpolate E and multiply by q*dt/(2*m*c)
                    dz*( dexdz + dy*d2exdydz ) );
            real_t hay  = qdt_2mc*(    ( ey    + dz*deydz    ) +
                    dx*( deydx + dz*d2eydzdx ) );
            real_t haz  = qdt_2mc*(    ( ez    + dx*dezdx    ) +
                    dy*( dezdy + dx*d2ezdxdy ) );

            //1D only
            //real_t hax = qdt_2mc*ex;
            // real_t hay = 0;
            // real_t haz = 0;

            cbx  = cbx + dx*dcbxdx;             // Interpolate B
            cby  = cby + dy*dcbydy;
            cbz  = cbz + dz*dcbzdz;

            real_t ux = velocity_x.access(s,i);   // Load velocity
            real_t uy = velocity_y.access(s,i);   // Load velocity
            real_t uz = velocity_z.access(s,i);   // Load velocity

            ux  += hax;                               // Half advance E
            uy  += hay;
            uz  += haz;

            real_t v0   = qdt_2mc/sqrtf(one + (ux*ux + (uy*uy + uz*uz)));
            /**/                                      // Boris - scalars
            real_t v1   = cbx*cbx + (cby*cby + cbz*cbz);
            real_t v2   = (v0*v0)*v1;
            real_t v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
            real_t v4   = v3/(one+v1*(v3*v3));
            v4  += v4;
            v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
            v1   = uy + v3*( uz*cbx - ux*cbz );
            v2   = uz + v3*( ux*cby - uy*cbx );
            ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation 
            uy  += v4*( v2*cbx - v0*cbz );
            uz  += v4*( v0*cby - v1*cbx );
            ux  += hax;                               // Half advance E
            uy  += hay;
            uz  += haz;

            velocity_x.access(s,i) = ux;
            velocity_y.access(s,i) = uy;
            velocity_z.access(s,i) = uz;

            v0   = one; ///sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));
            /**/                                      // Get norm displacement
            ux  *= cdt_dx;
            uy  *= cdt_dy;
            uz  *= cdt_dz;
            ux  *= v0;
            uy  *= v0;
            uz  *= v0;
            v0   = dx + ux;                           // Streak midpoint (inbnds)
            v1   = dy + uy;
            v2   = dz + uz;
            v3   = v0 + ux;                           // New position
            v4   = v1 + uy;
            real_t v5   = v2 + uz;

				real_t new_x = v3; 
				real_t new_y = v4; 
				real_t new_z = v5;

            real_t q = weight.access(s,i)*qsp;   // Load charge
                
					 // moving calculation of j before the updating of position for semi-implicit purposes //
					 #define CALC_J(X,Y,Z)                                        \
                v4  = q*u##X;   /* v2 = q ux                            */   \
                v1  = v4*d##Y;  /* v1 = q ux dy                         */   \
                v0  = v4-v1;    /* v0 = q ux (1-dy)                     */   \
                v1 += v4;       /* v1 = q ux (1+dy)                     */   \
                v4  = one+d##Z; /* v4 = 1+dz                            */   \
                v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */   \
                v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */   \
                v4  = one-d##Z; /* v4 = 1-dz                            */   \
                v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */   \
                v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */   
					
					if ( deposit_current ) 
					{
                CALC_J( x,y,z );
                //std::cout << "Contributing " << v0 << ", " << v1 << ", " << v2 << ", " << v3 << std::endl;
                accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0; // q*ux*(1-dy)*(1-dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1; // q*ux*(1+dy)*(1-dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2; // q*ux*(1-dy)*(1+dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3; // q*ux*(1+dy)*(1+dz);

                // printf("push deposit v0 %e to %d where ux = %e uy = %e and uz = %e \n",
                //         v0, ii, ux, uy, uz);

                CALC_J( y,z,x );
                accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0; // q*ux;
                accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3; // 0.0;

                CALC_J( z,x,y );
                accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0; // q*ux;
                accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3; // 0.0;
					}

                #undef CALC_J

            // Check if inbnds
            //if(  v3<=one &&  v4<=one &&  v5<=one && -v3<=one && -v4<=one && -v5<=one )
				if( new_x<=one && new_y<=one && new_z<=one && -new_x<=one && -new_y <=one && -new_z<=one )
            {

                // Common case (inbnds).  Note: accumulator values are 4 times
                // the total physical charge that passed through the appropriate
                // current quadrant in a time-step


                // Store new position
                position_x.access(s,i) = new_x; //v3;
                position_y.access(s,i) = new_y; //v4;
                position_z.access(s,i) = new_z; //v5;

                //dx = v0;                                // Streak midpoint
                //dy = v1;
                //dz = v2;
                //v5 = q*ux*uy*uz*one_third;              // Compute correction

                //real_t* a  = (real_t *)( a0[ii].a );              // Get accumulator

                //1D only
                //_a(ii,0) += q*ux;
                //_a(ii,1) = 0;
                //_a(ii,2) = 0;
                //_a(ii,3) = 0;

                // accumulators_scatter_access(ii, accumulator_var::jx, 0) += 4.0f*q*ux;
                // accumulators_scatter_access(ii, accumulator_var::jx, 1) += 0.0;
                // accumulators_scatter_access(ii, accumulator_var::jx, 2) += 0.0;
                // accumulators_scatter_access(ii, accumulator_var::jx, 3) += 0.0;
					 
					 
                //#define CALC_J(X,Y,Z)                                        \
                v4  = q*u##X;   /* v2 = q ux                            */   \
                v1  = v4*d##Y;  /* v1 = q ux dy                         */   \
                v0  = v4-v1;    /* v0 = q ux (1-dy)                     */   \
                v1 += v4;       /* v1 = q ux (1+dy)                     */   \
                v4  = one+d##Z; /* v4 = 1+dz                            */   \
                v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */   \
                v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */   \
                v4  = one-d##Z; /* v4 = 1-dz                            */   \
                v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */   \
                v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */   \
                v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */   \
                v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */   \
                v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */   \
                v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
					 /*
                CALC_J( x,y,z );
                //std::cout << "Contributing " << v0 << ", " << v1 << ", " << v2 << ", " << v3 << std::endl;
                accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0; // q*ux*(1-dy)*(1-dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1; // q*ux*(1+dy)*(1-dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2; // q*ux*(1-dy)*(1+dz);
                accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3; // q*ux*(1+dy)*(1+dz);

                // printf("push deposit v0 %e to %d where ux = %e uy = %e and uz = %e \n",
                //         v0, ii, ux, uy, uz);

                CALC_J( y,z,x );
                accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0; // q*ux;
                accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3; // 0.0;

                CALC_J( z,x,y );
                accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0; // q*ux;
                accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2; // 0.0;
                accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3; // 0.0;

                #undef CALC_J
					 */
            }
            else
            {                                    // Unlikely
                local_pm.dispx = ux;
                local_pm.dispy = uy;
                local_pm.dispz = uz;

                local_pm.i = s*particle_list_t::vector_length + i; //i + itmp; //p_ - p0;

                // Handle particles that cross cells
                //move_p( position_x, position_y, position_z, cell, _a, q, local_pm,  g,  s, i, nx, ny, nz, num_ghosts, boundary );
                move_p( position_x, position_y, position_z, cell, a0, q, local_pm,  g,  s, i, nx, ny, nz, num_ghosts, boundary );

                // TODO: renable this
                //if ( move_p( p0, local_pm, a0, g, qsp ) ) { // Unlikely
                //if ( move_p( particles, local_pm, a0, g, qsp, s, i ) ) { // Unlikely
                //if( nm<max_nm ) {
                //pm[nm++] = local_pm[0];
                //}
                //else {
                //ignore++;                 // Unlikely
                //} // if
                //} // if

                /* // Copied from VPIC Kokkos for reference
                   if( move_p_kokkos( k_particles, k_local_particle_movers,
                   k_accumulators_sa, g, qsp ) ) { // Unlikely
                   if( k_nm(0)<max_nm ) {
                   nm = int(Kokkos::atomic_fetch_add( &k_nm(0), 1 ));
                   if (nm >= max_nm) Kokkos::abort("overran max_nm");
                   copy_local_to_pm(nm);
                   }
                   }
                   */
            }

            //} // end VLEN loop
        };

        Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
            vec_policy( 0, particles.size() );
        Cabana::simd_parallel_for( vec_policy, _push, "push()" );
	
    }
};
#endif
