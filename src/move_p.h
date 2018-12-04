#ifndef pic_move_p_h
#define pic_move_p_h

#include <types.h>

// TODO: add namespace etc?
// TODO: port this to cabana syntax
int move_p(
        particle_list_t particles,
        particle_mover_t* pm,
        accumulator_t* a0,
        const grid_t* g,
        const float qsp
    )
{

    // Grab accessors
    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto charge = particles.slice<Charge>();
    auto cell = particles.slice<Cell_Index>();


    // Kernel variables
    float s_dir[3];
    float v0, v1, v2, v3, v4, v5, q;
    int axis, face;
    int64_t neighbor;
    float *a;

    //particle_t* p = p0 + pm->i;
    int index = pm->i;

    q = qsp*p->w;

    for(;;) {
        /*
        s_midx = p->dx;
        s_midy = p->dy;
        s_midz = p->dz;
        */

        float s_midx = position_x.access(index);
        float s_midy = position_y.access(index);
        float s_midz = position_z.access(index);

        float s_dispx = pm->dispx;
        float s_dispy = pm->dispy;
        float s_dispz = pm->dispz;

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
        v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);

        int ii = cell.access(i);
        a = (float *)(a0 + ii);

#   define accumulate_j(X,Y,Z)                                        \
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
        a[0] += v0;                                                       \
        a[1] += v1;                                                       \
        a[2] += v2;                                                       \
        a[3] += v3
        accumulate_j(x,y,z); a += 4;
        accumulate_j(y,z,x); a += 4;
        accumulate_j(z,x,y);
#   undef accumulate_j

        // Compute the remaining particle displacment
        pm->dispx -= s_dispx;
        pm->dispy -= s_dispy;
        pm->dispz -= s_dispz;

        // Compute the new particle offset
        /*
        p->dx += s_dispx+s_dispx;
        p->dy += s_dispy+s_dispy;
        p->dz += s_dispz+s_dispz;
        */
        position_x.access(index) += s_dispx+s_dispx;
        position_y.access(index) += s_dispy+s_dispy;
        position_z.access(index) += s_dispz+s_dispz;


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

        (&(p->dx))[axis] = v0; // Avoid roundoff fiascos--put the particle

        // _exactly_ on the boundary.
        face = axis; if( v0>0 ) face += 3;
        neighbor = g->neighbor[ 6* ii + face ];

        if( UNLIKELY( neighbor==reflect_particles ) ) {
            // Hit a reflecting boundary condition.  Reflect the particle
            // momentum and remaining displacement and keep moving the
            // particle.
            (&(p->ux    ))[axis] = -(&(p->ux    ))[axis];
            (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
            continue;
        }

        if( UNLIKELY( neighbor<g->rangel || neighbor>g->rangeh ) ) {
            // Cannot handle the boundary condition here.  Save the updated
            // particle position, face it hit and update the remaining
            // displacement in the particle mover.
            //p->i = 8*p->i + face;
            cell.access(i) = 8 * ii + face;

            return 1; // Return "mover still in use"
        }

        // Crossed into a normal voxel.  Update the voxel index, convert the
        // particle coordinate system and keep moving the particle.

        //p->i = neighbor - g->rangel; // Compute local index of neighbor
        cell.access(i) = neighbor - g->rangel;

        /**/                         // Note: neighbor - g->rangel < 2^31 / 6
        (&(p->dx))[axis] = -v0;      // Convert coordinate system
    }

    return 0; // Return "mover not in use"
}

#endif // move_p
