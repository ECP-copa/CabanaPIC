#ifndef particle_bc_h
#define particle_bc_h

// I make no claims that this is a sensible way to do this.. I just want it working ASAP
// THIS DEALS WITH GHOSTS ITSELF
KOKKOS_INLINE_FUNCTION int detect_leaving_domain( size_t, size_t nx, size_t ny, size_t nz, size_t ix, size_t iy, size_t iz, size_t )
{

    //RANK_TO_INDEX(ii, ix, iy, iz, (nx+(2*num_ghosts)), (ny+(2*num_ghosts)));
    //std::cout << "i " << ii << " ix " << ix << " iy " << iy << " iz " << iz << std::endl;

    //printf("nx,ny,nz=%ld,%ld,%ld, i=%ld, ix=%ld, iy=%ld, iz=%ld\n",nx,ny,nz,ii,ix,iy,iz);

    int leaving = -1;

    if (ix == 0)
    {
        leaving = 0;
    }

    if (iy == 0)
    {
        leaving = 1;
    }

    if (iz == 0)
    {
        leaving = 2;
    }

    if (ix == nx+1)
    {
        leaving = 3;
    }

    if (iy == ny+1)
    {
        leaving = 4;
    }

    if (iz == nz+1)
    {
        leaving = 5;
    }


    // if(leaving>=0){
    //   printf("%d %d %d %d\n", ix,iy,iz,leaving);
    // }
    return leaving;
}

#endif
