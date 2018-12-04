#ifndef GRID_T
#define GRID_T

enum grid_enums {

  // Phase 2 boundary conditions
  anti_symmetric_fields = -1, // E_tang = 0
  pec_fields            = -1,
  metal_fields          = -1,
  symmetric_fields      = -2, // B_tang = 0, B_norm = 0
  pmc_fields            = -3, // B_tang = 0, B_norm floats
  absorb_fields         = -4, // Gamma = 0

  // Phase 3 boundary conditions
  reflect_particles = -1, // Cell boundary should reflect particles
  absorb_particles  = -2  // Cell boundary should absorb particles

  // Symmetry in the field boundary conditions refers to image charge
  // sign
  //
  // Anti-symmetric -> Image charges are opposite signed (ideal metal)
  //                   Boundary rho/j are accumulated over partial voxel+image
  // Symmetric      -> Image charges are same signed (symmetry plane or pmc)
  //                   Boundary rho/j are accumulated over partial voxel+image
  // Absorbing      -> No image charges
  //                   Boundary rho/j are accumulated over partial voxel only
  //
  // rho     -> Anti-symmetric      | rho     -> Symmetric
  // jf_tang -> Anti-symmetric      | jf_tang -> Symmetric
  // E_tang  -> Anti-symmetric      | E_tang  -> Symmetric
  // B_norm  -> Anti-symmetric + DC | B_norm  -> Symmetric      (see note)
  // B_tang  -> Symmetric           | B_tang  -> Anti-symmetric
  // E_norm  -> Symmetric           | E_norm  -> Anti-symmetric (see note)
  // div B   -> Symmetric           | div B   -> Anti-symmetric
  // 
  // Note: B_norm is tricky. For a symmetry plane, B_norm on the
  // boundary must be zero as there are no magnetic charges (a
  // non-zero B_norm would imply an infinitesimal layer of magnetic
  // charge). However, if a symmetric boundary is interpreted as a
  // perfect magnetic conductor, B_norm could be present due to
  // magnetic conduction surface charges. Even though there are no
  // bulk volumetric magnetic charges to induce a surface magnetic
  // charge, I think that radiation/waveguide modes/etc could (the
  // total surface magnetic charge in the simulation would be zero
  // though). As a result, symmetric and pmc boundary conditions are
  // treated separately. Symmetric and pmc boundaries are identical
  // except the symmetric boundaries explicitly zero boundary
  // B_norm. Note: anti-symmetric and pec boundary conditions would
  // have the same issue if norm E was located directly on the
  // boundary. However, it is not so this problem does not arise.
  //
  // Note: Absorbing boundary conditions make no effort to clean
  // divergence errors on them. They assume that the ghost div b is
  // zero and force the surface div e on them to be zero. This means
  // ghost norm e can be set to any value on absorbing boundaries.

};

typedef struct grid {

  // System of units
  float dt, cvac, eps0;

  // Time stepper.  The simulation time is given by
  // t = g->t0 + (double)g->dt*(double)g->step
  int64_t step;             // Current timestep
  double t0;                // Simulation time corresponding to step 0

  // Phase 2 grid data structures 
  float x0, y0, z0;         // Min corner local domain (must be coherent)
  float x1, y1, z1;         // Max corner local domain (must be coherent)
  int   nx, ny, nz;         // Local voxel mesh resolution.  Voxels are
                            // indexed FORTRAN style 0:nx+1,0:ny+1,0:nz+1
                            // with voxels 1:nx,1:ny,1:nz being non-ghost
                            // voxels.
  float dx, dy, dz, dV;     // Cell dimensions and volume (CONVENIENCE ...
                            // USE x0,x1 WHEN DECIDING WHICH NODE TO USE!)
  float rdx, rdy, rdz, r8V; // Inverse voxel dimensions and one over
                            // eight times the voxel volume (CONVENIENCE)
  int   sx, sy, sz, nv;     // Voxel indexing x-, y-,z- strides and the
                            // number of local voxels (including ghosts,
                            // (nx+2)(ny+2)(nz+2)), (CONVENIENCE)
  int   bc[27];             // (-1:1,-1:1,-1:1) FORTRAN indexed array of
                            // boundary conditions to apply at domain edge
                            // 0 ... nproc-1 ... comm boundary condition
                            // <0 ... locally applied boundary condition

  // Phase 3 grid data structures
  // NOTE: VOXEL INDEXING LIMITS NUMBER OF VOXELS TO 2^31 (INCLUDING
  // GHOSTS) PER NODE.  NEIGHBOR INDEXING FURTHER LIMITS TO
  // (2^31)/6.  BOUNDARY CONDITION HANDLING LIMITS TO 2^28 PER NODE
  // EMITTER COMPONENT ID INDEXING FURTHER LIMITS TO 2^26 PER NODE.
  // THE LIMIT IS 2^63 OVER ALL NODES THOUGH.
  int64_t* range;
                          // (0:nproc) indexed array giving range of
                          // global indexes of voxel owned by each
                          // processor.  Replicated on each processor.
                          // (range[rank]:range[rank+1]-1) are global
                          // voxels owned by processor "rank".  Note:
                          // range[rank+1]-range[rank] <~ 2^31 / 6

  int64_t* neighbor;
                          // (0:5,0:local_num_voxel-1) FORTRAN indexed
                          // array neighbor(0:5,lidx) are the global
                          // indexes of neighboring voxels of the
                          // voxel with local index "lidx".  Negative
                          // if neighbor is a boundary condition.

  int64_t rangel, rangeh; // Redundant for move_p performance reasons:
                          //   rangel = range[rank]
                          //   rangeh = range[rank+1]-1.
                          // Note: rangeh-rangel <~ 2^26

  // Nearest neighbor communications ports
  //mp_t * mp;

} grid_t;

#endif // header guard
