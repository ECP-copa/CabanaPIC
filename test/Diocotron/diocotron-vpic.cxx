// Magnetic reconnection in a Harris equilibrium thin current sheet
//
// This input deck reproduces the PIC simulations found in:
//   William Daughton. "Nonlinear dynamics of thin current sheets." Phys.
//   Plasmas. 9(9): 3668-3678. September 2002.
//
// This input deck was written by:
//   Kevin J Bowers, Ph.D.
//   Plasma Physics Group (X-1)
//   Applied Physics Division
//   Los Alamos National Lab
// August 2003      - original version
// October 2003     - heavily revised to utilize input deck syntactic sugar
// March/April 2004 - rewritten for domain decomposition V4PIC
 
// If you want to use global variables (for example, to store the dump
// intervals for your diagnostics section), it must be done in the globals
// section. Variables declared the globals section will be preserved across
// restart dumps. For example, if the globals section is:
//   begin_globals {
//     double variable;
//   } end_globals
// the double "variable" will be visible to other input deck sections as
// "global->variable". Note: Variables declared in the globals section are set
// to zero before the user's initialization block is executed. Up to 16K
// of global variables can be defined.
 
begin_globals {
  double energies_interval;
  double fields_interval;
  double ehydro_interval;
  double ihydro_interval;
  double eparticle_interval;
  double iparticle_interval;
  double restart_interval;
  int fft_ex_interval;
  int part_loc_interval;
};
 
begin_initialization {
  // At this point, there is an empty grid and the random number generator is
  // seeded with the rank. The grid, materials, species need to be defined.
  // Then the initial non-zero fields need to be loaded at time level 0 and the
  // particles (position and momentum both) need to be loaded at time level 0.
 
  // Arguments can be passed from the command line to the input deck
  // if( num_cmdline_arguments!=3 ) {
  //   sim_log( "Usage: " << cmdline_argument[0] << " mass_ratio seed" );
  //   abort(0);
  // }
  seed_entropy(1); //seed_entropy( atoi( cmdline_argument[2] ) );
 
  // Diagnostic messages can be passed written (usually to stderr)
  sim_log( "Computing simulation parameters");
 
  // Define the system of units for this problem (natural units)
  //double L    = 1; // Length normalization (sheet thickness)
  double de   = 1; // Length normalization (electron inertial length)
  double ec   = 1; // Charge normalization
  double me   = 1; // Mass normalization
  double c    = 1; // Speed of light
  double eps0 = 1; // Permittivity of space
 
  // Physics parameters
  double mi_me   = 1; //1836; //25; //atof(cmdline_argument[1]); // Ion mass / electron mass
  double vthe = 0.1; //0.0424264068711;       //0.424264068711;       // Electron thermal velocity
  double vthi = 0; //0.0424264068711;       //0.424264068711;       // Ion thermal velocity 
  double vthex =0; //0.0141421356237;      // 0.141421356237;      // Electron thermal velocity in x-direction.
  double vthix =0; //0.0141421356237;      // 0.141421356237;Ion thermal velocity in x-direction.
  double b0 = sqrt(20.0); //guide filed B
  
  double v0e   = 0; //*4.0; //*4; //drift velocity
  double v0i   = 0; //*4.0; //*4; //drift velocity
  
  double n0      = 1.0;    //  Background plasma density

  double tauwpe    = 200000;    // simulation wpe's to run

  // Numerical parameters
  double topology_x = nproc();  // Number of domains in x, y, and z
  double topology_y = 1; 
  double topology_z = 1;  // For load balance, best to keep "1" or "2" for Harris sheet
  double a = 0.1;
  double Lx        = 1.6; //*4.0; //4.62*de; //6.7*de; //10.0*de;  // How big should the box be in the x direction
  double Ly        = 1.6; //0.0721875*de;  // How big should the box be in the y direction
  double Lz        = 1; //0.0721875*de;  // How big should the box be in the z direction
  double nx        = 64;    // Global resolution in the x direction
  double ny        = 64;    // Global resolution in the y direction
  double nz        = 1; //32;     // Global resolution in the z direction
  double nppc      = 40; //125; //800; //200; //2048; //1024; //128;    // Average number of macro particles per cell (both species combined!)
  double cfl_req   = 0.99; //0.99;  // How close to Courant should we try to run
  double wpedt_max = 0.36;  // How big a timestep is allowed if Courant is not too restrictive
  double damp      = 0.0; // Level of radiation damping

 
  // Derived quantities
  double mi = me*mi_me;             // Ion mass
  double wpe  = c/de;               // electron plasma frequency
  double wpi  = wpe/sqrt(mi_me);    // ion plasma frequency
  double di   = c/wpi;              // ion inertial length
 
  double hx = Lx/nx;
  double hy = Ly/ny;
  double hz = Lz/nz;

  double Npe = n0*2.0*a*Lz*Lx;    // Number physical electrons.
  double Npi = Npe;            // Number of physical ions in box
  //  double nyy = 2.0*a/hy; 
  //  double Ne  = nppc*nx*nyy*nz;  // total macro electrons in box
  double Ne = nppc*nx*ny/8;
  Ne = trunc_granular(Ne,nproc());
  double Ni   = Ne;                                   // Total macro ions in box
  //double qe = -ec*Npe/Ne;  // Charge per macro electron
  //double qi = -ec*Npe/Ne;  // Charge per macro electron       

  
  double we   = Npe/Ne;                               // Weight of a macro electron
  double wi   = Npi/Ni;                               // Weight of a macro ion
 

  // Determine the timestep
  double dg = courant_length(Lx,Ly,Lz,nx,ny,nz);      // Courant length
  double dt = cfl_req*dg/c;                           // Courant limited time step
  // printf("in harris.cxx: dt=%.7f\n",  dt);
  // exit(1);
  if( wpe*dt>wpedt_max ) dt=wpedt_max/wpe;            // Override time step if plasma frequency limited
 
  ////////////////////////////////////////
  // Setup high level simulation parmeters
 
  num_step             = 20000; //20000; //10000; //1200; // int(tauwpe/(wpe*dt));
  status_interval      = 0; //2000;
  sync_shared_interval = 0; //status_interval;
  clean_div_e_interval = 0; //turn off cleaning (GY)//status_interval;
  clean_div_b_interval = 0; //status_interval; //(GY) 
 
  global->energies_interval  = 100; //status_interval;
  global->fields_interval    = 0;
  global->fft_ex_interval    = 100;
  global->part_loc_interval    = 100;
  global->ehydro_interval    = 0; //status_interval;
  global->ihydro_interval    = 0; //status_interval;
  global->eparticle_interval = 0; //status_interval; // Do not dump
  global->iparticle_interval = 0; //status_interval; // Do not dump
  global->restart_interval   = 0; //status_interval; // Do not dump
 
  ///////////////////////////
  // Setup the space and time
 
  // Setup basic grid parameters
  define_units( c, eps0 );
  define_timestep( dt );
  grid->dx = hx;
  grid->dy = hy;
  grid->dz = hz;
  grid->dt = dt;
  grid->cvac = c;
  //grid->damp = damp;
  double gx0  =-0.5*Lx;
  double gy0  =-0.5*Ly;
  double gz0  =-0.5*Lz;
  double gx1  = 0.5*Lx;
  double gy1  = 0.5*Ly;
  double gz1  = 0.5*Lz;

   define_periodic_grid(  gx0, gy0, gz0,    // Low corner
   			  gx1, gy1, gz1,    // High corner
   			  nx, ny, nz,             // Resolution
   			  topology_x, topology_y, topology_z); // Topology
  // Parition a periodic box among the processors sliced uniformly along y
  // define_periodic_grid( -0.5*Lx, 0, 0,    // Low corner
  //                        0.5*Lx, Ly, Lz,  // High corner
  //                        nx, ny, nz,      // Resolution
  //                        1, nproc(), 1 ); // Topology
  // define_periodic_grid(  0, -0.5*Ly, -0.5*Lz,    // Low corner
  // 			  Lx, 0.5*Ly, 0.5*Lz,     // High corner
  // 			  nx, ny, nz,             // Resolution
  // 			  topology_x, topology_y, topology_z); // Topology

  //   printf("in harris.cxx: g->neighbor[6*265]=%jd\n",  grid->neighbor[6*265]);
  // Override some of the boundary conditions to put a particle reflecting
  // perfect electrical conductor on the -y and +y boundaries
   // set_domain_field_bc( BOUNDARY(0,-1,0), pec_fields );
   // set_domain_field_bc( BOUNDARY(0, 1,0), pec_fields );
   // set_domain_particle_bc( BOUNDARY(0,-1,0), reflect_particles );
   // set_domain_particle_bc( BOUNDARY(0, 1,0), reflect_particles );
 
  define_material( "vacuum", 1 );
  // Note: define_material defaults to isotropic materials with mu=1,sigma=0
  // Tensor electronic, magnetic and conductive materials are supported
  // though. See "shapes" for how to define them and assign them to regions.
  // Also, space is initially filled with the first material defined.
 
  // If you pass NULL to define field array, the standard field array will
  // be used (if damp is not provided, no radiation damping will be used).
  define_field_array( NULL, damp );
 
  ////////////////////
  // Setup the species
 
  // Allow 50% more local_particles in case of non-uniformity
  // VPIC will pick the number of movers to use for each species
  // Both species use out-of-place sorting
  // species_t * ion      = define_species( "ion",       ec, mi, 1.5*Ni/nproc(), -1, 40, 1 );
  // species_t * electron = define_species( "electron", -ec, me, 1.5*Ne/nproc(), -1, 20, 1 );
  //species_t *electron = define_species("electron",-ec,me,2.4*Ne/nproc(),-1,25,0);
  //species_t *ion      = define_species("ion",      ec,mi,2.4*Ne/nproc(),-1,25,0);

  species_t *electron = define_species("electron",-ec,me,3*Ne/nproc(),-1,0,0); //turn off sorting (GY)
  //  species_t *ion      = define_species("ion",     -ec,mi,3*Ne/nproc(),-1,0,0); //(GY) 
 
  ///////////////////////////////////////////////////
  // Log diagnostic information about this simulation
 
  sim_log( "***********************************************" );
  sim_log ( "mi/me = " << mi_me );
  sim_log ( "tauwpe = " << tauwpe );
  sim_log ( "num_step = " << num_step );
  sim_log ( "xmin " << grid->x0 );
  sim_log ( "xmax " << grid->x1 );
  sim_log ( "ymin " << grid->y0 );
  sim_log ( "ymax " << grid->y1 );
  sim_log ( "zmin " << grid->z0 );
  sim_log ( "zmax " << grid->z1 );
  sim_log ( "Lx/di = " << Lx/di );
  sim_log ( "Lx/de = " << Lx/de );
  sim_log ( "Ly/di = " << Ly/di );
  sim_log ( "Ly/de = " << Ly/de );
  sim_log ( "Lz/di = " << Lz/di );
  sim_log ( "Lz/de = " << Lz/de );
  sim_log ( "nx = " << nx );
  sim_log ( "ny = " << ny );
  sim_log ( "nz = " << nz ); 
  sim_log ( "damp = " << damp );
  sim_log ( "courant = " << c*dt/dg );
  sim_log ( "nproc = " << nproc ()  );
  sim_log ( "nppc = " << nppc );
  sim_log ( " b0 = " << b0 );
  sim_log ( " di = " << di );
  sim_log ( " Ne = " << Ne );
  sim_log ( "total # of particles = " << Ne );
  sim_log ( "dt*wpe = " << wpe*dt ); 
  sim_log ( "dx/de = " << Lx/(de*nx) );
  sim_log ( "dy/de = " << Ly/(de*ny) );
  sim_log ( "dz/de = " << Lz/(de*nz) );
  sim_log ( "dx/debye = " << (Lx/nx)/(vthe/wpe)  );
  sim_log ( "n0 = " << n0 );
  sim_log ( "vthi/c = " << vthi/c );
  sim_log ( "vthe/c = " << vthe/c );
  sim_log ( "we = " << we );
  sim_log( "" );
 
  ////////////////////////////
  // Load fields and particles
 
  // sim_log( "Loading fields" );
  set_region_field( y>a, 0, -a, 0,                    // Electric field
		    0, 0, b0 ); // Magnetic field
  set_region_field( y>=-a&&y<=a, 0, -y, 0,                    // Electric field
		    0, 0, b0 ); // Magnetic field
  set_region_field( y<-a, 0, a, 0,                    // Electric field
		    0, 0, b0 ); // Magnetic field
 
  // set_region_field( everywhere, 0, 0, 0,                    // Electric field
  //                   0, -sn*b0*tanh(x/L), cs*b0*tanh(x/L) ); // Magnetic field
  // Note: everywhere is a region that encompasses the entire simulation
  // In general, regions are specied as logical equations (i.e. x>0 && x+y<2)
  
  sim_log( "Loading particles" );
 
  // Do a fast load of the particles
  //seed_rand( rng_seed*nproc() + rank() );  //Generators desynchronized
  double xmin = grid->x0 , xmax = grid->x1;
  double ymin = grid->y0 , ymax = grid->y1;
  double zmin = grid->z0 , zmax = grid->z1;

  // printf("rank=%d,xmin=%.14f,xmax=%.14f,dx=%.14f,nx=%d\n",rank(),grid->x0,grid->x1,grid->dx,grid->nx);
  // printf("rank=%d,xmin=%.14f,xmax=%.14f\n",rank(),xmin,xmax);
  // printf("rank=%d,xmin=%.14f,ymin=%.14f,zmin=%.14f\n",rank(),xmin,ymin,zmin);
  // printf("rank=%d,xmax=%.14f,ymax=%.14f,zmax=%.14f\n",rank(),xmax,ymax,zmax);
  // printf("rank=%d,gx0=%.14f,gy0=%.14f,gz0=%.14f\n",rank(),gx0,gy0,gz0);
  // printf("rank=%d,gx1=%.14f,gy1=%.14f,gz1=%.14f\n",rank(),gx1,gy1,gz1);
#define rand_float(min, max) (min + (max-min)*rand()/RAND_MAX)
  
  sim_log( "-> Uniform Bi-Maxwellian" );
  int seed = 1;
  int seedn= 1;
  double n1,n2,n3,n4,n5,n6;
  int signx,signy,signz;
  int Nlocal=0;
  double dxp=Lx/Ne;
  int ip=0;
  repeat ( Ne ) {
    //double x = (ip+0.5)*dxp;
    ip++;
    //double x = uniform2( xmin, xmax , seed );
    //double y = uniform2( -a, a , seed );
    double z = 0; //uniform2( gz0, gz1 , seed );
    // double x = uniform( rng(0), xmin, xmax );
    // double y = uniform( rng(0), -a, a );
    double x = rand_float(xmin,xmax);
    double y = rand_float(-a,a);
    
   // double z = uniform( rng(0), zmin, zmax );
   // double x = uniform( rng(0), 0, 1 );
   // double y = uniform( rng(0), 0, 1 );
   // double z = uniform( rng(0), 0, 1 );
    //n1 = normal(rng(0),0,vthe );
    //n2 = normal(rng(0),0,vthe );
    //n3 = normal(rng(0),0,vthe );
   // n4 = normal(rng(0),v0i,vthix);
   // n5 = normal(rng(0),0,vthi );
   // n6 = normal(rng(0),0,vthi );
   n1 = 0;
   n2 = 0;
   n3 = 0;

   //mpi reproducing serial 
   if(x<xmin||x>xmax||y<ymin||y>ymax||z<zmin||z>zmax) continue;
 
   double na = 0; //1e-1*sin(2.0*3.1415926*x/Lx);
   inject_particle( electron, x*(1+na), y, z,
		    n1,
		    n2,
		    n3,we, 0, 0);



   //   inject_particle( ion, x, y, z,
   //		    n4*(1.0+na),
   //		    n5,
   //		    n6,wi, 0 ,0 );
   Nlocal++;
  }

  // //quiet start
  // repeat ( Ne/8 ) {
  //   double x = uniform2( gx0, gx1 , seed );
  //   double y = uniform2( gy0, gy1 , seed );
  //   double z = uniform2( gz0, gz1 , seed );
  //   n1 = v0e;
  //   n2 = 0;
  //   n3 = 0;
  //   n4 = v0i;
  //   n5 = 0;
  //   n6 = 0;

  //    signx = -1;
  //    signy = -1;
  //    signz = -1;
  //    for(int i=0; i<2; i++){
  //      signx = -signx;
  //      for(int j=0; j<2; j++){
  //  	signy = -signy;
  //  	for(int k=0; k<2; k++){
  //  	  signz = -signz;
  // 	  inject_particle( electron, x, y, z,
  // 		    n1*signx,
  //                   n2*signy,
  //                   n3*signz,we, 0, 0);
  // 	  inject_particle( ion, x, y, z,
  //                   n4*signx,
  //                   n5*signy,
  //                   n6*signz,wi, 0 ,0 );

  //  	}
  //      }
  //    }
 
    
  //   signx = -1;
  //   signy = -1;
  //   signz = -1;
  //   for(int i=0; i<2; i++){
  //     signx = -signx;
  //     for(int j=0; j<2; j++){
  // 	signy = -signy;
  // 	for(int k=0; k<2; k++){
  // 	  signz = -signz;

  // 	}
  //     }
  //   }
  
  //}
  // printf("Nlocal=%d (of %f)\n",Nlocal,Ne);
  sim_log( "Finished loading particles" );
  // field_t *_f = &field(33,0,1);
  // printf("ey=%e %e\n",_f->ey,_f->ey_n);
 
  //exit(1);

  // Upon completion of the initialization, the following occurs:
  // - The synchronization error (tang E, norm B) is computed between domains
  //   and tang E / norm B are synchronized by averaging where discrepancies
  //   are encountered.
  // - The initial divergence error of the magnetic field is computed and
  //   one pass of cleaning is done (for good measure)
  // - The bound charge density necessary to give the simulation an initially
  //   clean divergence e is computed.
  // - The particle momentum is uncentered from u_0 to u_{-1/2}
  // - The user diagnostics are called on the initial state
  // - The physics loop is started
  //
  // The physics loop consists of:
  // - Advance particles from x_0,u_{-1/2} to x_1,u_{1/2}
  // - User particle injection at x_{1-age}, u_{1/2} (use inject_particles)
  // - User current injection (adjust field(x,y,z).jfx, jfy, jfz)
  // - Advance B from B_0 to B_{1/2}
  // - Advance E from E_0 to E_1
  // - User field injection to E_1 (adjust field(x,y,z).ex,ey,ez,cbx,cby,cbz)
  // - Advance B from B_{1/2} to B_1
  // - (periodically) Divergence clean electric field
  // - (periodically) Divergence clean magnetic field
  // - (periodically) Synchronize shared tang e and norm b
  // - Increment the time step
  // - Call user diagnostics
  // - (periodically) Print a status message
}
 
begin_diagnostics {
 
# define should_dump(x) (global->x##_interval>0 && remainder(step(),global->x##_interval)==0)
 
  if( step()==-10 ) {
    // A grid dump contains all grid parameters, field boundary conditions,
    // particle boundary conditions and domain connectivity information. This
    // is stored in a binary format. Each rank makes a grid dump
    dump_grid("grid");
 
    // A materials dump contains all the materials parameters. This is in a
    // text format. Only rank 0 makes the materials dump
    dump_materials("materials");
 
    // A species dump contains the physics parameters of a species. This is in
    // a text format. Only rank 0 makes the species dump
    dump_species("species");
  }
 
  // Energy dumps store all the energies in various directions of E and B
  // and the total kinetic (not including rest mass) energies of each species
  // species in a simple text format. By default, the energies are appended to
  // the file. However, if a "0" is added to the dump_energies call, a new
  // energies dump file will be created. The energies are in the units of the
  // problem and are all time centered appropriately. Note: When restarting a
  // simulation from a restart dump made at a prior time step to the last
  // energies dump, the energies file will have a "hiccup" of intervening
  // time levels. This "hiccup" will not occur if the simulation is aborted
  // immediately following a restart dump. Energies dumps are in a text
  // format and the layout is documented at the top of the file. Only rank 0
  // makes makes an energies dump.
  if( should_dump(energies) ) {
    dump_energies( "energies", step()==0 ? 0 : 1 );
  }
  
  // Field dumps store the raw electromagnetic fields, sources and material
  // placement and a number of auxilliary fields. E, B and RHOB are
  // timecentered, JF and TCA are half a step old. Material fields are static
  // and the remaining fields (DIV E ERR, DIV B ERR and RHOF) are for
  // debugging purposes. By default, field dump filenames are tagged with
  // step(). However, if a "0" is added to the call, the filename will not be
  // tagged. The JF that gets stored is accumulated with a charge-conserving
  // algorithm. As a result, JF is not valid until at least one timestep has
  // been completed. Field dumps are in a binary format. Each rank makes a
  // field dump.
  if( step()==-10 )         dump_fields("fields"); // Get first valid total J
  if( should_dump(fields) ) dump_fields("fields");
  if( should_dump(fft_ex) ) dump_ex1d("ex1d", step()==0 ? 0 : 1 );
  if( should_dump(part_loc) ) dump_partloc("electron","partloc", step()==0 ? 0 : 1 );
 
  // Hydro dumps store particle charge density, current density and
  // stress-energy tensor. All these quantities are known at the time
  // t = time().  All these quantities are accumulated trilinear
  // node-centered. By default, species dump filenames are tagged with
  // step(). However, if a "0" is added to the call, the filename will not
  // be tagged. Note that the current density accumulated by this routine is
  // purely diagnostic. It is not used by the simulation and it is not
  // accumulated using a self-consistent charge-conserving method. Hydro dumps
  // are in a binary format. Each rank makes a hydro dump.
  if( should_dump(ehydro) ) dump_hydro("electron","ehydro");
  if( should_dump(ihydro) ) dump_hydro("ion",     "ihydro");
 
  // Particle dumps store the particle data for a given species. The data
  // written is known at the time t = time().  By default, particle dumps
  // are tagged with step(). However, if a "0" is added to the call, the
  // filename will not be tagged. Particle dumps are in a binary format.
  // Each rank makes a particle dump.
  if( should_dump(eparticle) ) dump_particles("electron","eparticle");
  if( should_dump(iparticle) ) dump_particles("ion",     "iparticle");
 
  // A checkpt is made by calling checkpt( fbase, tag ) where fname is a string
  // and tag is an integer.  A typical usage is:
  //   checkpt( "checkpt", step() ).
  // This will cause each process to write their simulation state to a file
  // whose name is based on fbase, tag and the node's rank.  For the above
  // usage, if called on step 314 on a 4 process run, the four files:
  //   checkpt.314.0, checkpt.314.1, checkpt.314.2, checkpt.314.3
  // to be written.  The simulation can then be restarted from this point by
  // invoking the application with "--restore checkpt.314".  checkpt must be 
  // the _VERY_ LAST_ diagnostic called.  If not, diagnostics performed after
  // the checkpt but before the next timestep will be missed on restore.
  // Restart dumps are in a binary format unique to the each simulation.

  if( should_dump(restart) ) checkpt( "checkpt", step() );

  // If you want to write a checkpt after a certain amount of simulation time,
  // use uptime() in conjunction with checkpt.  For example, this will cause
  // the simulation state to be written after 7.5 hours of running to the
  // same file every time (useful for dealing with quotas on big machines).
  //if( uptime()>=27000 ) {
  //  checkpt( "timeout", 0 );
  //  abort(0);
  //}
 
# undef should_dump
 
}
 
begin_particle_injection {
 
  // No particle injection for this simulation
 
}
 
begin_current_injection {
 
  // No current injection for this simulation
 
}
 
begin_field_injection {
 
  // No field injection for this simulation
 
}

begin_particle_collisions{ 

  // No collisions for this simulation 

}
