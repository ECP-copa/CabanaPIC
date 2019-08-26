#ifndef SIMULATION_PARAMETERS_H
#define SIMULATION_PARAMETERS_H

enum Boundary {
    Reflect = 0,
    Periodic
};


template <class real_> class Parameters_
{

    // TODO: Setters and getters
    public:

        // NOTE: It would be nice to standardize the units used here?

        // Define Consts
        const real_ de; // Length normalization (electron inertial length)
        const real_ ec; // Charge normalization
        const real_ me; // Mass normalization
        const real_ mu; // permeability of free space
        const real_ c; // Speed of light
        const real_ eps; // permittivity of free space


        // Params
        real_ n0; // Background plasma density
        size_t num_species;
        size_t NX_global;
        size_t NY_global;
        size_t NZ_global;
        size_t nx;
        size_t ny;
        size_t nz;

        size_t num_ghosts;
        size_t num_cells; // This should *include* the ghost cells
        size_t NPPC;
        size_t num_particles;
        double dt;
        int num_steps;

        // Assume domain starts at [0,0,0] and goes to [len,len,len]
        real_ len_x_global;
        real_ len_y_global;
        real_ len_z_global;
        real_ len_x;
        real_ len_y;
        real_ len_z;
        real_ local_x_min;
        real_ local_y_min;
        real_ local_z_min;
        real_ local_x_max;
        real_ local_y_max;
        real_ local_z_max;
        real_ dx;
        real_ dy;
        real_ dz;
        real_ v0; //drift velocity

        size_t ghost_offset; // Where the cell id needs to start for a "real" cell, basically nx
        size_t num_real_cells;

        //Boundary BOUNDARY_TYPE = Boundary::Reflect;
        Boundary BOUNDARY_TYPE = Boundary::Periodic;


        // TODO: how useful are these arguments now its a singleton?
        Parameters_(size_t _nc = 16, size_t _nppc = 32) :
            de(1.0),
            ec(1.0),
            me(1.0),
            c(1.0),
            eps( 1.0),
            mu(1.0),
            n0(1.0),
            num_species(1),
            NX_global(_nc),
            NY_global(1),
            NZ_global(1),
            nx(NX_global),
            ny(NY_global),
            nz(NZ_global),
            num_ghosts(1),
            num_cells( ((num_ghosts*2)+NX_global) * ((num_ghosts*2)+NY_global) * ((num_ghosts*2)+NZ_global)),
            NPPC(_nppc),
            num_real_cells(NX_global * NY_global * NZ_global),
            num_particles(NPPC*num_real_cells),
            dt(0.1),
            num_steps(10),
            len_x_global(1.0),
            len_y_global(1.0),
            len_z_global(1.0),
            len_x(len_x_global),
            len_y(len_y_global),
            len_z(len_z_global),
            local_x_min(0.0),
            local_y_min(0.0),
            local_z_min(0.0),
            local_x_max(len_x),
            local_y_max(len_y),
            local_z_max(len_z),
            dx(len_x/nx),
            dy(len_y/ny),
            dz(len_z/nz),
            ghost_offset(nx*num_ghosts),
            v0(0)
    {
        /* v0 = 0.0866025403784439*4.0; */
        /* real_ gam = 1.0/sqrt(1.0-v0*v0); */
        /* len_x_global =  0.628318530717959*(gam*sqrt(gam))*4.0; */
        /* len_x = len_x_global; */
        /* std::cout<<num_particles<<std::endl; */
        /* std::cout<<NPPC<<std::endl; */
        /* std::cout<<num_real_cells<<std::endl; */
        std::cout << "#Singeton Constructor" << std::endl;
    }

        static  Parameters_& instance()
        {
            static Parameters_ instance_;
            return instance_;
        }


        void print_run_details()
        {
            std::cout << "#~~~ Run Specifications ~~~ " << std::endl;
            std::cout << "#Nx: " << nx << " Ny: " << ny << " Nz: " << nz << " Num Ghosts: " << num_ghosts << ". Cells Total: " << num_cells << std::endl;
            std::cout << "#Len X: " << len_x << " Len Y: " << len_y << " Len Z: " << len_z << "number of ghosts: "<<num_ghosts << std::endl;
            std::cout << "#Approx Particle Count: " << num_particles << " (NPPC: " << NPPC << ")" << std::endl;
            std::cout << "#~~~~~~~~~~~~~~~~~~~~~~~~~~ " << std::endl;
            std::cout << std::endl;
        }
};
#endif // SIMULATION_PARAMETERS_H
