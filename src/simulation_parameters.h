#ifndef SIMULATION_PARAMETERS_H
#define SIMULATION_PARAMETERS_H

enum Boundary {
    Reflect = 0
};


template <class real_> class Parameters_
{

    // TODO: Setters and getters
    public:

        // NOTE: It would be nice to standardize the units used here?

        // Define Consts
        const real_ mu; // permeability of free space
        const real_ c; // Speed of light
        const real_ eps; // permittivity of free space

        // Params
        size_t num_species;
        size_t NX_global;
        size_t NY_global;
        size_t NZ_global;
        size_t nx = NX_global;
        size_t ny = NY_global;
        size_t nz = NZ_global;
        size_t num_cells; // This should *include* the ghost cells
        size_t num_ghosts;
        size_t NPPC;
        size_t num_particles;
        double dt;
        int num_steps;

        Boundary BOUNDARY_TYPE = Boundary::Reflect;

        // Assume domain starts at [0,0,0] and goes to [len,len,len]
        real_ len_x_global = 1.0;
        real_ len_y_global = 1.0;
        real_ len_z_global = 1.0;
        real_ len_x = len_x_global;
        real_ len_y = len_y_global;
        real_ len_z = len_z_global;
        real_ local_x_min = 0.0;
        real_ local_y_min = 0.0;
        real_ local_z_min = 0.0;
        real_ local_x_max = len_x;
        real_ local_y_max = len_y;
        real_ local_z_max = len_z;
        real_ dx = len_x/nx;
        real_ dy = len_y/ny;
        real_ dz = len_z/nz;

        // TODO: how useful are these arguments now its a singleton?
        Parameters_(size_t _nc = 16, size_t _nppc = 32) :
            mu(4.0 * M_PI * 1.0e-7),
            c(299792458),
            eps( 1.0 / (c * c * mu)),
            num_species(0),
            NX_global(_nc),
            NY_global(_nc),
            NZ_global(_nc),
            nx(NX_global),
            ny(NY_global),
            nz(NZ_global),
            num_cells(NX_global * NY_global * NZ_global),
            num_ghosts(1),
            NPPC(_nppc),
            num_particles(NPPC*num_cells),
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
            dz(len_z/nz)
        {
            std::cout << "Singleton Constructor" << std::endl;
        }

        static Parameters_& instance()
        {
            static Parameters_ instance_;
            return instance_;
        }


        void print_run_details()
        {
            std::cout << "~~~ Run Specifications ~~~ " << std::endl;
            std::cout << "Nx: " << nx << " Ny: " << ny << " Nz: " << nz << " Num Ghosts: " << num_ghosts << ". Cells Total: " << num_cells << std::endl;
            std::cout << "Len X: " << len_x << " Len Y: " << len_y << " Len Z: " << len_z << num_ghosts << std::endl;
            std::cout << "Approx Particle Count: " << NPPC*num_cells << " (NPPC: " << NPPC << ")" << std::endl;
            std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~ " << std::endl;
            std::cout << std::endl;
        }
};
#endif // SIMULATION_PARAMETERS_H
