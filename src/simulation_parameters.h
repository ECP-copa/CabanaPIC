#ifndef SIMULATION_PARAMETERS_H
#define SIMULATION_PARAMETERS_H

template <class real_> class Parameters_
{

    // TODO: Setters and getters
    public:

        static Parameters_& instance()
        {
            static Parameters_ instance_;
            return instance_;
        }

        // NOTE: It would be nice to standardize the units used here?

        // Define Consts
        const real_ mu = 4.0 * M_PI * 1.0e-7; // permeability of free space
        const real_ c = 299792458; // Speed of light
        const real_ eps = 1.0 / (c * c * mu); // permittivity of free space

        // Other Params
        size_t num_species = 0;

        size_t NX_global = 64;
        size_t NY_global = 64;
        size_t NZ_global = 64;

        size_t nx = NX_global;
        size_t ny = NY_global;
        size_t nz = NZ_global;

        size_t NPPC = 32;

        double dt = 0.1;
        int num_steps = 10;

        real_ len_x_global = 1.0;
        real_ len_y_global = 1.0;
        real_ len_z_global = 1.0;

        real_ len_x = len_x_global;
        real_ len_y = len_y_global;
        real_ len_z = len_z_global;

        // Assume domain starts at [0,0,0] and goes to [len,len,len]
        real_ local_x_min = 0.0;
        real_ local_y_min = 0.0;
        real_ local_z_min = 0.0;

        real_ local_x_max = len_x;
        real_ local_y_max = len_y;
        real_ local_z_max = len_z;

        real_ dx = len_x/nx;
        real_ dy = len_y/ny;
        real_ dz = len_z/nz;
};
#endif // SIMULATION_PARAMETERS_H
