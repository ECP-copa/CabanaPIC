#ifdef SIMULATION_PARAMETERS_H
#define SIMULATION_PARAMETERS_H

template <class real_t> class Parameters_
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
        const real_t mu = 4.0 * M_PI * 1.0e-7; // permeability of free space
        const real_t c = 299792458; // Speed of light
        const real_t eps = 1.0 / (c * c * mu); // permittivity of free space

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

        real_t len_x_global = 1.0;
        real_t len_y_global = 1.0;
        real_t len_z_global = 1.0;

        real_t len_x = len_x_global;
        real_t len_y = len_y_global;
        real_t len_z = len_z_global;

        // Assume domain starts at [0,0,0] and goes to [len,len,len]
        real_t local_x_min = 0.0;
        real_t local_y_min = 0.0;
        real_t local_z_min = 0.0;

        real_t local_x_max = len_x;
        real_t local_y_max = len_y;
        real_t local_z_max = len_z;

        real_t dx = len_x/nx;
        real_t dy = len_y/ny;
        real_t dz = len_z/nz;
};
#endif // SIMULATION_PARAMETERS_H
