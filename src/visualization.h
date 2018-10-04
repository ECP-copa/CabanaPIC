#ifndef pic_visualization_h
#define pic_visualization_h

#include <iostream>
#include <fstream>

class Visualizer {

    public:
        std::ofstream vis_file;

        void write_header(size_t total_num_particles, size_t step) {

            std::stringstream sstm;

            sstm << "vis/step" << step << ".vtk";
            std::string file_name = sstm.str();

            vis_file.open(file_name);

            vis_file << "# vtk DataFile Version 2.0" << std::endl;
            vis_file << "Unstructured Grid Example" << std::endl;
            vis_file << "ASCII" << std::endl;
            vis_file << "" << std::endl;
            vis_file << "DATASET UNSTRUCTURED_GRID" << std::endl;

            vis_file << "POINTS " << total_num_particles << " float" << std::endl;
        }

        void write_particles_position(particle_list_t& particles)
        {
            auto position_x = particles.slice<PositionX>();
            auto position_y = particles.slice<PositionY>();
            auto position_z = particles.slice<PositionZ>();

            size_t write_count = 0;
            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        real_t x = position_x(idx);
                        real_t y = position_y(idx);
                        real_t z = position_z(idx);

                        vis_file << x << " " << y << " " << z << std::endl;
                        write_count++;
            }
        }

        void write_cell_types(size_t num_particles)
        {
            vis_file << "CELL_TYPES " << num_particles << std::endl;

            for (size_t p = 0; p < num_particles; p++)
            {
                vis_file << "1" << std::endl;
            }
        }

        void pre_scalars(size_t num_particles)
        {
            vis_file << "POINT_DATA " << num_particles << std::endl;
        }

        void write_particles_property_header(std::string name, size_t num_particles)
        {
            vis_file << "SCALARS " << name << " float 1"  << std::endl;
            vis_file << "LOOKUP_TABLE default" << std::endl;
        }

        void write_particles_w(particle_list_t& particles)
        {
            auto weight = particles.slice<Charge>();

            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        real_t w = weight(idx);

                        vis_file << w << std::endl;
            }
        }
        void write_particles_sp(particle_list_t& particles, size_t sn)
        {
            auto position_x = particles.slice<PositionX>();
            auto position_y = particles.slice<PositionY>();
            auto position_z = particles.slice<PositionZ>();

            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        real_t x = position_x(idx);
                        real_t y = position_y(idx);
                        real_t z = position_z(idx);

                        vis_file << sn << std::endl;
            }
        }

        void finalize()
        {
            vis_file.close();
        }

};

#endif // Visualizer
