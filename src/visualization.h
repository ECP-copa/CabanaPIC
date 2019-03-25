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

        // TODO: all these loops are the same, we could replace it with vtemplate
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

        void write_particles_index(particle_list_t& particles)
        {
            auto cell = particles.slice<Cell_Index>();

            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        real_t w = cell(idx);

                        vis_file << w << std::endl;
            }
        }

        void write_particles_w(particle_list_t& particles)
        {
            auto weight = particles.slice<Weight>();

            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        real_t w = weight(idx);

                        vis_file << w << std::endl;
            }
        }

        void write_particles_sp(particle_list_t& particles, size_t sn)
        {
            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        vis_file << sn << std::endl;
            }
        }

        void finalize()
        {
            vis_file.close();
        }

        void write_vis(particle_list_t particles, size_t step)
        {

            size_t total_num_particles = particles.size();

            // TODO: this needs to be updated once species are introduced
            /*
               for (unsigned int sn = 0; sn < species.size(); sn++)
               {
               int particle_count = species[sn].num_particles;
               total_num_particles += particle_count;
               }
            */

            write_header(total_num_particles, step);

            //for (unsigned int sn = 0; sn < species.size(); sn++)
            //{
            //auto particles_accesor = get_particle_accessor(m, species[sn].key);
            write_particles_position(particles);
            //}

            write_cell_types(total_num_particles);

            pre_scalars(total_num_particles);
            write_particles_property_header("weight", total_num_particles);

            //for (unsigned int sn = 0; sn < species.size(); sn++)
            //{
            //auto particles_accesor = get_particle_accessor(m, species[sn].key);
            write_particles_w(particles);
            //}
            //*/
            write_particles_property_header("cells", total_num_particles);
            write_particles_index(particles);

            write_particles_property_header("species", total_num_particles);

            //for (unsigned int sn = 0; sn < species.size(); sn++)
            //{
            //auto particles_accesor = get_particle_accessor(m, species[sn].key);
            write_particles_sp(particles, 1);
            //}
            finalize();

        }

};

#endif // Visualizer
