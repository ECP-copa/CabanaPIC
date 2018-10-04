#ifndef pic_visualization_h
#define pic_visualization_h

#include <iostream>
#include <fstream>

namespace pic {
    class Visualizer {

        public:
            std::ofstream vis_file;

            void write_header(size_t total_num_particles, size_t step) {

                std::stringstream sstm;

                sstm << "vis" << step << ".vtk";
                std::string file_name = sstm.str();

                vis_file.open(file_name);

                vis_file << "# vtk DataFile Version 2.0" << std::endl;
                vis_file << "Unstructured Grid Example" << std::endl;
                vis_file << "ASCII" << std::endl;
                vis_file << "" << std::endl;
                vis_file << "DATASET UNSTRUCTURED_GRID" << std::endl;

                vis_file << "POINTS " << total_num_particles << " float" << std::endl;
            }

            void write_particle_pos(auto particles_accesor, size_t num_particles, mesh_t& m)
            {

                size_t write_count = 0;
                for ( auto c : m.cells() ) {

                    auto& cell_particles = particles_accesor[c];

                    // TODO: Is there a way abstract this loop structure with the current particle structure
                    // TODO: this may need to be "active ppc" or similar

                    // Only iterate over used blocks
                    for (size_t i = 0; i < cell_particles.block_number+1; i++)
                    {
                        for (size_t v = 0; v < PARTICLE_BLOCK_SIZE; v++)
                        {
                            real_t x = cell_particles.get_x(i, v);
                            real_t y = cell_particles.get_y(i, v);
                            real_t z = cell_particles.get_z(i, v);

                            // TODO: Find a better way to filter
                            // particles out, USE MASK? (both here and
                            // below)
                            if (x == 0 && y == 0 && z == 0) continue;

                            vis_file << x << " " << y << " " << z << std::endl;
                            write_count++;
                        }
                    }
                }
            }

            /*
               for (unsigned int sn = 0; sn < species.size(); sn++) {

               int particle_count = species[sn].num_particles;
               for (int particle_number = 0; particle_number < particle_count; particle_number++) {

               double* part_x_in;
               double* part_y_in;

               load_particle_position(sn, particle_number, &part_x_in, &part_y_in);

               vis_file << *part_x_in << " " << *part_y_in << " 0.0" << std::endl;
               }
               }
               */

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

            void write_particles_w(auto particles_accesor, mesh_t& m)
            {

                for ( auto c : m.cells() ) {

                    auto& cell_particles = particles_accesor[c];
                    for (size_t i = 0; i < cell_particles.block_number+1; i++)
                    {
                        for (size_t v = 0; v < PARTICLE_BLOCK_SIZE; v++)
                        {
                            real_t x = cell_particles.get_x(i, v);
                            real_t y = cell_particles.get_y(i, v);
                            real_t z = cell_particles.get_z(i, v);

                            real_t w = cell_particles.get_w(i, v);

                            if (x == 0 && y == 0 && z == 0) continue;

                            vis_file << w << std::endl;
                        }
                    }
                }
            }

            void write_particles_sp(auto particles_accesor, mesh_t& m, size_t sn)
            {
                for ( auto c : m.cells() ) {

                    auto& cell_particles = particles_accesor[c];
                    for (size_t i = 0; i < cell_particles.block_number+1; i++)
                    {
                        for (size_t v = 0; v < PARTICLE_BLOCK_SIZE; v++)
                        {
                            real_t x = cell_particles.get_x(i, v);
                            real_t y = cell_particles.get_y(i, v);
                            real_t z = cell_particles.get_z(i, v);

                            if (x == 0 && y == 0 && z == 0) continue;

                            vis_file << sn << std::endl;
                        }
                    }
                }
            }

            void finalize()
            {
                vis_file.close();
            }

    };
} // namespace pic

#endif // Visualizer
