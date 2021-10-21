#ifndef pic_visualization_h
#define pic_visualization_h

#include <iostream>
#include <fstream>

class Visualizer {

    public:
        std::ofstream vis_file;

		  bool writeParticles = false;
		  bool writeE = false;
		  bool writeJ = false;
		  bool writeGrid = false;

        void write_header(size_t total_num_particles, size_t step, std::string data_name) {

            std::stringstream sstm;

            sstm << "vis/" << data_name << "_step" << step << ".vtk";
            std::string file_name = sstm.str();

            vis_file.open(file_name);

            vis_file << "# vtk DataFile Version 2.0" << std::endl;
            vis_file << "Unstructured Grid Example" << std::endl;
            vis_file << "ASCII" << std::endl;
            vis_file << "" << std::endl;
            vis_file << "DATASET UNSTRUCTURED_GRID" << std::endl;

            vis_file << "POINTS " << total_num_particles << " float" << std::endl;
        }

		  void write_grid(const int nx, const int ny, const int nz, const int ng, const real_t dx, const real_t dy, const real_t dz)
		  {
		      real_t x_val, y_val, z_val;
				for ( int x = ng; x < nx+ng; ++x)
				{
					for ( int y = ng; y < ny+ng; ++y)
					{
						for ( int z = ng; z < nz+ng; ++z)
						{
							x_val = (x-ng)*dx; y_val = (y-ng)*dy; z_val = (z-ng)*dz;
							vis_file << x_val << " " << y_val << " " << z_val << std::endl;
						}
					}
				}
		  }

		  void write_efield(field_array_t& fields, const int nx, const int ny, const int nz, const int ng)
		  {
		  		auto ex = Cabana::slice<FIELD_EX>(fields);
		  		auto ey = Cabana::slice<FIELD_EY>(fields);
		  		auto ez = Cabana::slice<FIELD_EZ>(fields);
				size_t write_count = 0;

				int idx;

				for ( int x = ng; x < nx+ng; ++x)
				{
					for ( int y = ng; y < ny+ng; ++y)
					{
						for ( int z = ng; z < nz+ng; ++z)
						{
							idx = VOXEL(x, y, z, nx, ny, nz, ng);
							
							real_t this_ex = ex(idx);
							real_t this_ey = ey(idx);
							real_t this_ez = ez(idx);

							vis_file << this_ex << " " << this_ey << " " << this_ez << std::endl;
							write_count++;
						}
					}
				}
		  }
		  
		  void write_current(field_array_t& fields, const int nx, const int ny, const int nz, const int ng)
		  {
		  		auto jx = Cabana::slice<FIELD_JFX>(fields);
		  		auto jy = Cabana::slice<FIELD_JFY>(fields);
		  		auto jz = Cabana::slice<FIELD_JFZ>(fields);
				size_t write_count = 0;

				int idx;

				for ( int x = ng; x < nx+ng; ++x)
				{
					for ( int y = ng; y < ny+ng; ++y)
					{
						for ( int z = ng; z < nz+ng; ++z)
						{
							idx = VOXEL(x, y, z, nx, ny, nz, ng);
							
							real_t this_jx = jx(idx);
							real_t this_jy = jy(idx);
							real_t this_jz = jz(idx);

							vis_file << this_jx << " " << this_jy << " " << this_jz << std::endl;
							write_count++;
						}
					}
				}
		  }

        // TODO: all these loops are the same, we could replace it with vtemplate
        void write_particles_position(particle_list_t& particles)
        {
            auto position_x = Cabana::slice<PositionX>(particles);
            auto position_y = Cabana::slice<PositionY>(particles);
            auto position_z = Cabana::slice<PositionZ>(particles);

            auto velocity_x = Cabana::slice<VelocityX>(particles);
            auto velocity_y = Cabana::slice<VelocityY>(particles);
            auto velocity_z = Cabana::slice<VelocityZ>(particles);

            size_t write_count = 0;
            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        real_t x = position_x(idx);
                        real_t y = position_y(idx);
                        real_t z = position_z(idx);

								real_t vx = velocity_x(idx);
								real_t vy = velocity_y(idx);
								real_t vz = velocity_z(idx);

                        vis_file << x << " " << y << " " << z << " " << vx << " " << vy << " " << vz << std::endl;
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
            auto cell = Cabana::slice<Cell_Index>(particles);

            for ( std::size_t idx = 0; idx != particles.size(); ++idx )
            {
                        real_t w = cell(idx);

                        vis_file << w << std::endl;
            }
        }

        void write_particles_w(particle_list_t& particles)
        {
            auto weight = Cabana::slice<Weight>(particles);

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

        void write_vis(particle_list_t particles, field_array_t fields, size_t step, const int nx, const int ny, const int nz, const int ng,
		  					  real_t dx, real_t dy, real_t dz)
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
				if ( writeParticles ) 
				{
						  write_header(total_num_particles, step, "particles");

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
					if ( writeE ) 
					{
						write_header(total_num_particles, step, "efield");
						write_efield( fields, nx, ny, nz, ng );
						finalize();
					}
					if ( writeJ )
					{
						write_header(total_num_particles, step, "current");
						write_current( fields, nx, ny, nz, ng );
						finalize();
					}
					if ( writeGrid )
					{
						write_header(total_num_particles, step, "grid");
						write_grid(nx, ny, nz, ng, dx, dy, dz);
						finalize();
					}

        }

};

#endif // Visualizer
