#ifndef sparse_filter_h
#define sparse_filter_h

#include "fields.h"

class Binomial_Filters
{
private:
    
    std::vector<field_array_t> d_grids;
    std::vector<field_array_t> d_grids2;
    std::vector<field_array_t> d_grids_gen_p, d_grids_gen_m;

    std::vector<int> xres_p, yres_p, xres_m, yres_m;

    field_array_t d_fields_out;
    int num_coarsenings;
    bool verbose = true;
    
public:
    //constructor
    Binomial_Filters(int nx_out, int ny_out, int nz_out, int ng, int minres):d_fields_out("fields_interpolated", (nx_out + 2*ng)*(ny_out + 2*ng)*(nz_out + 2*ng))
	 {
			size_t nx_tmp = nx_out;
			size_t ny_tmp = ny_out;
			if(ny_tmp<2) {
			    std::cout<<"Error: Need 2D grid. Quit.\n";			    
			    exit(1);
			}
			int num_x_coarsenings, num_y_coarsenings;
			num_x_coarsenings=0; num_y_coarsenings = 0;
			while (nx_tmp > minres && nx_tmp % 2 == 0) { nx_tmp /= 2; num_x_coarsenings += 1; }
			while (ny_tmp > minres && ny_tmp % 2 == 0) { ny_tmp /= 2; num_y_coarsenings += 1; }
			num_coarsenings = std::min(num_x_coarsenings, num_y_coarsenings);
			if ( verbose ) { std::cout << "Number of coarsenings = " << num_coarsenings << std::endl; }
	 }

	 void printGridResolutions() {
			std::cout << "Plus-diagonal resolutions are:" << std::endl;
			std::cout << "x: ";
			for ( int i=0; i<xres_p.size(); i++ ) {
				std::cout << xres_p[i] << " ";
			}
			std::cout << std::endl;
			std::cout << "y: ";
			for ( int i=0; i<yres_p.size(); i++ ) {
				std::cout << yres_p[i] << " ";
			}
			std::cout << std::endl;

			std::cout << "Minus diagonal resolutions are:" << std::endl;
			std::cout << "x: ";
			for ( int i=0; i<xres_m.size(); i++ ) {
				std::cout << xres_m[i] << " ";
			}
			std::cout << std::endl;
			std::cout << "y: ";
			for ( int i=0; i<yres_m.size(); i++ ) {
				std::cout << yres_m[i] << " ";
			}
			std::cout << std::endl;

			
	 }
		
		// This is the overall function that applies the (simple, 2D) SG filtering... composed of the functions in the class below
		void SGfilter(
			field_array_t& fields, // this argument is assumed to already have all its ghost cells filled properly 
			size_t nx, 
			size_t ny, 
			size_t nz, 
			size_t ng,
			size_t step = 0
			)
		{
			if ( verbose ) { std::cout << "Starting SGCT" << std::endl; }
			constructSGCTcomponentGrids(fields, nx, ny, nz, ng,step); // Constructs the component grids in combination technique
			if ( verbose ) { std::cout << "Component grids constructed" << std::endl; }
			SGinterpolate( nx, ny, nz, ng);	// Interpolates component grids onto original grid resolution
			if ( verbose ) { std::cout << "Output grid reinterpolated" << std::endl; }

			// The rest of the function just copies the current density of d_fields_out into the original fields argument

			auto jfx_orig = Cabana::slice<FIELD_JFX>(fields);
			auto jfy_orig = Cabana::slice<FIELD_JFY>(fields);
			auto jfz_orig = Cabana::slice<FIELD_JFZ>(fields);

			auto jfx_sg = Cabana::slice<FIELD_JFX>(d_fields_out);
			auto jfy_sg = Cabana::slice<FIELD_JFY>(d_fields_out);
			auto jfz_sg = Cabana::slice<FIELD_JFZ>(d_fields_out);

			auto _copy_to_orig = KOKKOS_LAMBDA( const int i ) 
				{
					jfx_orig(i) = jfx_sg(i);
					jfy_orig(i) = jfy_sg(i);
					jfz_orig(i) = jfz_sg(i);
				};
			Kokkos::parallel_for( fields.size(), _copy_to_orig, "copy_to_orig()" );
			
			std::cout << "aa" << std::endl;
			serial_update_ghosts_B(jfx_orig, jfy_orig, jfz_orig, nx, ny, nz, ng); // refill ghost cells
		    //clean up the grids
		    //d_grids.clear();
		    //d_grids2.clear();
			 d_grids_gen_p.clear();
			 std::cout << "a" << std::endl;
			 d_grids_gen_m.clear();
			 std::cout << "b" << std::endl;
			 xres_p.clear(); yres_p.clear();
			 xres_m.clear(); yres_m.clear();
			 std::cout << "c" << std::endl;
		}
	
		void filter_on_axis(
					field_array_t& fields_out, 
					field_array_t& fields_in, 
					size_t nx, 
					size_t ny, 
					size_t nz, 
					size_t ng,
					size_t axis
					)
		{
				assert(fields_out.size()==fields_in.size());
            auto jfx_in = Cabana::slice<FIELD_JFX>(fields_in);
            auto jfy_in = Cabana::slice<FIELD_JFY>(fields_in);
            auto jfz_in = Cabana::slice<FIELD_JFZ>(fields_in);
            
				auto jfx_out = Cabana::slice<FIELD_JFX>(fields_out);
            auto jfy_out = Cabana::slice<FIELD_JFY>(fields_out);
            auto jfz_out = Cabana::slice<FIELD_JFZ>(fields_out);
	    		
				//serial_update_ghosts(jfx_in, jfy_in, jfz_in, nx, ny, nz, ng);
            //serial_update_ghosts_B(jfx_in, jfy_in, jfz_in, nx, ny, nz, ng);
		    
			 	auto _filter = KOKKOS_LAMBDA( const int x, const int y, const int z)
				{
			      const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
					int ip1, im1;
					if (axis == 0) 
					{
						ip1 = VOXEL(x+1, y, z, nx, ny, nz, ng);
						im1 = VOXEL(x-1, y, z, nx, ny, nz, ng);
					}
					else if (axis == 1)
					{
						ip1 = VOXEL(x, y+1, z, nx, ny, nz, ng);
						im1 = VOXEL(x, y-1, z, nx, ny, nz, ng);
					}
					else
					{
						ip1 = VOXEL(x, y, z+1, nx, ny, nz, ng);
						im1 = VOXEL(x, y, z-1, nx, ny, nz, ng);
					}

					jfx_out(i) = 0.25*jfx_in(im1) + 0.5*jfx_in(i) + 0.25*jfx_in(ip1);
					jfy_out(i) = 0.25*jfy_in(im1) + 0.5*jfy_in(i) + 0.25*jfy_in(ip1);
					jfz_out(i) = 0.25*jfz_in(im1) + 0.5*jfz_in(i) + 0.25*jfz_in(ip1);
			      
				};
	    		Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});
            //Kokkos::MDRangePolicy<ExecutionSpace> exec_policy( 0, fields_in.size() );
            Kokkos::parallel_for( exec_policy, _filter, "binomial_filter()" );
		}

/*****************************************************************************************************************/
		field_array_t filter_and_restrict(
					const field_array_t& fields_in, 
					size_t nx_in, 
					size_t ny_in,
					size_t nz_in, 
					size_t ng,
					int axis
					)
		{
            auto jfx_in = Cabana::slice<FIELD_JFX>(fields_in);
            auto jfy_in = Cabana::slice<FIELD_JFY>(fields_in);
            auto jfz_in = Cabana::slice<FIELD_JFZ>(fields_in);

			/*if (axis==1) {
				for ( int i=0; i<=fields_in.size(); i++ ) {
					std::cout << jfx_in(i) << "	" << jfy_in(i) << "	" << jfz_in(i) << std::endl;
				}
			}*/
				size_t nx_out = nx_in;
				size_t ny_out = ny_in;
				size_t nz_out = nz_in;
				if ( axis == 0 ) { assert(nx_out % 2 == 0); nx_out /= 2; }
				else if (axis==1){ assert(ny_out % 2 == 0); ny_out /= 2; }
				else 				  { assert(nz_out % 2 == 0); nz_out /= 2; }
				size_t num_cells_out = (nx_out + 2*ng)*(ny_out+2*ng)*(nz_out+2*ng);
				field_array_t fields_out("fields_filtered", num_cells_out);
            
				auto jfx_out = Cabana::slice<FIELD_JFX>(fields_out);
				auto jfy_out = Cabana::slice<FIELD_JFY>(fields_out);
				auto jfz_out = Cabana::slice<FIELD_JFZ>(fields_out);

				const int coarse_fac_x = (int) nx_in/nx_out;
				const int coarse_fac_y = (int) ny_in/ny_out;
				const int coarse_fac_z = (int) nz_in/nz_out;
				const int inc_x = coarse_fac_x - 1;
				const int inc_y = coarse_fac_y - 1;
				const int inc_z = coarse_fac_z - 1;
				//std::cout << coarse_fac_x << coarse_fac_y << coarse_fac_z << std::endl;
				assert(coarse_fac_x==1 || coarse_fac_x==2);
				assert(coarse_fac_y==1 || coarse_fac_y==2);
				assert(coarse_fac_z==1 || coarse_fac_z==2);
				assert(coarse_fac_x*coarse_fac_y*coarse_fac_z==2); // Exactly one direction should be coarsened by a factor of two... 
																					// just a sanity check on the coarsening we did above, and that the number of elements in the coarsened direction was even
																					//
				//if (axis==1) { std::cout << coarse_fac_x << "	" << coarse_fac_y << "	" << coarse_fac_z << std::endl; }

		    
			 	auto _filter = KOKKOS_LAMBDA( const int x, const int y, const int z)
				{
			      const int i = VOXEL(x,   y,   z,   nx_out, ny_out, nz_out, ng);  // index in the OUTPUT array (coarser resolution)
					const int x2 = coarse_fac_x*x;
					const int y2 = coarse_fac_y*y;
					const int z2 = coarse_fac_z*z;
					const int i2 = VOXEL(x2, y2, z2, nx_in, ny_in, nz_in, ng); // index in the INPUT array (finer resolution)
					const int ip1 = VOXEL(x2+inc_x,y2+inc_y,z2+inc_z, nx_in, ny_in, nz_in, ng);
					const int im1 = VOXEL(x2-inc_x,y2-inc_y,z2-inc_z, nx_in, ny_in, nz_in, ng);

					jfx_out(i) = 0.25*jfx_in(im1) + 0.5*jfx_in(i2) + 0.25*jfx_in(ip1);
					jfy_out(i) = 0.25*jfy_in(im1) + 0.5*jfy_in(i2) + 0.25*jfy_in(ip1);
					jfz_out(i) = 0.25*jfz_in(im1) + 0.5*jfz_in(i2) + 0.25*jfz_in(ip1);

				};
	    		Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({ng,ng,ng}, {nx_out+ng,ny_out+ng,nz_out+ng});
            Kokkos::parallel_for( exec_policy, _filter, "binomial_filter_with_restriction()" );
		 
		 		serial_update_ghosts_B(jfx_out, jfy_out, jfz_out, nx_out, ny_out, nz_out, ng); // Fill ghost cells in output grid

				return fields_out;
		}

/*****************************************************************************************************************/
		void constructSGCTcomponentGrids(
				field_array_t& fields,
				size_t nx, 
				size_t ny, 
				size_t nz, 
				size_t ng,
				size_t step = 0
				)
		{

			 if ( verbose ) { std::cout << "Cleared vectors" << std::endl; }
		    
			//auto jfx_in = Cabana::slice<FIELD_JFX>(fields);
			//auto jfy_in = Cabana::slice<FIELD_JFY>(fields);
			//auto jfz_in = Cabana::slice<FIELD_JFZ>(fields);

         //serial_update_ghosts_B(jfx_in, jfy_in, jfz_in, nx, ny, nz, ng);

			//I'm going to implement a 2D-only version first as a proof-of-principle
			if ( verbose ) { std::cout << "Starting construction of SGCT grids" << std::endl; }
			//The grids on the "plus" diagonal
			d_grids_gen_p.push_back(fields);
			xres_p.push_back(nx); yres_p.push_back(ny);
			size_t nx_tmp = nx;
			size_t ny_tmp = ny;
			for (int i=1; i<=num_coarsenings; i++) {
				//field_array_t coarsened = 
				d_grids_gen_p.push_back(filter_and_restrict(d_grids_gen_p[i-1], nx_tmp, ny, nz, ng, 0));
				nx_tmp /= 2;
				xres_p.push_back(nx_tmp); yres_p.push_back(ny_tmp);
			}
			if ( verbose ) { std::cout << "Finished x-coarsening for plus grids" << std::endl; }
			// The above loop gets the right dimensions in vector indices 1 to num_coarsenings, this next one does that for index 0
			for (int i=0; i<=num_coarsenings; i++) {
				for (int j=1; j<=num_coarsenings-i; j++) {
					d_grids_gen_p[i] = filter_and_restrict(d_grids_gen_p[i], xres_p[i], ny_tmp, nz, ng, 1);
					ny_tmp /= 2;
					yres_p[i] = ny_tmp;
				}
				ny_tmp = ny;
			}
			if ( verbose ) { std::cout << "Constructed plus grids" << std::endl; }

			// Now make the grids on the "minus" diagonal
			for ( int i=0; i<num_coarsenings; i++ ) {
				//field_array_t coarsened = 
				d_grids_gen_m.push_back(filter_and_restrict(d_grids_gen_p[i], xres_p[i], yres_p[i], nz, ng, 0));//oarsened); 
				xres_m.push_back(xres_p[i]/2); yres_m.push_back(yres_p[i]);
			}

			if ( verbose ) { std::cout << "Constructed minus grids" << std::endl; printGridResolutions(); }
			/*
			if(step==100){
			for(int n=0; n<=num_coarsenings; ++n){
			    nx = xres_p[n];
			    ny = yres_p[n];
			    auto jfx = Cabana::slice<FIELD_JFX>(d_grids_gen_p[n]);
			    auto jfy = Cabana::slice<FIELD_JFY>(d_grids_gen_p[n]);
			    auto jfz = Cabana::slice<FIELD_JFZ>(d_grids_gen_p[n]);

			    for(int i=0; i<nx+2*ng; ++i){
				for(int j=0; j<ny+2*ng; ++j){
				    int ii = VOXEL(i,j,1,nx,ny,nz,ng);
				    printf("%d %d %e %e %e \n",i,j,jfx(ii),jfy(ii),jfz(ii));
				}
				printf("\n");
			    }
			    printf("\n\n");
			}
			}*/
			
		}

/*****************************************************************************************************************/
		field_array_t interpolate_on_axis(
				field_array_t fields_in, 
				size_t nx_in, 
				size_t ny_in, 
				size_t nz_in, 
				size_t ng, 
				int axis)

		{
         auto jfx_in = Cabana::slice<FIELD_JFX>(fields_in);
         auto jfy_in = Cabana::slice<FIELD_JFY>(fields_in);
         auto jfz_in = Cabana::slice<FIELD_JFZ>(fields_in);

			// fill ghost cells
         serial_update_ghosts_B(jfx_in, jfy_in, jfz_in, nx_in, ny_in, nz_in, ng);

			size_t nx_out = nx_in;
			size_t ny_out = ny_in;
			size_t nz_out = nz_in;
			size_t refine_fac_x = 1;
			size_t refine_fac_y = 1;
			size_t refine_fac_z = 1;
			if (axis==0) 	   { nx_out *= 2; refine_fac_x = 2; }
			else if (axis==1) { ny_out *= 2; refine_fac_y = 2; }
			else 					{ nz_out *= 2; refine_fac_z = 2; }
			
			size_t num_cells_out = (nx_out + 2*ng)*(ny_out + 2*ng)*(nz_out + 2*ng);
			field_array_t fields_out("fields_interpolated", num_cells_out);

			auto jfx_out = Cabana::slice<FIELD_JFX>(fields_out);
			auto jfy_out = Cabana::slice<FIELD_JFY>(fields_out);
			auto jfz_out = Cabana::slice<FIELD_JFZ>(fields_out);
				
			auto _interpolate = KOKKOS_LAMBDA( const int x, const int y, const int z)
				{
					const int i = VOXEL(x,   y,   z,   nx_out, ny_out, nz_out, ng);  // index in the OUTPUT array (finer resolution)
					int x2, y2, z2;
					bool interpolate = false;
					if (axis==0) {
						if (x % 2 == 0) { x2 = x/2; }
						else { x2 = (x-1)/2; interpolate = true; }
					}
					else { x2 = x; }
					if (axis==1) {
						if (y % 2 == 0) { y2 = y/2; }
						else { y2 = (y-1)/2; interpolate = true; }
					}
					else { y2 = y; }
					if (axis==2) {
						if (z % 2 == 0) { z2 = z/2; }
						else { z2 = (z-1)/2; interpolate = true; }
					}
					else { z2 = z; }

					const int i2 = VOXEL(x2, y2, z2, nx_in, ny_in, nz_in, ng); // index in the INPUT array (coarser resolution)
					int ip1;
					if (axis == 0) 
					{
						ip1 = VOXEL(x2+1, y2, z2, nx_in, ny_in, nz_in, ng);
					}
					else if (axis == 1)
					{
						ip1 = VOXEL(x2, y2+1, z2, nx_in, ny_in, nz_in, ng);
					}
					else
					{
						ip1 = VOXEL(x2, y2, z2+1, nx_in, ny_in, nz_in, ng);
					}

					if (interpolate) {
						jfx_out(i) = 0.5*jfx_in(ip1) + 0.5*jfx_in(i2); 
						jfy_out(i) = 0.5*jfy_in(ip1) + 0.5*jfy_in(i2);
						jfz_out(i) = 0.5*jfz_in(ip1) + 0.5*jfz_in(i2);
					}
					else {
						jfx_out(i) = jfx_in(i2);
						jfy_out(i) = jfy_in(i2);
						jfz_out(i) = jfz_in(i2);
					}
					
				};
				Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({ng,ng,ng}, {nx_out+ng,ny_out+ng,nz_out+ng});
				Kokkos::parallel_for( exec_policy, _interpolate, "interpolate()" );

            serial_update_ghosts_B(jfx_out, jfy_out, jfz_out, nx_out, ny_out, nz_out, ng); // Make sure output has ghost cells
				return fields_out;

		}

/*****************************************************************************************************************/

		void add_fields_inplace(
				field_array_t fields, 
				field_array_t fields_to_be_added, 
				real_t fac)
		{
			
			assert(fields.size() == fields_to_be_added.size()); // ensures same total number of grid points... user is responsible for making sure
																				 // these flattened arrays represent grids of the same shape (e.g. this assert won't keep 
																				 // you from adding together a 32x64x16 grid and a 16x32x64 grid and getting nonsense)

      	auto jfx = Cabana::slice<FIELD_JFX>(fields);
			auto jfy = Cabana::slice<FIELD_JFY>(fields);
			auto jfz = Cabana::slice<FIELD_JFZ>(fields);
					  
			auto jfx_add = Cabana::slice<FIELD_JFX>(fields_to_be_added);
			auto jfy_add = Cabana::slice<FIELD_JFY>(fields_to_be_added);
			auto jfz_add = Cabana::slice<FIELD_JFZ>(fields_to_be_added);

			auto _add = KOKKOS_LAMBDA( const int i ) 
				{
					jfx(i) += fac*jfx_add(i);
					jfy(i) += fac*jfy_add(i);
					jfz(i) += fac*jfz_add(i);
				};
			Kokkos::parallel_for( fields.size(), _add, "add_fields()" );

		}

/*****************************************************************************************************************/

		void SGinterpolate(
				size_t nx_out, 
				size_t ny_out, 
				size_t nz_out, 
				size_t ng)
		{

			// Interpolate grids on "plus" diagonal until they have the right resolution
		    //for ( int i=0; i<=num_coarsenings; i++ ) {
		    for ( int i=num_coarsenings; i>=0; i-- ) {
			    std::cout << "In plus grids, i = " << i << " of number_coarsenings: "<<num_coarsenings<<std::endl;
				while ( xres_p[i] < nx_out ) {
					d_grids_gen_p[i] = interpolate_on_axis(d_grids_gen_p[i], xres_p[i], yres_p[i], nz_out, ng, 0);
					xres_p[i] *= 2;
					std::cout << xres_p[i] << std::endl;
				}
				std::cout << "x-interpolation successful" << std::endl;
				while ( yres_p[i] < ny_out ) {
					d_grids_gen_p[i] = interpolate_on_axis(d_grids_gen_p[i], xres_p[i], yres_p[i], nz_out, ng, 1);
					yres_p[i] *= 2;
					std::cout << yres_p[i] << std::endl;
				}
				std::cout << "y-interpolation successful" << std::endl;
				
			}
			if ( verbose ) { std::cout << "Interpolated plus grids" << std::endl; }

			// Same for the "minus" diagonal
			//for ( int i=0; i<num_coarsenings; i++ ) {
			for ( int i=num_coarsenings-1; i>=0; i-- ) {
				while ( xres_m[i] < nx_out ) {
					d_grids_gen_m[i] = interpolate_on_axis(d_grids_gen_m[i], xres_m[i], yres_m[i], nz_out, ng, 0);
					xres_m[i] *= 2;
					std::cout << xres_m[i] << std::endl;
				}
				std::cout << "x-interpolation successful" << std::endl;
				while ( yres_m[i] < ny_out ) {
					d_grids_gen_m[i] = interpolate_on_axis(d_grids_gen_m[i], xres_m[i], yres_m[i], nz_out, ng, 1);
					yres_m[i] *= 2;
					std::cout << yres_m[i] << std::endl;
				}
				std::cout << "y-interpolation successful" << std::endl;
			}
			if ( verbose ) { 
				std::cout << "Interpolated minus grids" << std::endl;
				std::cout << "Printing grid resolutions after interpolation" << std::endl;
				printGridResolutions();
			}


      	auto jfx = Cabana::slice<FIELD_JFX>(d_fields_out);
         auto jfy = Cabana::slice<FIELD_JFY>(d_fields_out);
         auto jfz = Cabana::slice<FIELD_JFZ>(d_fields_out);

			auto _init = KOKKOS_LAMBDA( const int i )
				{
					jfx(i) = 0.0;
					jfy(i) = 0.0;
					jfz(i) = 0.0;
				};

			Kokkos::parallel_for( d_fields_out.size(), _init, "init_output_fields()" );

			// Add up contributions from "plus" diagonal
			for ( int i=0; i<=num_coarsenings; i++ ) {
				add_fields_inplace(d_fields_out, d_grids_gen_p[i], 1.);
			}
			// And subtract contributions from "minus" diagonal
			for ( int i=0; i<num_coarsenings; i++ ) {
				add_fields_inplace(d_fields_out, d_grids_gen_m[i], -1.);
			}

		}
											 

};

#endif
