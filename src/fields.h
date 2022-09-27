#ifndef pic_EM_fields_h
#define pic_EM_fields_h

#include <fstream>
#include "Cabana_Parallel.hpp" // Simd parallel for
#include "Cabana_DeepCopy.hpp" // Cabana::deep_copy
#include "input/deck.h"

#ifdef USE_GPU
#include <cufft.h>
#else
#include "fftw3.h"
#endif

#include <cassert>
//#include <vector>

// TODO: Namespace this stuff?

template<class Slice_X, class Slice_Y, class Slice_Z>
//KOKKOS_INLINE_FUNCTION
void serial_update_ghosts_B(
        //field_array_t& fields,
        Slice_X slice_x,
        Slice_Y slice_y,
        Slice_Z slice_z,
        int nx, int ny, int nz, int ng)
{
    const Boundary boundary = deck.BOUNDARY_TYPE;
    if (boundary == Boundary::Reflect)
    {
        // TODO: this
        exit(1);
    }
    else { // assume periodic

        // TODO: it may be worth turning these into fewer kernels, as they
        // really don't have a lot of work

        //for (int z = 1; z < nz+1; z++) {
            //for (int y = 1; y < ny+1; y++) {
        auto _zy_boundary = KOKKOS_LAMBDA( const int z, const int y )
        {
            // Copy x from LHS -> RHS
            int from = VOXEL(1   , y, z, nx, ny, nz, ng);
            int to   = VOXEL(nx+1, y, z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);

            // Copy x from RHS -> LHS
            from = VOXEL(nx  , y, z, nx, ny, nz, ng);
            to   = VOXEL(0   , y, z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);
        };
        //}

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> zy_policy({1,1}, {nz+1,ny+1});
        Kokkos::parallel_for( zy_policy, _zy_boundary, "zy boundary()" );

        //for (int x = 0; x < nx+2; x++) {
            //for (int z = 1; z < nz+1; z++) {
        auto _xz_boundary = KOKKOS_LAMBDA( const int x, const int z )
        {
            int from = VOXEL(x,    1, z, nx, ny, nz, ng);
            int to   = VOXEL(x, ny+1, z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);

            from = VOXEL(x, ny  , z, nx, ny, nz, ng);
            to   = VOXEL(x, 0   , z, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);
        };
        //}
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> xz_policy({0,1}, {nx+2,nz+1});
        Kokkos::parallel_for( xz_policy, _xz_boundary, "xz boundary()" );

        //for (int y = 0; y < ny+2; y++) {
            //for (int x = 0; x < nx+2; x++) {
        auto _yx_boundary = KOKKOS_LAMBDA( const int y, const int x )
        {
            int from = VOXEL(x, y, 1   , nx, ny, nz, ng);
            int to   = VOXEL(x, y, nz+1, nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);

            from = VOXEL(x, y, nz  , nx, ny, nz, ng);
            to   = VOXEL(x, y, 0   , nx, ny, nz, ng);

            slice_x(to) = slice_x(from);
            slice_y(to) = slice_y(from);
            slice_z(to) = slice_z(from);
        };
        //}
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> yx_policy({0,0}, {ny+2,nx+2});
        Kokkos::parallel_for( yx_policy, _yx_boundary, "yx boundary()" );
    }
}

template<class Slice_X, class Slice_Y, class Slice_Z>
//KOKKOS_INLINE_FUNCTION
void serial_update_ghosts(
        //field_array_t& fields,
        Slice_X slice_x,
        Slice_Y slice_y,
        Slice_Z slice_z,
        int nx, int ny, int nz, int ng)
{

    const Boundary boundary = deck.BOUNDARY_TYPE;
    if (boundary == Boundary::Reflect)
    {
        // TODO: this
        exit(1);
    }
    else { // assume periodic

        /*
           To fill in contributions from places of periodic BC
           */
        //TODO: theses again don't have a sensible amount of work

        //for ( x = 1; x <= nx; x++ ){
        auto _x_boundary = KOKKOS_LAMBDA( const int x )
        {
            for(int z = 1; z <= nz+1; z++){
                //y first
                int from = VOXEL(x, ny+1, z, nx, ny, nz, ng);
                int to   = VOXEL(x, 1   , z, nx, ny, nz, ng);
                slice_x(to) += slice_x(from);
            }

            for(int y = 1; y <= ny+1; y++){
                //z next
                int from = VOXEL(x, y, nz+1, nx, ny, nz, ng);
                int to   = VOXEL(x, y, 1   , nx, ny, nz, ng);
                slice_x(to) += slice_x(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> x_policy(1, nx+1);
        Kokkos::parallel_for( x_policy, _x_boundary, "_x_boundary()" );

        //for ( y = 1; y <= ny; y++ ){
        auto _y_boundary = KOKKOS_LAMBDA( const int y )
        {
            for (int x = 1; x <= nx+1; x++){
                //z first
                int from = VOXEL(x   , y, nz+1, nx, ny, nz, ng);
                int to   = VOXEL(x   , y, 1   , nx, ny, nz, ng);
                slice_y(to) += slice_y(from);
            }

            for (int z = 1; z <= nz+1; z++){
                //x next
                int from = VOXEL(nx+1, y, z   , nx, ny, nz, ng);
                int to   = VOXEL(1   , y, z   , nx, ny, nz, ng);
                slice_y(to) += slice_y(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> y_policy(1, ny+1);
        Kokkos::parallel_for( y_policy, _y_boundary, "_y_boundary()" );

        //for ( z = 1; z <= nz; z++ ){
        auto _z_boundary = KOKKOS_LAMBDA( const int z )
        {
            for (int y = 1; y <= ny+1; y++){
                //x first
                int from = VOXEL(nx+1, y   , z, nx, ny, nz, ng);
                int to   = VOXEL(1   , y   , z, nx, ny, nz, ng);
                slice_z(to) += slice_z(from);
            }

            for (int x = 1; x <= nx+1; x++){
                //y next
                int from = VOXEL(x   , ny+1, z, nx, ny, nz, ng);
                int to   = VOXEL(x   , 1   , z, nx, ny, nz, ng);
                slice_z(to) += slice_z(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> z_policy(1, nz+1);
        Kokkos::parallel_for( z_policy, _z_boundary, "_z_boundary()" );

        // // Copy x from RHS -> LHS
        // int x = 1;
        // // (1 .. nz+1)
        // for (int z = 1; z < nz+2; z++)
        //   {
        // 	for (int y = 1; y < ny+2; y++)
        //       {
        // 	    // TODO: loop over ng?
        // 	    int to = VOXEL(x, y, z, nx, ny, nz, ng);
        // 	    int from = VOXEL(nx+1, y, z, nx, ny, nz, ng);

        // 	    // Only copy jf? can we copy more values?
        // 	    // TODO: once we're in parallel this needs to be a second loop with
        // 	    // a buffer
        // 	    // Cache value to so we don't lose it during the update
        // 	    tmp_slice_y = slice_y(to);
        // 	    tmp_slice_z = slice_z(to);

        // 	    slice_y(to) += slice_y(from);
        // 	    slice_z(to) += slice_z(from);
        // 	    // printf("slice y %e = %e + %e \n", slice_y(to), tmp_slice_y, slice_y(from) );
        // 	    // printf("slice z %e = %e + %e \n", slice_z(to), tmp_slice_z, slice_z(from) );

        // 	    // printf("%e + %e = %e \n", tmp_slice_y, slice_y(from), slice_y(to) );

        // 	    // TODO: this could just be assignment to slice_y(to)
        // 	    slice_y(from) += tmp_slice_y;
        // 	    slice_z(from) += tmp_slice_z;

        // 	    // TODO: does this copy into the corners twice?
        //       }
        //   }

        // int y = 1;
        // for (int z = 1; z < nz+2; z++)
        //   {
        // 	for (int x = 1; x < nx+2; x++)
        //       {
        // 	    // TODO: loop over ng?
        // 	    int to = VOXEL(x, y, z, nx, ny, nz, ng);
        // 	    int from = VOXEL(x, ny+1, z, nx, ny, nz, ng);

        // 	    // Only copy jf? can we copy more values?
        // 	    // TODO: once we're in parallel this needs to be a second loop with
        // 	    // a buffer
        // 	    // Cache value to so we don't lose it during the update
        // 	    tmp_slice_x = slice_x(to);
        // 	    tmp_slice_z = slice_z(to);

        // 	    slice_x(to) += slice_x(from);
        // 	    slice_z(to) += slice_z(from);
        // 	    // printf("slice x %e = %e + %e \n", slice_x(to), tmp_slice_x, slice_x(from) );
        // 	    // printf("slice z %e = %e + %e \n", slice_z(to), tmp_slice_z, slice_z(from) );

        // 	    slice_x(from) += tmp_slice_x;
        // 	    slice_z(from) += tmp_slice_z;
        //       }
        //   }

        // int z = 1;
        // for (int y = 1; y < ny+2; y++)
        //   {
        // 	for (int x = 1; x < nx+2; x++)
        //       {
        // 	    // TODO: loop over ng?
        // 	    int to = VOXEL(x, y, z, nx, ny, nz, ng);
        // 	    int from = VOXEL(x, y, nz+1, nx, ny, nz, ng);

        // 	    // Only copy jf? can we copy more values?
        // 	    // TODO: once we're in parallel this needs to be a second loop with
        // 	    // a buffer
        // 	    // Cache value to so we don't lose it during the update
        // 	    tmp_slice_x = slice_x(to);
        // 	    tmp_slice_y = slice_y(to);

        // 	    slice_x(to) += slice_x(from);
        // 	    slice_y(to) += slice_y(from);
        // 	    // printf("slice x %e = %e + %e \n", slice_x(to), tmp_slice_x, slice_x(from) );
        // 	    // printf("slice y %e = %e + %e \n", slice_y(to), tmp_slice_y, slice_y(from) );

        // 	    slice_x(from) += tmp_slice_x;
        // 	    slice_y(from) += tmp_slice_y;
        //       }
        //   }

    }
}

template<class Slice_X, class Slice_Y, class Slice_Z>
//KOKKOS_INLINE_FUNCTION
void serial_fill_ghosts(
        //field_array_t& fields,
        Slice_X slice_x,
        Slice_Y slice_y,
        Slice_Z slice_z,
        int nx, int ny, int nz, int ng)
{

    const Boundary boundary = deck.BOUNDARY_TYPE;
    if (boundary == Boundary::Reflect)
    {
        // TODO: this
        exit(1);
    }
    else { // assume periodic

        /*
           To fill in contributions from places of periodic BC
           */
        //TODO: theses again don't have a sensible amount of work

        //for ( x = 1; x <= nx; x++ ){
        auto _x_boundary = KOKKOS_LAMBDA( const int x )
        {
            for(int z = 1; z <= nz+1; z++){
                //y first
                int from = VOXEL(x, 1, z, nx, ny, nz, ng);
                int to   = VOXEL(x, ny+1   , z, nx, ny, nz, ng);
                slice_x(to) = slice_x(from);
            }

            for(int y = 1; y <= ny+1; y++){
                //z next
                int from = VOXEL(x, y, 1, nx, ny, nz, ng);
                int to   = VOXEL(x, y, nz+1   , nx, ny, nz, ng);
                slice_x(to) = slice_x(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> x_policy(1, nx+1);
        Kokkos::parallel_for( x_policy, _x_boundary, "_x_boundary()" );

        //for ( y = 1; y <= ny; y++ ){
        auto _y_boundary = KOKKOS_LAMBDA( const int y )
        {
            for (int x = 1; x <= nx+1; x++){
                //z first
                int from = VOXEL(x   , y, 1, nx, ny, nz, ng);
                int to   = VOXEL(x   , y, nz+1   , nx, ny, nz, ng);
                slice_y(to) = slice_y(from);
            }

            for (int z = 1; z <= nz+1; z++){
                //x next
                int from = VOXEL(1, y, z   , nx, ny, nz, ng);
                int to   = VOXEL(nx+1   , y, z   , nx, ny, nz, ng);
                slice_y(to) = slice_y(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> y_policy(1, ny+1);
        Kokkos::parallel_for( y_policy, _y_boundary, "_y_boundary()" );

        //for ( z = 1; z <= nz; z++ ){
        auto _z_boundary = KOKKOS_LAMBDA( const int z )
        {
            for (int y = 1; y <= ny+1; y++){
                //x first
                int from = VOXEL(1, y   , z, nx, ny, nz, ng);
                int to   = VOXEL(nx+1   , y   , z, nx, ny, nz, ng);
                slice_z(to) = slice_z(from);
            }

            for (int x = 1; x <= nx+1; x++){
                //y next
                int from = VOXEL(x   , 1, z, nx, ny, nz, ng);
                int to   = VOXEL(x   , ny+1   , z, nx, ny, nz, ng);
                slice_z(to) = slice_z(from);
            }
        };
        Kokkos::RangePolicy<ExecutionSpace> z_policy(1, nz+1);
        Kokkos::parallel_for( z_policy, _z_boundary, "_z_boundary()" );
    }
}

// Policy base class
template<typename Solver_Type> class Field_Solver : public Solver_Type
{
    public:

        //constructor
        Field_Solver(field_array_t& fields)
        {
            // Zero the fields so everything has a safe value.
            // This occurs before we parse any custom fields in a user deck
            init(fields);
        }

        void init(field_array_t& fields)
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

            auto _init_fields =
                KOKKOS_LAMBDA( const int i )
                {
                    ex(i) = 0.0;
                    ey(i) = 0.0;
                    ez(i) = 0.0;
                    cbx(i) = 0.0;
                    cby(i) = 0.0;
                    cbz(i) = 0.0;
                    jfx(i) = 0.0;
                    jfy(i) = 0.0;
                    jfz(i) = 0.0;
                };

            Kokkos::parallel_for( fields.size(), _init_fields, "init_fields()" );
        }

        void advance_b(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            Solver_Type::advance_b( fields, px, py, pz, nx, ny, nz, ng);
        }
        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            Solver_Type::advance_e( fields, px, py, pz, nx, ny, nz, ng, dt_eps0);
        }
		  
		  // Given E_n and E_{n+1/2}, puts E_{n+1} into the array of E_{n+1/2}
		  void extend_e(
		  			field_array_t& fields_nph, 
					field_array_t& fields_n)
		  {
            auto ex = Cabana::slice<FIELD_EX>(fields_nph);
            auto ey = Cabana::slice<FIELD_EY>(fields_nph);
            auto ez = Cabana::slice<FIELD_EZ>(fields_nph);
            
            auto ex_old = Cabana::slice<FIELD_EX>(fields_n);
            auto ey_old = Cabana::slice<FIELD_EY>(fields_n);
            auto ez_old = Cabana::slice<FIELD_EZ>(fields_n);
				
				auto _extend_e = KOKKOS_LAMBDA( const int i )
            {
                ex(i) = ex(i) + ( ex(i) - ex_old(i) ) ;
                ey(i) = ey(i) + ( ey(i) - ey_old(i) ) ;
                ez(i) = ez(i) + ( ez(i) - ez_old(i) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields_nph.size() );
            Kokkos::parallel_for( exec_policy, _extend_e, "extend_e()" );
		  }
        
		  real_t e_energy(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto _e_energy = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
            {
                //lsum += ez(i)*ez(i);

                const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                lsum += ex(i)*ex(i) + ey(i)*ey(i) + ez(i)*ez(i);
            };

            real_t e_tot_energy=0;
            //Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            //Kokkos::parallel_reduce("e_energy", exec_policy, _e_energy, e_tot_energy );
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({1,1,1}, {nx+1,ny+1,nz+1});
            Kokkos::parallel_reduce("e_energy", exec_policy, _e_energy, e_tot_energy );
            return e_tot_energy*0.5f;
        }
};


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
			size_t ng
			)
		{
			if ( verbose ) { std::cout << "Starting SGCT" << std::endl; }
			constructSGCTcomponentGrids(fields, nx, ny, nz, ng); // Constructs the component grids in combination technique
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
				size_t ng
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
			for ( int i=0; i<=num_coarsenings; i++ ) {
				std::cout << "In plus grids, i = " << i << std::endl;
				while ( xres_p[i] < nx_out ) {
					d_grids_gen_p[i] = interpolate_on_axis(d_grids_gen_p[i], xres_p[i], yres_p[i], nz_out, ng, 0);
					xres_p[i] *= 2;
					std::cout << xres_p[i] << std::endl;
				}
				std::cout << "x-interpolation successful" << std::endl;
				while ( yres_p[i] < ny_out ) {
					d_grids_gen_p[i] = interpolate_on_axis(d_grids_gen_p[i], xres_p[i], yres_p[i], nz_out, ng, 1);
					yres_p[i] *= 2;
				}
				std::cout << "y-interpolation successful" << std::endl;
				
			}
			if ( verbose ) { std::cout << "Interpolated plus grids" << std::endl; }

			// Same for the "minus" diagonal
			for ( int i=0; i<num_coarsenings; i++ ) {
				while ( xres_m[i] < nx_out ) {
					d_grids_gen_m[i] = interpolate_on_axis(d_grids_gen_m[i], xres_m[i], yres_m[i], nz_out, ng, 0);
					xres_m[i] *= 2;
				}
				while ( yres_m[i] < ny_out ) {
					d_grids_gen_m[i] = interpolate_on_axis(d_grids_gen_m[i], xres_m[i], yres_m[i], nz_out, ng, 1);
					yres_m[i] *= 2;
				}
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


// FIXME: Field_solver is repeated => bad naming
/*class ES_Field_Solver
{
    public:

        void advance_b(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            // No-op, becasue ES
        }

        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

	    serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);

	    //remove the average (1D problems only)
	    assert(nx*ny==1||ny*nz==1||nz*nx==1);
	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> reduce_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});
	    size_t n_inner_cell = nx*ny*nz;
	    real_t jx_avg = 0,jy_avg=0,jz_avg=0;
	    if(nx==1||ny==1||nz==1){
		if(nx>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfx(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", reduce_exec_policy, _find_javg, jx_avg );
		    jx_avg /=n_inner_cell;
		}

		if(ny>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfy(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", reduce_exec_policy, _find_javg, jy_avg );
		    jy_avg /=n_inner_cell;
		}

		if(nz>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfz(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", reduce_exec_policy, _find_javg, jz_avg );
		    jz_avg /=n_inner_cell;
		}
	    }

            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                const real_t cj =dt_eps0;
                ex(i) = ex(i) + ( - cj * (jfx(i)-jx_avg) ) ;
                ey(i) = ey(i) + ( - cj * (jfy(i)-jy_avg) ) ;
                ez(i) = ez(i) + ( - cj * (jfz(i)-jz_avg) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_e, "es_advance_e()" );
        }

        real_t e_energy(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
		size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto _e_energy = KOKKOS_LAMBDA( const int i, real_t & lsum )
            {
                lsum += ex(i) * ex(i)
                    +ey(i) * ey(i)
                    +ez(i) * ez(i);
            };

            real_t e_tot_energy=0;
            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_reduce("es_e_energy_1d()", exec_policy, _e_energy, e_tot_energy );
            return e_tot_energy*0.5f;
        }

            void dump_fields(FILE * fp,
                field_array_t& d_fields,
                real_t xmin,
                real_t ymin,
                real_t zmin,
                real_t dx,
                real_t dy,
                real_t dz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
	    //TODO: implement this
	}
};*/

class ES_Field_Solver
{
    public:

        void advance_b(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            // No-op, becasue ES
        }

        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

	    serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);

	    size_t n_inner_cell = nx*ny*nz;
	    real_t iLx = px/(nx*dt_eps0);
	    real_t iLy = py/(ny*dt_eps0);
	    real_t iLz = pz/(nz*dt_eps0);

	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> fft_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});

	    //for many ffts
	    int rank = 3;
	    int n[3] = {(int) nx, (int) ny, (int) nz};
	    int depth = 3;
	    int idist = 1, odist = 1;
	    int istride = 3,ostride = 3;
	    int *inembed = n, *onembed = n;

	    ViewVecComplex fft_coefficients("fft_coef", n_inner_cell*3);
	    ViewVecComplex fft_out("fft_out", n_inner_cell*3);
	    
	    //start fft
#ifdef USE_GPU
	    // CUFFT plans
	    cufftHandle forward_plan, inverse_plan;
	    cufftPlanMany(&forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, depth);

	    cufftPlanMany(&inverse_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, depth);
#else
	    	    //plan
	    fftwf_plan plan_fft = fftwf_plan_many_dft (rank, //rank
						       n, //dims -- this doesn't include zero-padding
						       depth, //howmany
						       reinterpret_cast<fftwf_complex *>(fft_coefficients.data()), //in
						       inembed, //inembed
						       depth, //istride
						       idist, //idist
						       reinterpret_cast<fftwf_complex *>(fft_out.data()), //out
						       onembed, //onembed
						       depth, //ostride
						       odist, //odist
						       FFTW_FORWARD,  
						       FFTW_ESTIMATE );
	    
	    fftwf_plan plan_ifft = fftwf_plan_many_dft (rank, //rank
						       n, //dims -- this doesn't include zero-padding
						       depth, //howmany
							reinterpret_cast<fftwf_complex *>(fft_out.data()), //in
						       inembed, //inembed
						       depth, //istride
						       idist, //idist
							reinterpret_cast<fftwf_complex *>(fft_coefficients.data()), //out
						       onembed, //onembed
						       depth, //ostride
						       odist, //odist
						       FFTW_BACKWARD,  
						       FFTW_ESTIMATE );
#endif

	    auto _find_jf_fft = KOKKOS_LAMBDA( const int i, const int j, const int k  ){ //for(int i=0; i<ny; ++i){
		const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
		const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
		fft_coefficients(3*f0+0) = jfx(f1);
		fft_coefficients(3*f0+1) = jfy(f1);
		fft_coefficients(3*f0+2) = jfz(f1);

		//printf ("real part = %15.8e  imag part = %15.8e\n", fft_coefficients[i].x, fft_coefficients[i].y);
	    };   

	    Kokkos::parallel_for("find_jf_fft", fft_exec_policy, _find_jf_fft );

#ifdef USE_GPU	    
	    // Transform signal 
	    cufftExecC2C(forward_plan, (cufftComplex *)(fft_coefficients.data()), (cufftComplex *)(fft_out.data()), CUFFT_FORWARD);
#else
	    fftwf_execute(plan_fft);
#endif	    
	    //Perform div,poisson solve, and gradient in Fourier space
	    auto _DPG_Fourier = KOKKOS_LAMBDA(  const int i, const int j, const int k ){ 
		int i0 = i-ng;
		int j0 = j-ng;
		int k0 = k-ng;

		real_t kx,ky,kz;
		const int f0 = VOXEL(i0,   j0,   k0,   nx, ny, nz, 0);
		//d(jfx)/dx
		if(i0<(nx+1)/2) {
		    kx = (real_t)i0*2.0*M_PI*iLx;
		}else{
		    kx = (real_t)(nx-i0)*2.0*M_PI*iLx;
		}

		//d(jfy)/dy
		if(j0<(ny+1)/2) {
		    ky = (real_t)j0*2.0*M_PI*iLy;
		}else{
		    ky = (real_t)(ny-j0)*2.0*M_PI*iLy;
		}

		//d(jfz)/dz
		if(k0<(nz+1)/2) {
		    kz = (real_t)k0*2.0*M_PI*iLz;
		}else{
		    kz = (real_t)(nz-k0)*2.0*M_PI*iLz;
		}
		
		Kokkos::complex<real_t> kdotj = fft_out(3*f0+0)*kx+fft_out(3*f0+1)*ky+fft_out(3*f0+2)*kz;
		real_t k2 = kx*kx+ky*ky+kz*kz;
		if(i0==0&&j0==0&&k0==0) k2 = 1.;
		Kokkos::complex<real_t> phif = kdotj/k2;
		if(i0==0&&j0==0&&k0==0) phif = 0.;
		fft_out(3*f0+0)=kx*phif;
		fft_out(3*f0+1)=ky*phif;
		fft_out(3*f0+2)=kz*phif;
	    };

	    Kokkos::parallel_for("DPG_Fourier", fft_exec_policy, _DPG_Fourier );

#ifdef USE_GPU
	    // Transform signal back
	    cufftExecC2C(inverse_plan, (cufftComplex *)(fft_out.data()), (cufftComplex *)(fft_coefficients.data()), CUFFT_INVERSE);
	    cufftDestroy(inverse_plan);
	    cufftDestroy(forward_plan);
#else
	    fftwf_execute(plan_ifft);
	    fftwf_destroy_plan(plan_fft);
	    fftwf_destroy_plan(plan_ifft);
#endif

	    auto _find_jf_ifft = KOKKOS_LAMBDA( const int i, const int j, const int k  ){ //for(int i=0; i<ny; ++i){
		const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
	    	const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
	    	jfx(f1) = fft_coefficients(3*f0+0).real()/n_inner_cell;
	    	jfy(f1) = fft_coefficients(3*f0+1).real()/n_inner_cell;
	    	jfz(f1) = fft_coefficients(3*f0+2).real()/n_inner_cell;
	    };

	    Kokkos::parallel_for("find_jf_ifft", fft_exec_policy, _find_jf_ifft );	    
	    //end fft

	    serial_fill_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);

	    //remove the average (for 1D problems only)
	    real_t jx_avg = 0,jy_avg=0,jz_avg=0;
	    /*
	    if(nx==1||ny==1||nz==1){
		if(nx>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfx(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jx_avg );
		    jx_avg /=n_inner_cell;
		}

		if(ny>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfy(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jy_avg );
		    jy_avg /=n_inner_cell;
		}

		if(nz>1){
		    auto _find_javg = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
			{
			    const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
			    lsum += jfz(i);
			};

		    Kokkos::parallel_reduce("find_jz_avg", fft_exec_policy, _find_javg, jz_avg );
		    jz_avg /=n_inner_cell;
		}
	    }
	    */
            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
	    const real_t cjx = (nx>1)?dt_eps0:0;
	    const real_t cjy = (ny>1)?dt_eps0:0;
	    const real_t cjz = (nz>1)?dt_eps0:0;
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                ex(i) = ex(i) + ( - cjx * (jfx(i)-jx_avg) ) ;
                ey(i) = ey(i) + ( - cjy * (jfy(i)-jy_avg) ) ;
                ez(i) = ez(i) + ( - cjz * (jfz(i)-jz_avg) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_e, "es_advance_e()" );
        }

        /*real_t e_energy(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
		size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto _e_energy = KOKKOS_LAMBDA( const int i, real_t & lsum )
            {
                lsum += ex(i) * ex(i)
                    +ey(i) * ey(i)
                    +ez(i) * ez(i);
            };

            real_t e_tot_energy=0;
            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_reduce("es_e_energy_1d()", exec_policy, _e_energy, e_tot_energy );
            return e_tot_energy*0.5f;
        }*/

            void dump_fields(FILE * fp,
                field_array_t& d_fields,
                real_t xmin,
                real_t ymin,
                real_t zmin,
                real_t dx,
                real_t dy,
                real_t dz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
	    //TODO: implement this
	}
};

class ES_Field_Solver_1D
{
    public:

        /*real_t e_energy(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto _e_energy = KOKKOS_LAMBDA( const int i, real_t & lsum )
            {
                lsum += ex(i) * ex(i)
                    +ey(i) * ey(i)
                    +ez(i) * ez(i);
            };

            real_t e_tot_energy=0;
            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_reduce("es_e_energy_1d()", exec_policy, _e_energy, e_tot_energy );
            return e_tot_energy*0.5f;
        }*/

        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);

            //serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);

            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                const real_t cj = dt_eps0;
                ex(i) = ex(i) + ( - cj * jfx(i) ) ;
                ey(i) = ey(i) + ( - cj * jfy(i) ) ;
                ez(i) = ez(i) + ( - cj * jfz(i) ) ;
            };

            Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            Kokkos::parallel_for( exec_policy, _advance_e, "es_advance_e_1d()" );
        }
};

// EM HERE: UNFINISHED
// TODO: Finish

class EM_Field_Solver
{
    public:

        //how to formalize/generalize this?

        // TODO: is this the right place for this vs in the helper?
        void dump_fields(FILE * fp,
                field_array_t& d_fields,
                real_t xmin,
                real_t ymin,
                real_t zmin,
                real_t dx,
                real_t dy,
                real_t dz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            // Host
            field_array_t::host_mirror_type fields("host_fields", d_fields.size());

            // Copy device field to host
            Cabana::deep_copy(fields, d_fields);

            auto ex = Cabana::slice<FIELD_EX>(fields);

            for( size_t i=1; i<nx+1; i++ )
            {
                real_t x = xmin + (i-0.5)*dx;
                size_t ii = VOXEL(i,1,1,nx,ny,nz,ng);
                //	  fprintf(fp,"%e %e %e %e %e %e %e\n",x,y,ey(ii),jfx(ii),jfy(ii),jfz(ii),cbz(ii));
                fprintf(fp,"%e %e\n",x,ex(ii));
            }

            fprintf(fp,"\n\n");

        }

        /*real_t e_energy(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto _e_energy = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
            {
                //lsum += ez(i)*ez(i);

                const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                lsum += ex(i)*ex(i) + ey(i)*ey(i) + ez(i)*ez(i);
            };

            real_t e_tot_energy=0;
            //Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
            //Kokkos::parallel_reduce("e_energy", exec_policy, _e_energy, e_tot_energy );
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({1,1,1}, {nx+1,ny+1,nz+1});
            Kokkos::parallel_reduce("e_energy", exec_policy, _e_energy, e_tot_energy );
            real_t dV = 1.0; //Parameters::instance().dx * Parameters::instance().dy * Parameters::instance().dz;
            return e_tot_energy*0.5f*dV;
        }*/

        real_t b_energy(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto _b_energy = KOKKOS_LAMBDA( const int x, const int y, const int z, real_t & lsum )
            {
                const int i = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                lsum += cbx(i)*cbx(i) + cby(i)*cby(i) + cbz(i)*cbz(i);
            };

            real_t b_tot_energy=0;
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({1,1,1}, {nx+1,ny+1,nz+1});
            Kokkos::parallel_reduce("b_energy", exec_policy, _b_energy, b_tot_energy );
            //TODO: no access to parameters here
            real_t dV = 1.0; //Parameters::instance().dx * Parameters::instance().dy * Parameters::instance().dz;
            return b_tot_energy*0.5f*dV;
        }


        void advance_e(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng,
                real_t dt_eps0
                )
        {
            //    auto ng = Parameters::instance().num_ghosts;
            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);
            auto jfx = Cabana::slice<FIELD_JFX>(fields);
            auto jfy = Cabana::slice<FIELD_JFY>(fields);
            auto jfz = Cabana::slice<FIELD_JFZ>(fields);
            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);


            //serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            //serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);
            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int x, const int y, const int z)
            {
                const real_t cj = dt_eps0;

                const int f0 = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                const int fx = VOXEL(x-1, y,   z,   nx, ny, nz, ng);
                const int fy = VOXEL(x,   y-1, z,   nx, ny, nz, ng);
                const int fz = VOXEL(x,   y,   z-1, nx, ny, nz, ng);

                ex(f0) = ex(f0) + ( - cj * jfx(f0) ) + ( py * (cbz(f0) - cbz(fy)) - pz * (cby(f0) - cby(fz)) );
                ey(f0) = ey(f0) + ( - cj * jfy(f0) ) + ( pz * (cbx(f0) - cbx(fz)) - px * (cbz(f0) - cbz(fx)) );
                ez(f0) = ez(f0) + ( - cj * jfz(f0) ) + ( px * (cby(f0) - cby(fx)) - py * (cbx(f0) - cbx(fy)) );

                //ex(f0) +=  ( - cj * jfx(f0) ) + ( py * (cbz(f0) - cbz(fy)) );

            };

            Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
            Kokkos::parallel_for( zyx_policy, _advance_e, "advance_e()" );
        }


        void advance_b(
                field_array_t& fields,
                real_t px,
                real_t py,
                real_t pz,
                size_t nx,
                size_t ny,
                size_t nz,
                size_t ng
                )
        {
            //f0 = &f(x,  y,  z  );
            //fx = &f(x+1,y,  z  );
            //fy = &f(x,  y+1,z  );
            //fz = &f(x,  y,  z+1);

            auto ex = Cabana::slice<FIELD_EX>(fields);
            auto ey = Cabana::slice<FIELD_EY>(fields);
            auto ez = Cabana::slice<FIELD_EZ>(fields);

            auto cbx = Cabana::slice<FIELD_CBX>(fields);
            auto cby = Cabana::slice<FIELD_CBY>(fields);
            auto cbz = Cabana::slice<FIELD_CBZ>(fields);

            auto _advance_b = KOKKOS_LAMBDA( const int x, const int y, const int z)
            {

                // Update value
                /*
                   f0->cbx -= ( py*( fy->ez-f0->ez ) - pz*( fz->ey-f0->ey ) );
                   f0->cby -= ( pz*( fz->ex-f0->ex ) - px*( fx->ez-f0->ez ) );
                   f0->cbz -= ( px*( fx->ey-f0->ey ) - py*( fy->ex-f0->ex ) );
                   */

                const int f0 = VOXEL(x,   y,   z,   nx, ny, nz, ng);
                const int fx = VOXEL(x+1, y,   z,   nx, ny, nz, ng);
                const int fy = VOXEL(x,   y+1, z,   nx, ny, nz, ng);
                const int fz = VOXEL(x,   y,   z+1, nx, ny, nz, ng);

                cbx(f0) -= ( py*( ez(fy) - ez(f0) ) - pz*( ey(fz) - ey(f0) ) );
                cby(f0) -= ( pz*( ex(fz) - ex(f0) ) - px*( ez(fx) - ez(f0) ) );
                cbz(f0) -= ( px*( ey(fx) - ey(f0) ) - py*( ex(fy) - ex(f0) ) );


                //cbz(f0) -= - py*( ex(fy) - ex(f0) ) ;

            };

            Kokkos::MDRangePolicy<Kokkos::Rank<3>> zyx_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
            Kokkos::parallel_for( zyx_policy, _advance_b, "advance_b()" );
            serial_update_ghosts_B(cbx, cby, cbz, nx, ny, nz, ng);
        }
};

// Requires C++14
static auto make_field_solver(field_array_t& fields)
{
    // TODO: make this support 1/2/3d
#ifdef ES_FIELD_SOLVER
    std::cout << "Created ES Solver" << std::endl;
    Field_Solver<ES_Field_Solver> field_solver(fields);
#else // EM
    std::cout << "Created EM Solver" << std::endl;
    Field_Solver<EM_Field_Solver> field_solver(fields);
#endif
    return field_solver;
}

template<typename field_solver_t>
real_t dump_energies(
	const particle_list_t& particles,
        field_solver_t& field_solver,
        field_array_t& fields,
        int step,
        real_t time,
        real_t px,
        real_t py,
        real_t pz,
        size_t nx,
        size_t ny,
        size_t nz,
        size_t ng,
	real_t dV,
	real_t tot_en0=0
        )
{
    auto vx = Cabana::slice<VelocityX>(particles);
    auto vy = Cabana::slice<VelocityY>(particles);
    auto vz = Cabana::slice<VelocityZ>(particles);

    auto weight = Cabana::slice<Weight>( particles );

    // compute total kinetic energy
    auto _k_energy = KOKKOS_LAMBDA( const int i, real_t & lsum )
	{
	    lsum += weight(i)*( vx(i) * vx(i)
				+vy(i) * vy(i)
				+vz(i) * vz(i) );
	};

    real_t k_tot_energy=0, tot_energy,den;
    Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, particles.size() );
    Kokkos::parallel_reduce("k_energy()", exec_policy, _k_energy, k_tot_energy );
    k_tot_energy = 0.5*k_tot_energy;


    real_t e_en = dV*field_solver.e_energy(fields, px, py, pz, nx, ny, nz, ng);
    tot_energy = e_en+k_tot_energy; //for ES
    if(step==0) den = 0;	
    else den = tot_energy - tot_en0;
	
    
    // Print energies to screen *and* dump them to disk
    // TODO: is it ok to keep opening and closing the file like this?
    // one per time step is probably fine?
    std::ofstream energy_file;

    if (step == 0)
    {
        // delete what is there
        energy_file.open("energies.txt", std::ofstream::out | std::ofstream::trunc);
    }
    else {
        energy_file.open("energies.txt", std::ios::app); // append
    }

    energy_file << step << " " << time << " " << e_en<<" "<<k_tot_energy<<" "<<den;
#ifndef ES_FIELD_SOLVER
    // Only write b info if it's available
    real_t b_en = field_solver.b_energy(fields, px, py, pz, nx, ny, nz, ng);
    energy_file << " " << b_en;
    printf("%d %f %e %e\n",step, time, e_en, b_en);
#else
    printf("%d %f %e %e %e\n",step, time, e_en,k_tot_energy,den);
#endif
    energy_file << std::endl;
    energy_file.close();
    return tot_energy;
}

#endif // pic_EM_fields_h
