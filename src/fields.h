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
};

// FIXME: Field_solver is repeated => bad naming
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

#ifdef USE_GPU
	    size_t SIGNAL_SIZE = ny;
	    int mem_size = sizeof(cufftComplex) * SIGNAL_SIZE;
	    // Allocate device memory for signal
	    cufftComplex* fft_coefficients;
	    cufftComplex* fft_out;
	    cudaMalloc((void**)&fft_coefficients, mem_size);
	    cudaMalloc((void**)&fft_out, mem_size);	    

	    auto _find_jf_fft_1dy = KOKKOS_LAMBDA( const int i ){ //for(int i=0; i<ny; ++i){
		const int f0 = VOXEL(1,   i+ng,   1,   nx, ny, nz, ng);
		fft_coefficients[i].x = jfx(f0);
		fft_coefficients[i].y = 0;
		//printf ("real part = %15.8e  imag part = %15.8e\n", fft_coefficients[i].x, fft_coefficients[i].y);
	    };   
            Kokkos::RangePolicy<ExecutionSpace> exec_policy_1dy( 0, ny );
	    Kokkos::parallel_for( exec_policy_1dy, _find_jf_fft_1dy, "find_jf_fft_1dy" );

	    // CUFFT plan
	    cufftHandle plan;
	    cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1);
	    // Transform signal 
	    cufftExecC2C(plan, (cufftComplex *)fft_coefficients, (cufftComplex *)fft_out, CUFFT_FORWARD);

	    // Transform signal back
	    cufftExecC2C(plan, (cufftComplex *)fft_out, (cufftComplex *)fft_coefficients, CUFFT_INVERSE);

	    auto _find_jf_1dy = KOKKOS_LAMBDA( const int i ){ //for(int i=0; i<ny; ++i){
		const int f0 = VOXEL(1,   i+ng,   1,   nx, ny, nz, ng);
		jfx(f0) = fft_coefficients[i].x/SIGNAL_SIZE;
	    };   

            Kokkos::parallel_for( exec_policy_1dy, _find_jf_1dy, "find_jf_1dy" );

	    cudaFree(fft_coefficients);
	    cudaFree(fft_out);
	    
#else //on CPU

	    auto fft_coefficients = new std::complex<real_t>[n_inner_cell];
	    auto fft_out = new std::complex<real_t>[n_inner_cell];


	    // auto _find_jf_fft_1dy = KOKKOS_LAMBDA( const int i ){ //for(int i=0; i<ny; ++i){
	    // 	const int f0 = VOXEL(1,   i+ng,   1,   nx, ny, nz, ng);
	    // 	fft_coefficients[i] = jfx(f0);
	    // 	//std::cout<<i<<" "<<fft_coefficients[i]<<std::endl;
	    // };   
	    Kokkos::RangePolicy<ExecutionSpace> exec_policy_1dy( 0, ny );
	    //Kokkos::parallel_for( exec_policy_1dy, _find_jf_fft_1dy, "find_jf_fft_1dy" );

	    auto _find_jf_fft = KOKKOS_LAMBDA( const int i, const int j, const int k ){ 
		const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
		const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
		fft_coefficients[f0] = jfy(f1);
		//std::cout<<i<<" "<<fft_coefficients[i]<<std::endl;
	    };   

	    Kokkos::MDRangePolicy<Kokkos::Rank<3>> fft_exec_policy({ng,ng,ng}, {nx+ng,ny+ng,nz+ng});
	    Kokkos::parallel_for("find_jf_fft", fft_exec_policy, _find_jf_fft );

	    fftwf_plan plan_fft = fftwf_plan_dft_3d(nx,ny,nz, reinterpret_cast<fftwf_complex *>(fft_coefficients),
						  reinterpret_cast<fftwf_complex *>(fft_out), FFTW_FORWARD,  FFTW_ESTIMATE);
						 
	    fftwf_execute(plan_fft);

	    fftwf_plan plan_ifft = fftwf_plan_dft_3d(nx,ny,nz, reinterpret_cast<fftwf_complex *>(fft_out),
						  reinterpret_cast<fftwf_complex *>(fft_coefficients), FFTW_BACKWARD,  FFTW_ESTIMATE);

	    fftwf_execute(plan_ifft);

	    //std::cout<<std::endl;
	    // auto _find_jf_1dy = KOKKOS_LAMBDA( const int i ){ //for(int i=0; i<ny; ++i){
	    // 	const int f0 = VOXEL(1,   i+ng,   1,   nx, ny, nz, ng);
	    // 	jfy(f0) = fft_coefficients[i].real()/ny;
	    // 	//std::cout<<i<<" "<<fft_out[i]<<std::endl;
	    // };   

            // Kokkos::parallel_for( exec_policy_1dy, _find_jf_1dy, "find_jf_1dy" );

	    auto _find_jf_ifft = KOKKOS_LAMBDA( const int i, const int j, const int k ){ 
	    	const int f1 = VOXEL(i,   j,   k,   nx, ny, nz, ng);
	    	const int f0 = VOXEL(i-ng,   j-ng,   k-ng,   nx, ny, nz, 0); //fft_coefficients does not include ghost cells
	    	jfy(f1) = fft_coefficients[f0].real()/n_inner_cell;
	    	//std::cout<<i<<" "<<fft_coefficients[i]<<std::endl;
	    };   

	    Kokkos::parallel_for("find_jf_ifft", fft_exec_policy, _find_jf_ifft );


	    delete[] fft_coefficients;
	    delete[] fft_out;
#endif	    

            // NOTE: this does work on ghosts that is extra, but it simplifies
            // the logic and is fairly cheap
            auto _advance_e = KOKKOS_LAMBDA( const int i )
            {
                const real_t cj =dt_eps0;
                ex(i) = ex(i) + ( - cj * jfx(i) ) ;
                ey(i) = ey(i) + ( - cj * jfy(i) ) ;
                ez(i) = ez(i) + ( - cj * jfz(i) ) ;
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
};


class ES_Field_Solver_1D
{
    public:

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

            serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);

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
            real_t dV = 1.0; //Parameters::instance().dx * Parameters::instance().dy * Parameters::instance().dz;
            return e_tot_energy*0.5f*dV;
        }

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


            serial_update_ghosts(jfx, jfy, jfz, nx, ny, nz, ng);
            serial_update_ghosts_B(jfx, jfy, jfz, nx, ny, nz, ng);
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
void dump_energies(
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
        size_t ng
        )
{
    real_t e_en = field_solver.e_energy(fields, px, py, pz, nx, ny, nz, ng);
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

    energy_file << step << " " << time << " " << e_en;
#ifndef ES_FIELD_SOLVER
    // Only write b info if it's available
    real_t b_en = field_solver.b_energy(fields, px, py, pz, nx, ny, nz, ng);
    energy_file << " " << b_en;
    printf("%d %f %e %e\n",step, time, e_en, b_en);
#else
    printf("%d %f %e\n",step, time, e_en);
#endif
    energy_file << std::endl;
    energy_file.close();
}

#endif // pic_EM_fields_h
