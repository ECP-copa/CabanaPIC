#ifndef pic_EM_fields_h
#define pic_EM_fields_h

#include "Cabana_Parallel.hpp" // Simd parallel for
#include "input/deck.h"

// TODO: Namespace this stuff?


template<class Slice_X, class Slice_Y, class Slice_Z>
KOKKOS_INLINE_FUNCTION
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
				int from, to;

				for (int z = 1; z < nz+1; z++)
				{
						for (int y = 1; y < ny+1; y++)
						{
								// Copy x from LHS -> RHS
								from = VOXEL(1   , y, z, nx, ny, nz, ng);
								to   = VOXEL(nx+1, y, z, nx, ny, nz, ng);

								slice_x(to) = slice_x(from);
								slice_y(to) = slice_y(from);
								slice_z(to) = slice_z(from);

								// Copy x from RHS -> LHS
								from = VOXEL(nx  , y, z, nx, ny, nz, ng);
								to   = VOXEL(0   , y, z, nx, ny, nz, ng);

								slice_x(to) = slice_x(from);
								slice_y(to) = slice_y(from);
								slice_z(to) = slice_z(from);
						}
				}


				for (int x = 0; x < nx+2; x++)
				{
						for (int z = 1; z < nz+1; z++)
						{
								from = VOXEL(x,    1, z, nx, ny, nz, ng);
								to   = VOXEL(x, ny+1, z, nx, ny, nz, ng);

								slice_x(to) = slice_x(from);
								slice_y(to) = slice_y(from);
								slice_z(to) = slice_z(from);

								from = VOXEL(x, ny  , z, nx, ny, nz, ng);
								to   = VOXEL(x, 0   , z, nx, ny, nz, ng);

								slice_x(to) = slice_x(from);
								slice_y(to) = slice_y(from);
								slice_z(to) = slice_z(from);
						}
				}


				for (int y = 0; y < ny+2; y++)
				{
						for (int x = 0; x < nx+2; x++)
						{
								from = VOXEL(x, y, 1   , nx, ny, nz, ng);
								to   = VOXEL(x, y, nz+1, nx, ny, nz, ng);

								slice_x(to) = slice_x(from);
								slice_y(to) = slice_y(from);
								slice_z(to) = slice_z(from);

								from = VOXEL(x, y, nz  , nx, ny, nz, ng);
								to   = VOXEL(x, y, 0   , nx, ny, nz, ng);

								slice_x(to) = slice_x(from);
								slice_y(to) = slice_y(from);
								slice_z(to) = slice_z(from);
						}
				}
		}
}

template<class Slice_X, class Slice_Y, class Slice_Z>
KOKKOS_INLINE_FUNCTION
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

				int x,y,z,from,to;
				for ( x = 1; x <= nx; x++){
						//y first
						from = VOXEL(x, ny+1, 1, nx, ny, nz, ng);
						to   = VOXEL(x, 1   , 1, nx, ny, nz, ng);
						slice_x(to) += slice_x(from);


						from = VOXEL(x, ny+1, nz+1, nx, ny, nz, ng);
						to   = VOXEL(x, 1   , nz+1, nx, ny, nz, ng);
						slice_x(to) += slice_x(from);


						//z next
						from = VOXEL(x, 1, nz+1, nx, ny, nz, ng);
						to   = VOXEL(x, 1, 1   , nx, ny, nz, ng);
						slice_x(to) += slice_x(from);

				}

				for ( y = 1; y <= ny; y++){
						//z first
						from = VOXEL(1   , y, nz+1, nx, ny, nz, ng);
						to   = VOXEL(1   , y, 1   , nx, ny, nz, ng);
						slice_y(to) += slice_y(from);


						from = VOXEL(nx+1, y, nz+1, nx, ny, nz, ng);
						to   = VOXEL(nx+1, y, 1   , nx, ny, nz, ng);
						slice_y(to) += slice_y(from);


						//x next
						from = VOXEL(nx+1, y, 1   , nx, ny, nz, ng);
						to   = VOXEL(1   , y, 1   , nx, ny, nz, ng);
						slice_y(to) += slice_y(from);
				}

				for ( z = 1; z <= nz; z++){
						//x first
						from = VOXEL(nx+1, 1   , z, nx, ny, nz, ng);
						to   = VOXEL(1   , 1   , z, nx, ny, nz, ng);
						slice_z(to) += slice_z(from);


						from = VOXEL(nx+1, ny+1, z, nx, ny, nz, ng);
						to   = VOXEL(1   , ny+1, z, nx, ny, nz, ng);
						slice_z(to) += slice_z(from);


						//y next
						from = VOXEL(1   , ny+1, z, nx, ny, nz, ng);
						to   = VOXEL(1   , 1   , z, nx, ny, nz, ng);
						slice_z(to) += slice_z(from);
				}



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
				// 	    float tmp_slice_y = slice_y(to);
				// 	    float tmp_slice_z = slice_z(to);

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
				// 	    float tmp_slice_x = slice_x(to);
				// 	    float tmp_slice_z = slice_z(to);

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
				// 	    float tmp_slice_x = slice_x(to);
				// 	    float tmp_slice_y = slice_y(to);

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
						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();

						auto cbx = fields.slice<FIELD_CBX>();
						auto cby = fields.slice<FIELD_CBY>();
						auto cbz = fields.slice<FIELD_CBZ>();

						auto _init_fields =
								KOKKOS_LAMBDA( const int i )
								{
										ex(i) = 0.0;
										ey(i) = 0.0;
										ez(i) = 0.0;
										cbx(i) = 0.0;
										cby(i) = 0.0;
										cbz(i) = 0.0;
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
						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();

						auto cbx = fields.slice<FIELD_CBX>();
						auto cby = fields.slice<FIELD_CBY>();
						auto cbz = fields.slice<FIELD_CBZ>();

						auto jfx = fields.slice<FIELD_JFX>();
						auto jfy = fields.slice<FIELD_JFY>();
						auto jfz = fields.slice<FIELD_JFZ>();

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
								size_t nz
        )
				{
						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();
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
								size_t nz
        )
				{
						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();
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
						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();
						auto jfx = fields.slice<FIELD_JFX>();
						auto jfy = fields.slice<FIELD_JFY>();
						auto jfz = fields.slice<FIELD_JFZ>();

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

				real_t e_energy(
								field_array_t& fields,
								real_t px,
								real_t py,
								real_t pz,
								size_t nx,
								size_t ny,
								size_t nz
								)
				{
						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();
						auto _e_energy = KOKKOS_LAMBDA( const int i, real_t & lsum )
						{
								//lsum += ez(i)*ez(i);
								lsum += ex(i)*ex(i) + ey(i)*ey(i) + ez(i)*ez(i);
						};

						real_t e_tot_energy=0;
						Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
						Kokkos::parallel_reduce("e_energy", exec_policy, _e_energy, e_tot_energy );
						return e_tot_energy*0.5f;
				}

				real_t b_energy(
								field_array_t& fields,
								real_t px,
								real_t py,
								real_t pz,
								size_t nx,
								size_t ny,
								size_t nz
								)
				{
						auto cbx = fields.slice<FIELD_CBX>();
						auto cby = fields.slice<FIELD_CBY>();
						auto cbz = fields.slice<FIELD_CBZ>();

						auto _b_energy = KOKKOS_LAMBDA( const int i, real_t & lsum )
						{
								lsum += cbx(i)*cbx(i) + cby(i)*cby(i) + cbz(i)*cbz(i);
						};

						real_t b_tot_energy=0;
						Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
						Kokkos::parallel_reduce("b_energy", exec_policy, _b_energy, b_tot_energy );
						return b_tot_energy*0.5f;
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
						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();
						auto jfx = fields.slice<FIELD_JFX>();
						auto jfy = fields.slice<FIELD_JFY>();
						auto jfz = fields.slice<FIELD_JFZ>();
						auto cbx = fields.slice<FIELD_CBX>();
						auto cby = fields.slice<FIELD_CBY>();
						auto cbz = fields.slice<FIELD_CBZ>();


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

						auto ex = fields.slice<FIELD_EX>();
						auto ey = fields.slice<FIELD_EY>();
						auto ez = fields.slice<FIELD_EZ>();

						auto cbx = fields.slice<FIELD_CBX>();
						auto cby = fields.slice<FIELD_CBY>();
						auto cbz = fields.slice<FIELD_CBZ>();

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
    std::cout << "Initialized ES Solver" << std::endl;
    Field_Solver<ES_Field_Solver> field_solver(fields);
#else // EM
    std::cout << "Initialized EM Solver" << std::endl;
    Field_Solver<EM_Field_Solver> field_solver(fields);
#endif
    return field_solver;
}

#endif // pic_EM_fields_h
