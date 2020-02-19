This is to test the performance of the G2P kernel of the PIC algorithm when
expressed in Cabana. On GPU, performance has been observed as "good", and
performs as expected. On CPU, Cabana is underperforming relative to both Kokkos
and a native implementation. We will use these benchmarks as an attempt to
understand why. 

We will implement the particle push and interpolation, by hand for the following:

- Vanilla C++ (data and execution)
- Kokkos data with C++ execution
- Kokkos data with Kokkos execution
- Cabana data with Kokkos execution
- Cabana data with Cabana execution
- Cajita


This will involve implementing the following core elements in each dir:

1. Particle Data Structure
2. Interpolator Data Structure
3. G2P kernel
4. Data initialization

The kernel is based on mini-pic, but only includes the vectorizable portion as a
fissisoned kernel
