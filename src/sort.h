#ifndef CabanaPIC_sort
#define CabanaPIC_sort

// Quick and dirty implementation of the spare aosoa sort
// It would be smart to base this heavily of the existing Kokkos sort, but
// there are some performance concerns

// TODO: add min/max finding
template<class Space>
class Sparse_sorter
{
    public:

    //using Space = Kokkos::DefaultExecutionSpace;

    int min;
    int num_bins;

    using bin_count_t = Kokkos::View< int*, Space>;
    using bin_offset_t = Kokkos::View< int*, Space>;

    bin_count_t bin_count;
    bin_offset_t bin_offset;

    Sparse_sorter(int _num_bins, int _min) : num_bins(_num_bins), min(_min)
    {
             bin_count = bin_count_t("bin_count", num_bins);
             bin_offset = bin_offset_t("bin_offset", num_bins);
    }

    template<class slice_t, class mask_t>
    void find_bin_counts(slice_t& keys, mask_t& masks)
    {
        int num_data = keys.size();

        Kokkos::deep_copy( bin_count, 0 );

        // TODO: should this be 1d or 2d?
        auto bin_count_kernel = KOKKOS_LAMBDA(const int i)
        {
            // Only bin if mask is set
            if (masks(i) == 0) { return; }
            else {
                int index = keys(i);
                Kokkos::atomic_increment(&bin_count(index));
            }
        };

        Kokkos::RangePolicy<Space> range_policy( 0, num_data );
        Kokkos::parallel_for("bin count", range_policy, bin_count_kernel);
    }

    template<class slice_t>
    void find_bin_offsets(slice_t& keys)
    {
        auto vec_len = slice_t::vector_length;

        int num_data = bin_count.size();

        auto bin_offset_kernel = KOKKOS_LAMBDA(const int& i, int& offset,
                  const bool& final_pass)
        {
            if ( final_pass )
            {
                bin_offset(i) = offset;
            }

            // If we are >0, round to nearest multiple of vec_len
            // If we are ==0, keep 0
            // Avoid rounding extra multiples of vlen (eg rounding 64 to 92)

            // TODO: there's probably a fancy way to do this, but this isn't so bad for now
            // // TODO: theere's probably a fancy way to do this, but this isn't so  bad for now
            int this_offset = bin_count( i );
            int mod = this_offset % vec_len;
            if (mod != 0)
            {
               this_offset = this_offset + (vec_len - mod);
            }
            //printf("Rounding %d to %d \n", bin_count(i), this_offset);

            offset += this_offset;
        };

        Kokkos::RangePolicy<Space> range_policy( 0, num_data );
        Kokkos::parallel_scan( "offset_scan",
            range_policy, bin_offset_kernel);
    }

    // TODO: we could calculate the permute vector instead, which would remove
    // the need to have keys and particles
    template<class slice_t, class AoSoA_t> void perform_sort(slice_t& keys, AoSoA_t& particles)
    {
        int num_data = keys.size();

        // Reset counts so we can reuse it
        Kokkos::deep_copy( bin_count, 0 );

        // TODO: find out if we have a propper copy semantic. What I want is a
        // shallow copy and to then overwrite the underlying data pointer

        // Remember, we have the potential to grow the list
        // Last offset + last count
        int new_size = bin_offset[ bin_offset.size() - 1] + bin_count[ bin_count.size() -1 ] + 1; // TODO: remove this debug +1
        std::cout << "Growing from " << particles.size() << " to " << new_size << std::endl;
        AoSoA_t scratch( particles.label(), new_size );

        // TODO: this is getting needlessly initialized if Cabana sets things
        // to  0 for us.
        auto scratch_mask = Cabana::slice<Mask>(scratch);
        auto zero_scratch = KOKKOS_LAMBDA(const int i)
        {
            scratch_mask(i) = 0;
        };
        Kokkos::RangePolicy<Space> scratch_range_policy( 0, scratch.size() );
        Kokkos::parallel_for("zero scratch", scratch_range_policy, zero_scratch);

        auto masks = Cabana::slice<Mask>(particles);

        auto sort_kernel = KOKKOS_LAMBDA(const int i)
        {
            // Populate scratch from particles according to bins/offsets

            // Only sort if mask is set
            if (masks(i) == 0) { return; }
            else {
                int index = keys(i);
                int count = Kokkos::atomic_fetch_add(&bin_count(index), 1);
                int bin = bin_offset( index );
                int dest = bin+count;

                scratch.setTuple( dest, particles.getTuple(i) );
            }
        };

        Kokkos::RangePolicy<Space> range_policy( 0, num_data );
        Kokkos::parallel_for("perform sort", range_policy, sort_kernel);

        // TODO: can we do a pointer swap?
        particles = scratch;
    }

    template<class slice_t, class AoSoA_t, class mask_t> void bin_sort(
            slice_t& keys, 
            AoSoA_t& particles,
            mask_t& mask
    )
    {
        find_bin_counts(keys, mask);
        find_bin_offsets(keys);
        perform_sort(keys, particles);
    }

    // TODO: delete me
    template<class view_t>
    void print(view_t in)
    {
        for (int i = 0; i < in.size(); i++)
        {
            std::cout << i << " = " << in(i) << std::endl;
        }
    }
};

#endif // CabanaPIC_sort header guard
