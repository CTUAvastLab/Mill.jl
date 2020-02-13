using CuArrays

# CUDA counterparts of basic Mill data structures
# (note that some type-parametrized data structures can be directly reused)

"""
    CUDA counterpart of AlignedBags

    BEWARE! There is no field :bags as in AlignedBags. Instead, two arrays of
    start indices and end indices are used.
"""
struct CuAlignedBags <: AbstractBags
    bs::CuArray{Int32}
    be::CuArray{Int32}
end
CuAlignedBags(x::AlignedBags) = CuAlignedBags(map(b->b.start, x.bags), map(b->b.stop, x.bags))
Base.length(x::CuAlignedBags) = length(x.bs)
