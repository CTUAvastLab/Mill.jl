module MillCUDAExt

using CUDA
using Flux
using Mill
using ChainRulesCore
using Adapt

# Import functions we need to extend
import Mill: segmented_sum_forw, segmented_mean_forw, segmented_max_forw
import Mill: segmented_pnorm_forw, segmented_lse_forw
import Mill: _bagnorm, _weight, _weightsum, _typemin
import Mill: AbstractBags, AlignedBags, ScatteredBags, maxindex, numobs

# Type aliases for GPU bag representations
const CuAlignedBags{T} = AlignedBags{T, <:CuArray{UnitRange{T}}}

# Include sub-modules
include("compressed_bags.jl")
include("kernels.jl")
include("segmented_sum.jl")
include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")
include("datanodes.jl")

# Export for testing
export CompressedBags, CuCompressedBags, CuAlignedBags, CuMaybeHotMatrix

end # module
