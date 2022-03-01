using HierarchicalUtils: encode, stringify

"""
    AbstractMillModel

Supertype for any model defined in `Mill.jl`.
"""
abstract type AbstractMillModel end

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("lazymodel.jl")
include("reflectinmodel.jl")
