using HierarchicalUtils: encode, stringify

"""
    AbstractMillModel

Supertype for any model defined in `Mill.jl`.
"""
abstract type AbstractMillModel end

const MillFunction = Union{Dense, Chain, Function}

_make_array_model(m::Union{MillFunction, Function}) = ArrayModel(m)
_make_array_model(m) = m

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("lazymodel.jl")
include("reflectinmodel.jl")
