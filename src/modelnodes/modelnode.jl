using HierarchicalUtils: encode, stringify

"""
    AbstractMillModel

Supertype for any model defined in `Mill.jl`.
"""
abstract type AbstractMillModel end

include("arraymodel.jl")

_make_mill_model(m) = ArrayModel(m)
_make_mill_model(m::AbstractMillModel) = m

include("bagmodel.jl")
include("productmodel.jl")
include("lazymodel.jl")
include("reflectinmodel.jl")
