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

function Base.show(io::IO, @nospecialize(m::AbstractMillModel))
    print(io, nameof(typeof(m)))
    if !get(io, :compact, false)
        _show_submodels(io, m)
    end
end

_show_submodels(io, m::ArrayModel) = print(io, "(", m.m, ")")
_show_submodels(io, m::BagModel) = print(io, " … ↦ ", m.a, " ↦ ", m.bm)
_show_submodels(io, m::ProductModel) = print(io, " … ↦ ", m.m)
_show_submodels(io, m::LazyModel{Name}) where {Name} = print(io, "{", Name, "}")
_show_submodels(io, _) = print(io)

include("reflectinmodel.jl")
