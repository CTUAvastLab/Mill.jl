const WILDCARD_REPR='␣'

struct MillString{T <: Optional{BitVector}, S <: AbstractString} <: AbstractString
    s::S
    m::T

    MillString(s::S, m::Nothing) where S <: AbstractString = new{Nothing, S}(s, m)
    function MillString(s::S, m::BitVector) where S <: AbstractString
        @assert length(s) == length(m)
        ns = _replace(s, m)
        new{BitVector, S}(ns, m)
    end
end

MillString(s::AbstractString) = MillString(s, nothing)

Flux.@forward MillString.s Base.length, Base.firstindex, Base.lastindex, Base.nextind, Base.reverse,
    Base.ncodeunits, Base.codeunit

macro mill_str(s::String)
    quote
        sc = collect($s)
        m = isequal.(WILDCARD_REPR, sc)
        if any(m)
            MillString(_replace!(sc, m), m)
        else
            MillString($s, nothing)
        end
    end
end

function _replace!(s::AbstractVector{<:Char}, m)
    s[m] .= Char(wildcard_code())
    String(s)
end
_replace(s::AbstractString, m) = _replace!(collect(s), m)

Base.show(io::IO, s::MillString{Nothing}) = print(io, s.s)

function Base.show(io::IO, s::MillString)
    print(io, "\"")
    for (i, si) in eachindex(s.s) |> enumerate
        if s.m[i]
            print(io, WILDCARD_REPR)
        else
            print(io, s.s[si])
        end
    end
    print(io, "\"")
end

Base.codeunit(s::MillString, i::Integer) = codeunit(s.s, i)
