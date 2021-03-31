using DataStructures: SortedDict, OrderedDict

_empty_range(t::Type{<:Signed}) = zero(t):-one(t)
_empty_range(t::Type{<:Unsigned}) = one(t):zero(t)

"""
    AbstractBags{T}

Supertype for structures storing indices of type `T` of bags' instances in [`BagNode`](@ref)s.
"""
abstract type AbstractBags{T} end

"""
    AlignedBags{T <: Integer} <: AbstractBags{T}

[`AlignedBags`](@ref) struct stores indices of bags' instances in one or more `UnitRange{T}`s.
This is only possible if instances of every bag are stored in one contiguous block.

See also: [`ScatteredBags`](@ref).
"""
struct AlignedBags{T <: Integer} <: AbstractBags{T}
    bags::Vector{UnitRange{T}}
end

Flux.@forward AlignedBags.bags Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex,
    Base.eachindex, Base.first, Base.last, Base.iterate, Base.eltype, Base.length

"""
    AlignedBags()

Construct a new [`AlignedBags`](@ref) struct containing no bags.

# Examples
```jldoctest
julia> AlignedBags()
AlignedBags{Int64}(UnitRange{Int64}[])
```
"""
AlignedBags() = AlignedBags(UnitRange{Int}[])

"""
    AlignedBags(bags::UnitRange{<:Integer}...)

Construct a new [`AlignedBags`](@ref) struct from bags in arguments.

# Examples
```jldoctest
julia> AlignedBags(1:3, 4:8)
AlignedBags{Int64}(UnitRange{Int64}[1:3, 4:8])
```
"""
AlignedBags(bags::UnitRange{<:Integer}...) = AlignedBags(collect(bags))

"""
    AlignedBags(k::Vector{<:Integer})

Construct a new [`AlignedBags`](@ref) struct from `Vector` `k` specifying the index of the bag each instance belongs to.
Throws `ArgumentError` if this is not possible.

# Examples
```jldoctest
julia> AlignedBags([2, 2, 1, 1, 1, 3])
AlignedBags{Int64}(UnitRange{Int64}[1:2, 3:5, 6:6])
```
"""
function AlignedBags(k::Vector{T}) where T <: Integer
    b = AlignedBags()
    !isempty(k) || return b
    a, v = 1, k[1]
    s = Set{T}(v)
    for (i, x) in enumerate(k[2:end])
        if x != v
            !(x in s) || ArgumentError("Scattered bags") |> throw
            push!(b.bags, a:i)
            v = x
            a = i+1
            push!(s, x)
        end
    end
    push!(b.bags, a:length(k))
    b
end

"""
    length2bags(ls::Vector{<:Integer})

Convert lengths of bags given in `ls` to [`AlignedBags`](@ref) with contiguous blocks.

# Examples
```jldoctest
julia> length2bags([1, 3, 2])
AlignedBags{Int64}(UnitRange{Int64}[1:1, 2:4, 5:6])
```

See also: [`AlignedBags`](@ref).
"""
function length2bags(ls::Vector{T}) where T <: Integer
    ls = vcat([zero(T)], cumsum(ls))
    bags = map(i -> i[1] + 1:i[2], zip(ls[1:end-1], ls[2:end]))
    bags = map(b -> isempty(b) ? _empty_range(T) : b, bags)
    AlignedBags(bags)
end

Zygote.@nograd length2bags

"""
    ScatteredBags{T <: Integer} <: AbstractBags{T}

[`ScatteredBags`](@ref) struct stores indices of bags' instances that are not necessarily contiguous.

See also: [`AlignedBags`](@ref).
"""
struct ScatteredBags{T <: Integer} <: AbstractBags{T}
    bags::Vector{Vector{T}}
end

Flux.@forward ScatteredBags.bags Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex,
    Base.eachindex, Base.first, Base.last, Base.iterate, Base.eltype, Base.length

"""
    ScatteredBags()

Construct a new [`ScatteredBags`](@ref) struct containing no bags.

# Examples
```jldoctest
julia> ScatteredBags()
ScatteredBags{Int64}(Vector{Int64}[])
```
"""
ScatteredBags() = ScatteredBags(Vector{Vector{Int}}())

"""
    ScatteredBags(k::Vector{<:Integer})

Construct a new [`ScatteredBags`](@ref) struct from `Vector` `k` specifying the index of the bag each instance belongs to.

# Examples
```jldoctest
julia> ScatteredBags([2, 2, 1, 1, 1, 3])
ScatteredBags{Int64}([[3, 4, 5], [1, 2], [6]])
```
"""
function ScatteredBags(k::Vector{T}) where T <: Integer
    !isempty(k) || return ScatteredBags()
    d = SortedDict{T, Vector{T}}()
    for (i, x) in enumerate(k)
        if !(x in keys(d))
            d[x] = Int[]
        end
        push!(d[x], i)
    end
    ScatteredBags(collect(values(d)))
end

"""
    bags(k::Vector{<:Integer})
    bags(k::Vector{T}) where T <: UnitRange{<:Integer}
    bags(b::AbstractBags)

Construct an [`AbstractBags`](@ref) structure that is most suitable for the input
([`AlignedBags`](@ref) if possible, [`ScatteredBags`](@ref) otherwise).

# Examples
```jldoctest
julia> bags([2, 2, 3, 1])
AlignedBags{Int64}(UnitRange{Int64}[1:2, 3:3, 4:4])

julia> bags([2, 3, 1, 2])
ScatteredBags{Int64}([[3], [1, 4], [2]])

julia> bags([1:3, 4:5])
AlignedBags{Int64}(UnitRange{Int64}[1:3, 4:5])

julia> bags(ScatteredBags())
ScatteredBags{Int64}(Vector{Int64}[])
```

See also: [`AlignedBags`](@ref), [`ScatteredBags`](@ref).
"""
bags(b::AbstractBags) = b
bags(b::Vector{T}) where T <: UnitRange{<:Integer} = AlignedBags(b)
function bags(k::Vector{<:Integer})
    try
        return AlignedBags(k)
    catch ErrorException
        return ScatteredBags(k)
    end
end

"""
    remapbags(b::AbstractBags, idcs::VecOrRange{<:Integer}) -> (rb, I)

Select a subset of bags in `b` corresponding to indices `idcs` and remap instance indices appropriately.
Return new bags `rb` as well as a `Vector` of remapped instances `I`.

# Examples
```jldoctest
julia> remapbags(AlignedBags([1:1, 2:3, 4:5]), [1, 3])
(AlignedBags{Int64}(UnitRange{Int64}[1:1, 2:3]), [1, 4, 5])

julia> remapbags(ScatteredBags([[1,3], [2], Int[]]), [2])
(ScatteredBags{Int64}([[1]]), [2])
```
"""
function remapbags(b::AlignedBags{T}, idcs::VecOrRange{<:Integer}) where T
    rb = AlignedBags(Vector{UnitRange{T}}(undef, length(idcs)))
    offset = one(T)
    for (i, j) in enumerate(idcs)
        rb[i] = (b[j] == _empty_range(T)) ? b[j] : b[j] .- b[j].start .+ offset
        offset += length(b[j])
    end
    rb, Array{T}(reduce(vcat, [collect(b[i]) for i in idcs]; init=AlignedBags[]))
end

function remapbags(b::ScatteredBags{T}, idcs::VecOrRange{<:Int}) where T
    rb = ScatteredBags(Vector{Vector{T}}(undef, length(idcs)))
    m = OrderedDict{T, Int}((v => i for (i, v) in enumerate(unique(vcat(b.bags[idcs]...)))))
    for (i, j) in enumerate(idcs)
        rb[i] = [m[v] for v in b[j]]
    end
    rb, collect(keys(m))
end

"""
    adjustbags(b::AlignedBags, mask::AbstractVector{Bool})

Remove indices of instances brom bags `b` and remap the remaining instances accordingly.

# Examples
```jldoctest
julia> adjustbags(AlignedBags([1:2, 0:-1, 3:4]), [false, false, true, true])
AlignedBags{Int64}(UnitRange{Int64}[0:-1, 0:-1, 1:2])
```
"""
adjustbags(b::AlignedBags, mask::AbstractVector{Bool}) = length2bags(map(b -> sum(@view mask[b]), b))

Base.vcat(bs::AbstractBags{T}...) where T = _catbags(collect(bs))

function _catbags(bs::Vector{AlignedBags{T}}) where T <: Integer
    nbs = AlignedBags(UnitRange{T}[])
    offset = zero(T)
    for b in bs
        !isempty(b) || continue
        append!(nbs.bags, [bb .+ (isempty(bb) ? zero(T) : offset) for bb in b])
        offset += max(zero(T), mapreduce(i -> isempty(i) ? zero(T) : maximum(i), max, b))
    end
    mask = length.(nbs.bags) .== 0
    if sum(mask) > 0
        nbs[mask] = fill(_empty_range(T), sum(mask))
    end
    nbs
end

function _catbags(bs::Vector{ScatteredBags{T}}) where T <: Integer
    nbs = ScatteredBags(Vector{T}[])
    offset = zero(T)
    for b in bs
        !isempty(b) || continue
        append!(nbs.bags, [bb .+ offset for bb in b])
        offset += max(zero(T), mapreduce(i -> isempty(i) ? zero(T) : maximum(i), max, b))
    end
    nbs
end

Base.hash(e::AlignedBags, h::UInt) where {A,C} = hash(e.bags, h)
e1::AlignedBags == e2::AlignedBags = e1.bags == e2.bags
Base.hash(e::ScatteredBags, h::UInt) where {A,C} = hash(e.bags, h)
e1::ScatteredBags == e2::ScatteredBags = e1.bags == e2.bags
