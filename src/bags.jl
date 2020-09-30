using DataStructures: SortedDict, OrderedDict

abstract type AbstractBags end

Base.getindex(b::AbstractBags, i) = b.bags[i]
Base.setindex!(b::AbstractBags, v, i) = setindex!(b.bags, v, i)
Base.iterate(b::AbstractBags) = iterate(b.bags)
Base.iterate(b::AbstractBags, s) = iterate(b.bags, s)
Base.length(b::AbstractBags) = length(b.bags)

# one instance belongs to only one bag
# sorted from left to right
struct AlignedBags <: AbstractBags
    bags::Vector{UnitRange{Int}}
end

AlignedBags() = AlignedBags(Vector{UnitRange{Int}}())
AlignedBags(ks::UnitRange{Int}...) = AlignedBags(collect(ks))
function AlignedBags(k::Vector{T}) where {T<:Integer}
    b = AlignedBags()
    !isempty(k) || return b
    a, v = 1, k[1]
    s = Set{T}(v)
    for (i, x) in enumerate(k[2:end])
        if x != v
            !(x in s) || error("Scattered bags")
            push!(b.bags, a:i)
            v = x
            a = i+1
            push!(s, x)
        end
    end
    push!(b.bags, a:length(k))
    b
end

# one instance may belong to more bags
struct ScatteredBags <: AbstractBags
    bags::Vector{Vector{Int}}
end

ScatteredBags() = ScatteredBags(Vector{Vector{Int}}())
function ScatteredBags(k::Vector{T}) where {T<:Integer}
    !isempty(k) || return ScatteredBags()
    d = SortedDict{T, Vector{Int}}()
    for (i, x) in enumerate(k)
        if !(x in keys(d))
            d[x] = Int[]
        end
        push!(d[x], i)
    end
    ScatteredBags(collect(values(d)))
end

Zygote.@nograd function length2bags(ls::Vector{Int})
    ls = vcat([0], cumsum(ls))
    bags = map(i -> i[1]+1:i[2],zip(ls[1:end-1],ls[2:end]))
    bags = map(b -> isempty(b) ? (0:-1) : b,bags)
    AlignedBags(bags)
end

# """
# 		function bags(k::Vector)

# 		creates AlignedBags if it is possible to do so, otherwise creates ScatteredBags instance
# """
bags(b::AbstractBags) = b
bags(b::Vector{UnitRange{Int}}) = AlignedBags(b)
function bags(k::Vector{T}) where {T<:Integer}
    try
        return AlignedBags(k)
    catch ErrorException
        return ScatteredBags(k)
    end
end

"""
    function remapbag(b::Bags,idcs::Vector{Int})

    bags corresponding to indices with collected indices
"""
function remapbag(b::AlignedBags, idcs::VecOrRange)
    rb = AlignedBags(Vector{UnitRange{Int}}(undef, length(idcs)))
    offset = 1
    for (i,j) in enumerate(idcs)
        rb[i] = (b[j] == 0:-1) ? b[j] : b[j] .- b[j].start .+ offset
        offset += length(b[j])
    end
    rb, Array{Int}(vcat([collect(b[i]) for i in idcs]...))
end

function remapbag(b::ScatteredBags, idcs::VecOrRange)
    rb = ScatteredBags(Vector{Vector{Int}}(undef, length(idcs)))
    m = OrderedDict{Int, Int}((v => i for (i, v) in enumerate(unique(vcat(b.bags[idcs]...)))))
    for (i,j) in enumerate(idcs)
        rb[i] = [m[v] for v in b[j]]
    end
    rb, collect(keys(m))
end

adjustbags(bags::AlignedBags, mask::T) where {T<:Union{Vector{Bool}, BitArray{1}}} = length2bags(map(b -> sum(@view mask[b]), bags))

Base.vcat(b1::AbstractBags, b2::AbstractBags) = _catbags([b1, b2])
Base.vcat(bs::AbstractBags...) = _catbags(collect(bs))

function _catbags(bs::Vector{AlignedBags})
    nbs = AlignedBags()
    offset = 0
    for b in bs
        !isempty(b) || continue
        append!(nbs.bags, [bb .+ (isempty(bb) ? 0 : offset) for bb in b])
        offset += max(0, mapreduce(i -> isempty(i) ? 0 : maximum(i), max, b))
    end
    mask = length.(nbs.bags) .== 0
    if sum(mask) > 0
        nbs[mask] = fill(0:-1, sum(mask))
    end
    nbs
end

function _catbags(bs::Vector{ScatteredBags})
    nbs = ScatteredBags()
    offset = 0
    for b in bs
        !isempty(b) || continue
        append!(nbs.bags, [bb .+ offset for bb in b])
        offset += max(0, mapreduce(i -> isempty(i) ? 0 : maximum(i), max, b))
    end
    nbs
end

Base.hash(e::AlignedBags, h::UInt) where {A,C} = hash(e.bags, h)
e1::AlignedBags == e2::AlignedBags = e1.bags == e2.bags
Base.hash(e::ScatteredBags, h::UInt) where {A,C} = hash(e.bags, h)
e1::ScatteredBags == e2::ScatteredBags = e1.bags == e2.bags
