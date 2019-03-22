import Base: show

    # TODO Float32 x Float64
#     # TODO what if the input is just Nothing in case of only empty bags? - reseno v BagNode
#     # TODO write tests na C
#     # TODO write tests for nothing input
#     # TODO metaprogram boilerplate code - BACKWARD and FORWRAD
#     # TODO opravit inbounds vsude
#     # TODO smazit masked verze?
#
abstract type AggregationFunction end

struct Aggregation{N}
    fs::NTuple{N, AggregationFunction}
    Aggregation(fs::Vararg{AggregationFunction, N}) where N = new{N}(fs)
    Aggregation(fs::NTuple{N, AggregationFunction}) where N = new{N}(fs)
end

Flux.@treelike Aggregation

(a::Aggregation)(args...) = vcat([f(args...) for f in a.fs]...)

Base.show(io::IO, a::Aggregation) = modelprint(io, a)

function modelprint(io::IO, a::Aggregation{N}; pad=[]) where N
    paddedprint(io, N == 1 ? "" : "⟨")
    for f in a.fs[1:end-1]
        modelprint(io, f, pad)
        paddedprint(io, ", ")
    end
    modelprint(io, a.fs[end], pad)
    paddedprint(io, (N == 1 ? "" : "⟩") * '\n')
end

const Tracked = Union{TrackedMatrix, TrackedVector}
const MaybeMatrix = Union{Matrix, Nothing}
const InputMatrix = Union{Matrix, TrackedMatrix}
const MaybeInputMatrix = Union{InputMatrix, Nothing}
const MaybeVector = Union{Vector, Nothing}
const InputVector = Union{Vector, TrackedVector}
const MaybeInputVector = Union{InputVector, Nothing}
const MaybeMask = Union{Vector{Bool}, Nothing}

macro do_nothing()
    quote quote end end
end

macro mask_rule(mask_type) 
    quote
        $(esc(mask_type)) <: Nothing ? $(@do_nothing) : :(!mask[bi] && continue)
    end
end

macro unpack(t, s)
    return quote
        $(esc(t)) <: Tracked ? :($$s = Flux.data($$s)) : $(@do_nothing)
    end
end

macro define_derivative(t, s)
    return quote
        $(esc(t)) <: Tracked ? :($(Symbol("d" * string($s))) = zeros($$s)) : $(@do_nothing)
    end
end

complete_body(init_rule, empty_bag_update_rule, init_bag_rule, mask_rule,
              bag_update_rule, after_bag_rule, return_rule) =
begin
    q= 
quote
    $init_rule
    for (j, b) in enumerate(bags)
        if isempty(b)
            for i in 1:size(x, 1)
                @inbounds $empty_bag_update_rule
            end
        else
            @inbounds $init_bag_rule
            for bi in b
                @inbounds $mask_rule
                for i in 1:size(x, 1)
                    @inbounds $bag_update_rule
                end
            end
            @inbounds $after_bag_rule
        end
    end
    $return_rule
end

println(q)
return q
end

include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")

export SegmentedMax, SegmentedMean, SegmentedPNorm, SegmentedLSE

const names = ["Mean", "Max", "PNorm", "LSE"]
for idxs in powerset(collect(1:length(names)))
    length(idxs) > 1 || continue
    for p in permutations(idxs)
        @eval function $(Symbol("Segmented", names[p]...))(d::Int)
            Aggregation($((:($(Symbol("Segmented" * n))(d)) for n in names[p])...))
        end
        @eval function $(Symbol("Segmented", names[p]...))(c::Vararg{Matrix{Float32}, $(length(p))})
            Aggregation($((:($(Symbol("Segmented" * n))(C[$i])) for (i,n) in enumerate(names[p]))...))
        end
        @eval export $(Symbol("Segmented", names[p]...))
    end
end

