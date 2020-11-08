# TODO replace all @adjoints in matrices by rrules once Composite gradients become available
# https://github.com/FluxML/Zygote.jl/issues/603

function _check_mul(A::AbstractMatrix, b::AbstractVector)
    if size(A, 2) != length(b)
        DimensionMismatch(
            "Number of columns of A ($(size(A, 2))) must correspond with length of b ($(length(b)))"
        ) |> throw
    end
end

function _check_mul(A::AbstractMatrix, B::AbstractMatrix)
    if size(A, 2) != size(B, 1)
        DimensionMismatch(
              "Number of columns of A ($(size(A, 2))) must correspond with number of rows of B ($(size(B, 1)))"
        ) |> throw
    end
end

include("maybe_hot_vector.jl")
include("maybe_hot_matrix.jl")
include("ngram_matrix.jl")
include("pre_imputing_matrix.jl")
include("post_imputing_matrix.jl")

const ImputingMatrix{T, C, U} = Union{PreImputingMatrix{T, C, U}, PostImputingMatrix{T, C, U}}
_print_params(io::IO, A::PreImputingMatrix) = print_array(io, A.ψ |> permutedims)
_print_params(io::IO, A::PostImputingMatrix) = print_array(io, A.ψ)

function print_array(io::IO, A::ImputingMatrix)
    println(io, "W:")
    print_array(io, A.W)
    println(io, "\n\nψ:")
    _print_params(io, A)
end

print_array(io::IO, A::NGramMatrix) = print_array(io, A.s)

function Base.show(io::IO, @nospecialize X::T) where T <: Union{ImputingMatrix, MaybeHotMatrix,
                                             MaybeHotVector, NGramMatrix}
    if get(io, :compact, false)
        print(io, join(size(X), "x"), " ", nameof(T))
    else
        _show(io, X)
    end
end

function Flux.params!(p::Params, A::ImputingMatrix, seen=IdSet())
    A in seen && return
    push!(seen, A)
    push!(p, A.W, A.ψ)
end

PreImputingDense(d::Dense) = Dense(PreImputingMatrix(d.W), d.b, d.σ)
PreImputingDense(args...) = PreImputingDense(Dense(args...))
PostImputingDense(d::Dense) = Dense(PostImputingMatrix(d.W), d.b, d.σ)
PostImputingDense(args...) = PostImputingDense(Dense(args...))

_name(::PreImputingMatrix) = "PreImputing"
_name(::PostImputingMatrix) = "PostImputing"
function Base.show(io::IO, @nospecialize l::Dense{F, <:ImputingMatrix}) where F
  print(io, "$(_name(l.W))Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
