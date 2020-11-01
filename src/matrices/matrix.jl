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
include("row_imputing_matrix.jl")
include("col_imputing_matrix.jl")

const ImputingMatrix{T, C, U} = Union{RowImputingMatrix{T, C, U}, ColImputingMatrix{T, C, U}}

_print_params(io::IO, A::RowImputingMatrix) = print_array(io, A.ψ')
_print_params(io::IO, A::ColImputingMatrix) = print_array(io, A.ψ)

function print_array(io::IO, A::ImputingMatrix)
    println(io, "W:")
    print_array(io, A.W)
    println(io, "\n\nψ:")
    _print_params(io, A)
end

function Flux.params!(p::Params, A::ImputingMatrix, seen=IdSet())
    A in seen && return
    push!(seen, A)
    push!(p, A.W, A.ψ)
end

RowImputingDense(d::Dense) = Dense(RowImputingMatrix(d.W), d.b, d.σ)
RowImputingDense(args...) = RowImputingDense(Dense(args...))
ColImputingDense(d::Dense) = Dense(ColImputingMatrix(d.W), d.b, d.σ)
ColImputingDense(args...) = ColImputingDense(Dense(args...))

_name(::RowImputingMatrix) = "RowImputing"
_name(::ColImputingMatrix) = "ColImputing"
function Base.show(io::IO, l::Dense{F, <:ImputingMatrix}) where F
  print(io, "$(_name(l.W))Dense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
