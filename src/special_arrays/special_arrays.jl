function _check_mul(A::AbstractMatrix, b::AbstractVector)
    if size(A, 2) ≠ length(b)
        DimensionMismatch(
            "Number of columns of A ($(size(A, 2))) must correspond with length of b ($(length(b)))"
        ) |> throw
    end
end

function _check_mul(A::AbstractMatrix, B::AbstractMatrix)
    if size(A, 2) ≠ size(B, 1)
        DimensionMismatch(
              "Number of columns of A ($(size(A, 2))) must correspond with number of rows of B ($(size(B, 1)))"
        ) |> throw
    end
end

include("maybe_hot_vector.jl")
include("maybe_hot_matrix.jl")

const MaybeHotArray{T} = Union{MaybeHotVector{T}, MaybeHotMatrix{T}}

# so we error on integers with missings (but not on maybehot which has only integers)
Flux.onecold(X::MaybeHotArray{Maybe{T}}, labels = 1:size(X, 1)) where T <: Integer = 
    throw(ArgumentError("$(typeof(X)) can't produce `onecold` encoding, use `maybecold` instead."))

"""
    maybecold(y, labels=1:size(y,1))

Similar to `Flux.onecold` but when `y` contains `missing` values, `missing` is in the result as well.

Therefore, it is roughly the inverse operation of [`maybehot`](@ref) or [`maybehotbatch`](@ref).

# Examples
```jldoctest
julia> maybehot(:b, [:a, :b, :c])
3-element MaybeHotVector with eltype Bool:
 ⋅
 1
 ⋅

julia> maybecold(ans, [:a, :b, :c])
:b

julia> maybehot(missing, 1:3)
3-element MaybeHotVector with eltype Missing:
 missing
 missing
 missing

julia> maybecold(ans)
missing

julia> maybecold(maybehotbatch([missing, 2], 1:3))
2-element Vector{Union{Missing, Int64}}:
  missing
 2
```

See also: `Flux.onecold`, [`maybehot`](@ref), [`maybehotbatch`](@ref).
"""
function maybecold end

include("ngram_matrix.jl")

ChainRulesCore.ProjectTo(X::NGramMatrix) = ProjectTo{typeof(X)}()
# Allow NGramMatrix to reach specialisation of * etc:
Flux._match_eltype(_, ::Type, x::NGramMatrix) = x

#=
Note on defining `{Post|Pre}ImputingMatrix as a subtype of `AbstractMatrix`. Advantages:
- conceptually "is" a matrix,
- can be reused inside `Flux.Dense`, which expects `AbstractMatrix`,
- can reuse `print_array` from `Base` for pretty printing.

On the other hand:
- Zygote.jl and also ChainRules.jl will define rules for AD, which we have to opt-out or correct them
- `Flux.trainables` will return the matrix itself, not its parameters as its recursion stops on any
    `AbstractMatrix{<:Number}`.

Relevant issues:
https://github.com/FluxML/Flux.jl/issues/2559
https://github.com/FluxML/Flux.jl/issues/2045
https://github.com/FluxML/Zygote.jl/issues/1146

If this continues to be a problem, we can stop definining `{Post|Pre}ImputingMatrix` as a subtype of
`AbstractMatrix` and take inspiration e.g. from `SVD` and other `Factorization` subtypes.
=#
include("preimputing_matrix.jl")
include("postimputing_matrix.jl")

const ImputingMatrix{T, U} = Union{PreImputingMatrix{T, U}, PostImputingMatrix{T, U}}
const PreImputingDense = Dense{T, <: PreImputingMatrix} where T
const PostImputingDense = Dense{T, <: PostImputingMatrix} where T

Base.zero(X::T) where T <: ImputingMatrix = T(zero(X.W), zero(X.ψ))
Base.similar(X::T) where T <: ImputingMatrix = T(similar(X.W), similar(X.ψ))

ChainRulesCore.ProjectTo(X::ImputingMatrix) = ProjectTo{typeof(X)}(W = ChainRulesCore.ProjectTo(X.W),
                                                                   ψ = ChainRulesCore.ProjectTo(X.ψ))

function _split_bc(bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}) where T <: ImputingMatrix
    bc1, bc2 = _split_bc(bc.args)
    Base.broadcasted(bc.f, bc1...), Base.broadcasted(bc.f, bc2...)
end
_split_bc(A::ImputingMatrix) = (A.W, A.ψ)
_split_bc(x::Tuple) = tuple(zip(_split_bc.(x)...)...)
_split_bc(x) = (x, x)

Base.BroadcastStyle(::Type{T}) where T <: ImputingMatrix = Broadcast.ArrayStyle{T}()

function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}) where T <: ImputingMatrix
    T(map(Base.copy, _split_bc(bc))...)
end
function Base.copyto!(A::T, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}) where T <: ImputingMatrix
    bc1, bc2 = _split_bc(bc)
    Base.copyto!(A.W, bc1)
    Base.copyto!(A.ψ, bc2)
    return A
end

"""
    preimputing_dense(in, out, σ)

Like `Flux.Dense`, but use a [`PreImputingMatrix`](@ref) instead of a standard matrix.

# Examples
```jldoctest
julia> d = preimputing_dense(2, 3)
[preimputing]Dense(2 => 3)  3 arrays, 11 params, 172 bytes

julia> typeof(d.weight)
PreImputingMatrix{Float32, Matrix{Float32}, Vector{Float32}}

julia> typeof(d.bias)
Vector{Float32} (alias for Array{Float32, 1})
```

See also: [`PreImputingMatrix`](@ref), [`postimputing_dense`](@ref), [`PostImputingMatrix`](@ref).
"""
preimputing_dense(d::Dense) = Dense(PreImputingMatrix(d.weight), d.bias, d.σ)
preimputing_dense(args...) = preimputing_dense(Dense(args...))

"""
    postimputing_dense(d_in, d_out, σ)

Like `Flux.Dense`, but use a [`PostImputingMatrix`](@ref) instead of a standard matrix.

# Examples
```jldoctest
julia> d = postimputing_dense(3, 2)
[postimputing]Dense(3 => 2)  3 arrays, 10 params, 168 bytes

julia> typeof(d.weight)
PostImputingMatrix{Float32, Matrix{Float32}, Vector{Float32}}

julia> typeof(d.bias)
Vector{Float32} (alias for Array{Float32, 1})
```

See also: [`PostImputingMatrix`](@ref), [`preimputing_dense`](@ref), [`PreImputingMatrix`](@ref).
"""
postimputing_dense(d::Dense) = Dense(PostImputingMatrix(d.weight), d.bias, d.σ)
postimputing_dense(args...) = postimputing_dense(Dense(args...))
