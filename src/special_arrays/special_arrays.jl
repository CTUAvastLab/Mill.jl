# TODO replace all @adjoints in matrices by rrules once Tangent gradients become available
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
include("preimputing_matrix.jl")
include("postimputing_matrix.jl")

const MaybeHotArray{T} = Union{MaybeHotVector{T}, MaybeHotMatrix{T}}

const ImputingMatrix{T, U} = Union{PreImputingMatrix{T, U}, PostImputingMatrix{T, U}}
const PreImputingDense = Dense{T, <: PreImputingMatrix} where T
const PostImputingDense = Dense{T, <: PostImputingMatrix} where T

# so we error on integers with missings
Flux.onecold(y::MaybeHotArray{Maybe{T}}, labels = 1:size(y, 1)) where T<:Integer = 
    throw(ArgumentError("MaybeHotArray{Union{T, Missing}} where T <:Integer can't produce onecold encoding, use maybecold instead."))
# but we don't error on maybehot which has only integers

y = maybehotbatch([1, missing, 3], 1:10)
function maybecold(y::AbstractArray, labels = 1:size(y, 1))
    indices = Flux._fast_argmax(y)
    xs = isbits(labels) ? indices : collect(indices) # non-bit type cannot be handled by CUDA
    return map(xi -> ismissing(xi) ? xi : labels[xi[1]], xs)
end

Base.zero(X::T) where T <: ImputingMatrix = T(zero(X.W), zero(X.ψ))
Base.similar(X::T) where T <: ImputingMatrix = T(similar(X.W), similar(X.ψ))

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

function Flux.update!(opt, x::ImputingMatrix, x̄)
    if !isnothing(x̄.W)
        Flux.Optimise.update!(opt, x.W, x̄.W)
    end
    if !isnothing(x̄.ψ)
        Flux.Optimise.update!(opt, x.ψ, x̄.ψ)
    end
end

"""
    preimputing_dense(in, out, σ)

Like `Flux.Dense`, but use a [`PreImputingMatrix`](@ref) instead of a standard matrix.

# Examples
```jldoctest
julia> d = preimputing_dense(2, 3)
[preimputing]Dense(2, 3)  # 9 parameters

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
julia> d = postimputing_dense(2, 3)
[postimputing]Dense(2, 3)  # 9 parameters

julia> typeof(d.weight)
PostImputingMatrix{Float32, Matrix{Float32}, Vector{Float32}}

julia> typeof(d.bias)
Vector{Float32} (alias for Array{Float32, 1})
```

See also: [`PostImputingMatrix`](@ref), [`preimputing_dense`](@ref), [`PreImputingMatrix`](@ref).
"""
postimputing_dense(d::Dense) = Dense(PostImputingMatrix(d.weight), d.bias, d.σ)
postimputing_dense(args...) = postimputing_dense(Dense(args...))
