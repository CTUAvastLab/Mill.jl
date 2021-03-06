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
include("pre_imputing_matrix.jl")
include("post_imputing_matrix.jl")

const MaybeHotArray{T} = Union{MaybeHotVector{T}, MaybeHotMatrix{T}}

const ImputingMatrix{T, U} = Union{PreImputingMatrix{T, U}, PostImputingMatrix{T, U}}
const PreImputingDense = Dense{T, <: PreImputingMatrix} where T
const PostImputingDense = Dense{T, <: PostImputingMatrix} where T

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

_print_params(io::IO, A::PreImputingMatrix) = Base.print_array(io, A.ψ |> permutedims)
_print_params(io::IO, A::PostImputingMatrix) = Base.print_array(io, A.ψ)

function Base.print_array(io::IO, A::ImputingMatrix)
    println(io, "W:")
    Base.print_array(io, A.W)
    println(io, "\n\nψ:")
    _print_params(io, A)
end

Base.print_array(io::IO, A::NGramMatrix) = Base.print_array(io, A.S)

function Flux.update!(opt, x::ImputingMatrix, x̄)
    if !isnothing(x̄.W)
        x.W .-= Flux.Optimise.apply!(opt, x.W, x̄.W)
    end
    if !isnothing(x̄.ψ)
        x.ψ .-= Flux.Optimise.apply!(opt, x.ψ, x̄.ψ)
    end
end
function Flux.update!(x::ImputingMatrix, x̄)
    if !isnothing(x̄.W)
        x.W .-= x̄.W
    end
    if !isnothing(x̄.ψ)
        x.ψ .-= x̄.ψ
    end
end

function Base.show(io::IO, X::T) where T <: Union{ImputingMatrix, MaybeHotArray, NGramMatrix}
    if get(io, :compact, false)
        if ndims(X) == 1
            print(io, length(X), "-element ", nameof(T))
        else
            print(io, join(size(X), "×"), " ", nameof(T))
        end
    else
        _show_fields(io, X)
    end
end

"""
    preimputing_dense(in, out, σ)

Like `Flux.Dense`, but use a [`PreImputingMatrix`](@ref) instead of a standard matrix.

# Examples
```jldoctest
julia> d = preimputing_dense(2, 3)
[pre_imputing]Dense(2, 3)

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
[post_imputing]Dense(2, 3)

julia> typeof(d.weight)
PostImputingMatrix{Float32, Matrix{Float32}, Vector{Float32}}

julia> typeof(d.bias)
Vector{Float32} (alias for Array{Float32, 1})
```

See also: [`PostImputingMatrix`](@ref), [`preimputing_dense`](@ref), [`PreImputingMatrix`](@ref).
"""
postimputing_dense(d::Dense) = Dense(PostImputingMatrix(d.weight), d.bias, d.σ)
postimputing_dense(args...) = postimputing_dense(Dense(args...))

_name(::PreImputingMatrix) = "[pre_imputing]"
_name(::PostImputingMatrix) = "[post_imputing]"
function Base.show(io::IO, l::Dense{F, <:ImputingMatrix}) where F
    print(io, "$(_name(l.weight))Dense(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end
