using Octavian, LoopVectorization, Zygote

struct Dense{F, M<:AbstractMatrix, B}
    weight::M
    bias::B
    σ::F
    function Dense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
        b = Flux.create_bias(W, bias, size(W,1))
        new{F,M,typeof(b)}(W, b, σ)
    end
end

function Dense(in::Integer, out::Integer, σ = identity;
                initW = nothing, initb = nothing,
                init = Flux.glorot_uniform, bias=true)

    W = if initW !== nothing
        Base.depwarn("keyword initW is deprecated, please use init (which similarly accepts a funtion like randn)", :Dense)
        initW(out, in)
    else
        init(out, in)
    end

    b = if bias === true && initb !== nothing
        Base.depwarn("keyword initb is deprecated, please simply supply the bias vector, bias=initb(out)", :Dense)
        initb(out)
    else
        bias
    end

    return Dense(W, b, σ)
end

TurboDense(args...) = Dense(args...)

Flux.@functor Dense

function (a::Dense)(x::AbstractVecOrMat)
    W, b, σ = a.weight, a.bias, a.σ
    A = weightmul(W, x)
    B = biasadd(A, b)
    nonlin(B, σ)
end

weightmul(w, x) = w * x
weightmul(w::VecOrMat, x::VecOrMat) = matmul(w, x)

function ChainRulesCore.rrule(::typeof(matmul), A::Matrix, B::VecOrMat)
    C = matmul(A, B)
    function matmul_pullback(tΔ)
        Δ = unthunk(tΔ)
        Δ = collect(Δ) #matmul does not support FillArrays
        matrixΔ = reshape(Δ, size(Δ, 1), size(Δ, 2))
        dA = InplaceableThunk(bA -> matmul!(bA, matrixΔ, B'), @thunk(matmul(matrixΔ, B')))
        dB = InplaceableThunk(bB -> matmul!(bB, A', Δ), @thunk(matmul(A', Δ)))
        return (NoTangent(), dA, dB)
    end
    C, matmul_pullback
end

function biasadd(A, b)
    @turbo A .+ b
end

function ChainRulesCore.rrule(::typeof(biasadd), A, b)
    o = biasadd(A, b)
    function biasadd_pullback(tΔ)
        Δ = unthunk(tΔ)
        dA = @thunk(Δ)
        db = @thunk begin
            tmp = zero(b)
            @turbo for c in 1:size(Δ, 2)
                for r in 1:size(Δ, 1)
                    tmp[r] += Δ[r, c]
                end
            end
            tmp
        end
        return (NoTangent(), dA, db)
    end
    o, biasadd_pullback
end

v_σs = [tanh]
Vectorized_σs = Union{map(typeof, v_σs)...}

function nonlin(A, σ::Vectorized_σs)
    @turbo σ.(A)
end

function nonlin(A, σ)
    σ.(A)
end

tanh_pullback(Δ, _, o) = @turbo Δ .* (1 .- o .^2)

for σ in map(Symbol, v_σs)
    nonlin_σ_pullback = Symbol("nonlin_", σ, "_pullback")
    σ_pullback = Symbol(σ, "_pullback")
    @eval begin
        function ChainRulesCore.rrule(::typeof(nonlin), A, _::typeof($σ))
            o = nonlin(A, $σ)
            $nonlin_σ_pullback(Δ) = (NoTangent(), $σ_pullback(Δ, A, o), NoTangent())
            o, $nonlin_σ_pullback
        end
    end
end

(a::Dense)(x::AbstractArray) =
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == Flux.Zeros() && print(io, "; bias=false")
    print(io, ")")
end
