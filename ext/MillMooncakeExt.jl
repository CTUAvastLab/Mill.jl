module MillMooncakeExt

using Mooncake
using Mill

import Mooncake:
    DefaultCtx, @is_primitive, @zero_derivative, rrule!!,
    CoDual, NoRData, primal, tangent, zero_fcodual

import Mill:
    AbstractBags,
    PreImputingMatrix, PostImputingMatrix,
    MaybeHotMatrix, NGramMatrix,
    segmented_sum_forw, segmented_mean_forw, segmented_max_forw,
    segmented_pnorm_forw, segmented_lse_forw

# ---- non-differentiable -------------------------------------------------------

@zero_derivative DefaultCtx Tuple{typeof(Mill._bagcount), Any, Any}

# ---- helpers ------------------------------------------------------------------

# Accumulate a gradient into an array-valued field of a struct's fdata.
# `get_tangent_field` unwraps PossiblyUninitTangent if present.
@inline _inc!(t, name::Symbol, Δ) = get_tangent_field(t, name) .+= Δ

# ---- segmented_sum_forw -------------------------------------------------------

@is_primitive DefaultCtx Tuple{typeof(segmented_sum_forw), AbstractMatrix, AbstractVector, AbstractBags, Nothing}
@is_primitive DefaultCtx Tuple{typeof(segmented_sum_forw), AbstractMatrix, AbstractVector, AbstractBags, AbstractVecOrMat}
@is_primitive DefaultCtx Tuple{typeof(segmented_sum_forw), Missing,        AbstractVector, AbstractBags, Nothing}
@is_primitive DefaultCtx Tuple{typeof(segmented_sum_forw), Missing,        AbstractVector, AbstractBags, AbstractVecOrMat}

function rrule!!(::CoDual{typeof(segmented_sum_forw)}, x::CoDual{<:AbstractMatrix}, ψ::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags}, w::CoDual)
    xp, ψp, bagsp, wp = primal(x), primal(ψ), primal(bags), primal(w)
    y = segmented_sum_forw(xp, ψp, bagsp, wp)
    ȳ = zero(y)
    function segmented_sum_pb!!(::NoRData)
        dx, dψ, _, dw = Mill.segmented_sum_back(ȳ, y, xp, ψp, bagsp, wp)
        tangent(x) .+= dx
        tangent(ψ) .+= dψ
        dw isa AbstractArray && (tangent(w) .+= dw)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_sum_pb!!
end

function rrule!!(::CoDual{typeof(segmented_sum_forw)}, x::CoDual{Missing}, ψ::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags}, w::CoDual)
    ψp, bagsp, wp = primal(ψ), primal(bags), primal(w)
    y = segmented_sum_forw(missing, ψp, bagsp, wp)
    ȳ = zero(y)
    function segmented_sum_missing_pb!!(::NoRData)
        _, dψ, _, _ = Mill.segmented_sum_back(ȳ, y, missing, ψp, bagsp, wp)
        tangent(ψ) .+= dψ
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_sum_missing_pb!!
end

# ---- segmented_mean_forw ------------------------------------------------------

@is_primitive DefaultCtx Tuple{typeof(segmented_mean_forw), AbstractMatrix, AbstractVector, AbstractBags, Nothing}
@is_primitive DefaultCtx Tuple{typeof(segmented_mean_forw), AbstractMatrix, AbstractVector, AbstractBags, AbstractVecOrMat}
@is_primitive DefaultCtx Tuple{typeof(segmented_mean_forw), Missing,        AbstractVector, AbstractBags, Nothing}
@is_primitive DefaultCtx Tuple{typeof(segmented_mean_forw), Missing,        AbstractVector, AbstractBags, AbstractVecOrMat}

function rrule!!(::CoDual{typeof(segmented_mean_forw)}, x::CoDual{<:AbstractMatrix}, ψ::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags}, w::CoDual)
    xp, ψp, bagsp, wp = primal(x), primal(ψ), primal(bags), primal(w)
    y = segmented_mean_forw(xp, ψp, bagsp, wp)
    ȳ = zero(y)
    function segmented_mean_pb!!(::NoRData)
        dx, dψ, _, dw = Mill.segmented_mean_back(ȳ, y, xp, ψp, bagsp, wp)
        tangent(x) .+= dx
        tangent(ψ) .+= dψ
        dw isa AbstractArray && (tangent(w) .+= dw)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_mean_pb!!
end

function rrule!!(::CoDual{typeof(segmented_mean_forw)}, x::CoDual{Missing}, ψ::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags}, w::CoDual)
    ψp, bagsp, wp = primal(ψ), primal(bags), primal(w)
    y = segmented_mean_forw(missing, ψp, bagsp, wp)
    ȳ = zero(y)
    function segmented_mean_missing_pb!!(::NoRData)
        _, dψ, _, _ = Mill.segmented_mean_back(ȳ, y, missing, ψp, bagsp, wp)
        tangent(ψ) .+= dψ
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_mean_missing_pb!!
end

# ---- segmented_max_forw -------------------------------------------------------

@is_primitive DefaultCtx Tuple{typeof(segmented_max_forw), AbstractMatrix, AbstractVector, AbstractBags}
@is_primitive DefaultCtx Tuple{typeof(segmented_max_forw), Missing,        AbstractVector, AbstractBags}

function rrule!!(::CoDual{typeof(segmented_max_forw)}, x::CoDual{<:AbstractMatrix}, ψ::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags})
    xp, ψp, bagsp = primal(x), primal(ψ), primal(bags)
    y = segmented_max_forw(xp, ψp, bagsp)
    ȳ = zero(y)
    function segmented_max_pb!!(::NoRData)
        dx, dψ, _ = Mill.segmented_max_back(ȳ, y, xp, ψp, bagsp)
        tangent(x) .+= dx
        tangent(ψ) .+= dψ
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_max_pb!!
end

function rrule!!(::CoDual{typeof(segmented_max_forw)}, x::CoDual{Missing}, ψ::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags})
    ψp, bagsp = primal(ψ), primal(bags)
    y = segmented_max_forw(missing, ψp, bagsp)
    ȳ = zero(y)
    function segmented_max_missing_pb!!(::NoRData)
        _, dψ, _ = Mill.segmented_max_back(ȳ, y, missing, ψp, bagsp)
        tangent(ψ) .+= dψ
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_max_missing_pb!!
end

# ---- segmented_pnorm_forw -----------------------------------------------------
# Weight gradients are not implemented upstream (@not_implemented in the back
# function), so we intentionally leave the weight fdata unmodified (zero gradient).

@is_primitive DefaultCtx Tuple{typeof(segmented_pnorm_forw), AbstractMatrix, AbstractVector, AbstractVector, AbstractBags, Any}
@is_primitive DefaultCtx Tuple{typeof(segmented_pnorm_forw), Missing,        AbstractVector, Any,            AbstractBags, Any}

function rrule!!(::CoDual{typeof(segmented_pnorm_forw)}, a::CoDual{<:AbstractMatrix}, ψ::CoDual{<:AbstractVector}, p::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags}, w::CoDual)
    ap, ψp, pp, bagsp, wp = primal(a), primal(ψ), primal(p), primal(bags), primal(w)
    M = Mill._pnorm_precomp(ap, bagsp)
    y = Mill._segmented_pnorm_norm(ap, ψp, pp, bagsp, wp, M)
    ȳ = zero(y)
    function segmented_pnorm_pb!!(::NoRData)
        da, dψ, dp, _, _ = Mill.segmented_pnorm_back(ȳ, y, ap, ψp, pp, bagsp, wp, M)
        tangent(a) .+= da
        tangent(ψ) .+= dψ
        tangent(p) .+= dp
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_pnorm_pb!!
end

function rrule!!(::CoDual{typeof(segmented_pnorm_forw)}, a::CoDual{Missing}, ψ::CoDual{<:AbstractVector}, p::CoDual, bags::CoDual{<:AbstractBags}, w::CoDual)
    ψp, bagsp = primal(ψ), primal(bags)
    y = segmented_pnorm_forw(missing, ψp, primal(p), bagsp, primal(w))
    ȳ = zero(y)
    function segmented_pnorm_missing_pb!!(::NoRData)
        _, dψ, _, _, _ = Mill.segmented_pnorm_back(ȳ, y, ψp, bagsp)
        tangent(ψ) .+= dψ
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_pnorm_missing_pb!!
end

# ---- segmented_lse_forw -------------------------------------------------------

@is_primitive DefaultCtx Tuple{typeof(segmented_lse_forw), AbstractMatrix, AbstractVector, AbstractVector, AbstractBags}
@is_primitive DefaultCtx Tuple{typeof(segmented_lse_forw), Missing,        AbstractVector, AbstractVector, AbstractBags}

function rrule!!(::CoDual{typeof(segmented_lse_forw)}, x::CoDual{<:AbstractMatrix}, ψ::CoDual{<:AbstractVector}, r::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags})
    xp, ψp, rp, bagsp = primal(x), primal(ψ), primal(r), primal(bags)
    M = Mill._lse_precomp(xp, rp, bagsp)
    y = Mill._segmented_lse_norm(xp, ψp, rp, bagsp, M)
    ȳ = zero(y)
    function segmented_lse_pb!!(::NoRData)
        dx, dψ, dr, _ = Mill.segmented_lse_back(ȳ, y, xp, ψp, rp, bagsp, M)
        tangent(x) .+= dx
        tangent(ψ) .+= dψ
        tangent(r) .+= dr
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_lse_pb!!
end

function rrule!!(::CoDual{typeof(segmented_lse_forw)}, x::CoDual{Missing}, ψ::CoDual{<:AbstractVector}, r::CoDual{<:AbstractVector}, bags::CoDual{<:AbstractBags})
    ψp, bagsp = primal(ψ), primal(bags)
    y = segmented_lse_forw(missing, ψp, primal(r), bagsp)
    ȳ = zero(y)
    function segmented_lse_missing_pb!!(::NoRData)
        _, dψ, _, _ = Mill.segmented_lse_back(ȳ, missing, ψp, bagsp)
        tangent(ψ) .+= dψ
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), segmented_lse_missing_pb!!
end

# ---- bagconv ------------------------------------------------------------------

@is_primitive DefaultCtx Tuple{typeof(Mill.bagconv), AbstractMatrix, AbstractBags, Vararg{Matrix}}

function rrule!!(::CoDual{typeof(Mill.bagconv)}, x::CoDual{<:AbstractMatrix}, bags::CoDual{<:AbstractBags}, fs::CoDual{<:Matrix}...)
    xp, bagsp, fps = primal(x), primal(bags), map(primal, fs)
    y = Mill.bagconv(xp, bagsp, fps...)
    ȳ = zero(y)
    function bagconv_pb!!(::NoRData)
        tangent(x) .+= Mill.∇xbagconv(ȳ, xp, bagsp, fps...)
        for (fi, dfi) in zip(fs, Mill.∇wbagconv(ȳ, xp, bagsp, fps...))
            tangent(fi) .+= dfi
        end
        return NoRData(), NoRData(), NoRData(), map(_ -> NoRData(), fs)...
    end
    return CoDual(y, ȳ), bagconv_pb!!
end

# ---- convsum ------------------------------------------------------------------

@is_primitive DefaultCtx Tuple{typeof(Mill.convsum), AbstractBags, Vararg{AbstractMatrix}}

function rrule!!(::CoDual{typeof(Mill.convsum)}, bags::CoDual{<:AbstractBags}, xs::CoDual{<:AbstractMatrix}...)
    bagsp, xps = primal(bags), map(primal, xs)
    y = Mill.convsum(bagsp, xps...)
    ȳ = zero(y)
    function convsum_pb!!(::NoRData)
        for (xi, dxi) in zip(xs, Mill.∇convsum(ȳ, bagsp, length(xs)))
            tangent(xi) .+= dxi
        end
        return NoRData(), NoRData(), map(_ -> NoRData(), xs)...
    end
    return CoDual(y, ȳ), convsum_pb!!
end

# ---- _mul_pi_maybe (PreImputingMatrix) ----------------------------------------

@is_primitive DefaultCtx Tuple{typeof(Mill._mul_pi_maybe), AbstractVector, AbstractVecOrMat}

function rrule!!(::CoDual{typeof(Mill._mul_pi_maybe)}, ψ::CoDual{<:AbstractVector}, B::CoDual{<:AbstractVecOrMat})
    X, m = Mill._preimpute(primal(ψ), primal(B))
    ȳ = zero(X)
    function mul_pi_maybe_pb!!(::NoRData)
        tangent(ψ) .+= vec(sum(.!m .* ȳ, dims=2))
        # Propagate gradient to the non-missing entries of B.
        # Missing entries have NoTangent tangent type and are skipped.
        tB = tangent(B)
        @inbounds for j in axes(tB, 2), i in axes(tB, 1)
            m[i, j] && (tB[i, j] += ȳ[i, j])
        end
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(X, ȳ), mul_pi_maybe_pb!!
end

# ---- _mul_pi_maybe_hot (PostImputingMatrix × MaybeHotMatrix) ------------------

@is_primitive DefaultCtx Tuple{typeof(Mill._mul_pi_maybe_hot), PostImputingMatrix, MaybeHotMatrix}

function rrule!!(::CoDual{typeof(Mill._mul_pi_maybe_hot)}, A::CoDual{<:PostImputingMatrix}, B::CoDual{<:MaybeHotMatrix})
    Ap, Bp = primal(A), primal(B)
    C, m = Mill._postimpute_maybe_hot(Ap, Bp)
    Ĉ = zero(C)
    function mul_pi_maybe_hot_pb!!(::NoRData)
        dW = tangent(A).data.W
        @inbounds for (k, j) in enumerate(Bp.I)
            if !ismissing(j)
                for i in axes(dW, 1)
                    dW[i, j] += Ĉ[i, k]
                end
            end
        end
        tangent(A).data.ψ .+= vec(sum(view(Ĉ, :, m), dims=2))
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(C, Ĉ), mul_pi_maybe_hot_pb!!
end

# ---- _mul_pi_ngram (PostImputingMatrix × NGramMatrix) -------------------------

@is_primitive DefaultCtx Tuple{typeof(Mill._mul_pi_ngram), PostImputingMatrix, NGramMatrix}

function rrule!!(::CoDual{typeof(Mill._mul_pi_ngram)}, A::CoDual{<:PostImputingMatrix}, B::CoDual{<:NGramMatrix})
    Ap, Bp = primal(A), primal(B)
    C = Mill._postimpute_ngram(Ap, Bp)
    Ĉ = zero(C)
    function mul_pi_ngram_pb!!(::NoRData)
        dW = tangent(A).data.W
        z = Mill._init_z(Bp.n, Bp.b)
        bn = Bp.b^Bp.n
        for (k, s) in enumerate(Bp.S)
            Mill._∇A_mul_ngram_vec!(Ĉ, s, Bp, bn, dW, k, z)
        end
        tangent(A).data.ψ .+= vec(sum(view(Ĉ, :, ismissing.(Bp.S)), dims=2))
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(C, Ĉ), mul_pi_ngram_pb!!
end

end # module MillMooncakeExt
