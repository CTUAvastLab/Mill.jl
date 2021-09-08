import HierarchicalUtils: nodeshow, nodecommshow

function datasummary(n::AbstractMillNode)
    bytes = Base.format_bytes(Base.summarysize(n))
    string("# Summary: ", nobs(n), " obs, ", bytes, ".")
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(n::AbstractMillNode))
    HierarchicalUtils.printtree(io, n; htrunc=3, vtrunc=5, breakline=false)
end

nodeshow(io::IO, ::Missing) = print(io, "∅")
nodeshow(io::IO, n::LazyNode{N, Nothing}) where N = print(io, "LazyNode{$N} ∅")

function nodecommshow(io::IO, @nospecialize(n::AbstractMillNode))
    bytes = Base.format_bytes(Base.summarysize(n) - (isleaf(n) ? 0 : Base.summarysize(data(n))))
    print(io, "\t# ", nobs(n), " obs, ", bytes)
end


function Base.show(io::IO, @nospecialize(n::AbstractMillNode))
    print(io, nameof(typeof(n)))
    if !get(io, :compact, false)
        _show_data(io, n)
    end
end

_show_data(io, n::LazyNode{Name}) where {Name} = print(io, "{", Name, "}")
_show_data(io, _) = print(io)

# params summary from https://github.com/FluxML/Flux.jl/blob/master/src/layers/show.jl
function modelsummary(m::AbstractMillModel)
    ps = params(m)
    npars = Flux.underscorise(sum(length, ps))
    bytes = Base.format_bytes(sum(Base.summarysize, collect(ps.params)))
    string("# Summary: ", length(ps), " arrays, ", npars, " params, ", bytes)
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(m::AbstractMillModel))
    HierarchicalUtils.printtree(io, m; htrunc=3, vtrunc=5, breakline=false)
end

_levelparams(m::ArrayModel) = params(m.m)
_levelparams(m::BagModel) = params(m.a, m.bm)
_levelparams(m::ProductModel) = params(m.m)
_levelparams(m::LazyModel) = Params([])

# params summary from https://github.com/FluxML/Flux.jl/blob/master/src/layers/show.jl
function nodecommshow(io::IO, @nospecialize(m::AbstractMillModel))
    ps = _levelparams(m)
    if !isempty(ps)
        npars = Flux.underscorise(sum(length, ps))
        print(io, "\t# ", length(ps), " arrays, ", npars, " params")
        if !isempty(ps) && Flux._all(iszero, ps)
            print(io, " (all zero)")
        elseif Flux._any(isnan, ps)
            print(io, " (some NaN)")
        elseif Flux._any(isinf, ps)
            print(io, " (some Inf)")
        end
        bytes = Base.format_bytes(sum(Base.summarysize, collect(ps.params)))
        print(io, ", ", bytes)
    end
end

function Base.show(io::IO, @nospecialize(n::AbstractMillModel))
    print(io, nameof(typeof(n)))
    if !get(io, :compact, false)
        _show_submodels(io, n)
    end
end

_show_submodels(io, m::ArrayModel) = print(io, "(", m.m, ")")
_show_submodels(io, m::BagModel) = print(io, " ↦ ", m.a, " ↦ ", m.bm)
_show_submodels(io, m::ProductModel) = print(io, " ↦ ", m.m)
_show_submodels(io, m::LazyModel{Name}) where {Name} = print(io, "{", Name, "}")
_show_submodels(io, _) = print(io)

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

Base.show(io::IO, ::MIME"text/plain", a::AbstractAggregation) = _show_fields(io, a)

function _show_fields(io, x::T; context=:compact=>true) where T
    print(io, nameof(T), "(", join(["$f = $(repr(getfield(x, f); context))" for f in fieldnames(T)],", "), ")")
end

function Base.show(io::IO, l::Dense{F, <:ImputingMatrix}) where F
    print(io, _name(l.weight), "Dense(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == Flux.Zeros() && print(io, "; bias=false")
    print(io, ")")
end

_name(::PreImputingMatrix) = "[preimputing]"
_name(::PostImputingMatrix) = "[postimputing]"

function Base.show(io::IO, a::T) where T <: AbstractAggregation
    print(io, nameof(T))
    if !get(io, :compact, false)
        print(io, "(", length(a), ")")
    end
end
