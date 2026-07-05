```@setup mooncake
using Mill, Flux, Mooncake
```

# Differentiating with Mooncake.jl

[`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) ships a native
[`Mooncake.jl`](https://github.com/compintell/Mooncake.jl) extension that registers efficient
differentiation rules for all custom primitives (bag aggregations, imputing matrices, convolutions).
Loading both packages is sufficient to activate it:

```julia
using Mill, Mooncake
```

## Why Mooncake over Zygote?

[`Zygote.jl`](https://github.com/FluxML/Zygote.jl) has been the default AD back-end for the
[`Flux.jl`](https://fluxml.ai) ecosystem for several years, and it works well for many use cases.
There are, however, good reasons to consider Mooncake as an alternative:

* **Active development.** Zygote is in maintenance mode; Mooncake is where new features and
  correctness fixes are being invested.
* **Mutation support.** Mooncake handles in-place operations correctly by design. Zygote silently
  produces wrong gradients for mutating code.
* **Correct variable-length inputs without padding.** [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl)
  is built around hierarchical, variable-sized structures such as sets of JSON documents. Padding
  all inputs to a fixed size (the workaround needed by static-graph backends like
  [`Reactant.jl`](https://github.com/EnzymeAD/Reactant.jl)) is often wasteful or impossible.
  Mooncake differentiates dynamic Julia code natively.
* **Composability.** Mooncake's `CoDual`-based rules compose predictably across custom types,
  nested AD, and mixed-mode AD.

## Basic usage

The API mirrors what you already know from Flux/Zygote. The key difference is that Mooncake
*compiles* a gradient cache once and then reuses it across training steps:

```julia
using Mill, Flux, Mooncake

# ---- data ----------------------------------------------------------------
ds = BagNode(ArrayNode(randn(Float32, 4, 10)), [1:3, 4:7, 8:10])
y  = onehotbatch([1, 2, 1], 1:2)   # using OneHotArrays

# ---- model ---------------------------------------------------------------
model = BagModel(
    Dense(4 => 8, tanh),
    SegmentedMeanMax(8),
    Chain(Dense(16 => 8, tanh), Dense(8 => 2)))

# ---- loss ----------------------------------------------------------------
loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)

# ---- compile the gradient cache once ------------------------------------
opt_state  = Flux.setup(Adam(), model)
grad_cache = Mooncake.prepare_gradient_cache(loss, model, ds, y)

# ---- training loop -------------------------------------------------------
for epoch in 1:100
    _, (_, grad_model, _, _) = Mooncake.value_and_gradient!!(grad_cache, loss, model, ds, y)
    Flux.update!(opt_state, model, grad_model)
end
```

`value_and_gradient!!` returns a `(value, (grad_f, grad_arg1, grad_arg2, ...))` tuple. The
leading `grad_f` is the gradient with respect to the function itself (always `NoTangent()` for a
plain Julia function), so the model gradient is the second element of the inner tuple.

The `!!` suffix signals that Mooncake accumulates cotangents **in-place** into the buffers inside
`grad_cache`. This is intentional: it eliminates allocations on the hot path. Calling
`value_and_gradient!!` a second time with the same cache is safe and efficient.

## Performance tips

### Compile once, run many times

`prepare_gradient_cache` triggers JIT compilation of the entire forward and reverse pass. This
takes a few seconds the first time but subsequent `value_and_gradient!!` calls have no compilation
overhead. Always hoist cache construction out of the training loop.

### Variable-sized inputs and `resize_gradient_cache`

A common pattern in [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) workflows is *minibatch
training* where each batch contains a different number of instances. Because the gradient cache
stores output buffers sized to match the inputs, using a batch with a different size than the
one passed to `prepare_gradient_cache` would require a fresh cache—and therefore recompilation.

Mooncake provides `resize_gradient_cache` to handle this efficiently. It checks whether the buffer
sizes still match and, if not, allocates new output buffers **without recompiling the rule**:

```julia
# Build the cache on the first batch.
batch1      = BagNode(ArrayNode(randn(Float32, 4, 10)), [1:5, 6:10])
grad_cache  = Mooncake.prepare_gradient_cache(loss, model, batch1, y1)

# Resize to a different batch — no recompilation.
batch2     = BagNode(ArrayNode(randn(Float32, 4, 7)), [1:3, 4:7])
grad_cache = Mooncake.resize_gradient_cache(grad_cache, loss, model, batch2, y2)

val, grads = Mooncake.value_and_gradient!!(grad_cache, loss, model, batch2, y2)
```

`resize_gradient_cache` returns the *same* cache object unchanged when sizes already match (zero
allocation, zero cost), so it is safe to call it at the top of every training step:

```julia
for (batch, labels) in dataloader
    grad_cache = Mooncake.resize_gradient_cache(grad_cache, loss, model, batch, labels)
    _, (_, grad_model, _, _) = Mooncake.value_and_gradient!!(grad_cache, loss, model, batch, labels)
    Flux.update!(opt_state, model, grad_model)
end
```

!!! note "Type stability requirement"
    `resize_gradient_cache` requires that the **type** of every argument stays constant across
    calls. Changing, for example, the element type from `Float32` to `Float64` raises a
    `PreparedCacheSpecError`. When that happens, call `prepare_gradient_cache` to obtain a fresh
    cache.

### Keeping data out of the gradient computation

When differentiating a function `loss(model, data, labels)`, Mooncake computes gradients for *all*
arguments by default—including `data` and `labels`, which are not parameters and have no gradient
you care about. This is harmless (the gradient is simply zero) but it does cause Mooncake to
allocate cotangent buffers for every array inside `data`.

For workflows where the graph or input structure changes every iteration—such as training a graph
neural network on streaming data—those per-iteration allocations can be significant. You can
eliminate them by wrapping non-differentiable arguments in a `struct` whose `tangent_type` is
declared as `NoTangent`:

```julia
struct ConstParam{T}
    val::T
end

Mooncake.tangent_type(::Type{<:ConstParam}) = Mooncake.NoTangent

# Wrap data and unwrap inside the loss:
loss_const(m, data, labels) = loss(m, data.val, labels.val)

# Compile once on any representative batch.
grad_cache = Mooncake.prepare_gradient_cache(
    loss_const, model, ConstParam(batch1), ConstParam(y1))

# Each iteration: rewrap new data, no new cache.
for (batch, labels) in dataloader
    _, (_, grad_model, _, _) = Mooncake.value_and_gradient!!(
        grad_cache, loss_const, model, ConstParam(batch), ConstParam(labels))
    Flux.update!(opt_state, model, grad_model)
end
```

The `ConstParam` wrapper tells Mooncake that the wrapped value is opaque: no cotangent buffers
are allocated for it, and no backward rule is run through its contents. The model gradient is
computed correctly because the model itself is not wrapped.

!!! note "No resize needed with `ConstParam`"
    Because `ConstParam` has `NoTangent`, Mooncake never inspects the size of the wrapped array.
    The same cache can be reused verbatim for every batch size without calling
    `resize_gradient_cache`.

## Complete training example

The example below is a Mooncake adaptation of the [Musk](@ref) example. It uses
[`reflectinmodel`](@ref) to construct the model automatically and trains it with Mooncake:

```julia
using FileIO, JLD2, Statistics, Mill, Flux, OneHotArrays, Mooncake

# --- load data ---
fMat   = load("musk.jld2", "fMat")
bagids = load("musk.jld2", "bagids")
y_raw  = load("musk.jld2", "y")

ds    = BagNode(ArrayNode(fMat), bagids)
y_bag = map(i -> maximum(y_raw[i]) + 1, ds.bags)
y_oh  = onehotbatch(y_bag, 1:2)

# --- build model via reflection ---
model = reflectinmodel(ds, d -> Dense(d, 50, tanh),
    BagCount ∘ SegmentedMeanMax;
    fsm = Dict("" => d -> Chain(Dense(d, 50, tanh), Dense(50, 2))))

# --- training with Mooncake ---
loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)

opt_state  = Flux.setup(Adam(), model)
grad_cache = Mooncake.prepare_gradient_cache(loss, model, ds, y_oh)

for epoch in 1:100
    if epoch % 10 == 1
        @info "Epoch $epoch" loss = loss(model, ds, y_oh)
    end
    _, (_, g, _, _) = Mooncake.value_and_gradient!!(grad_cache, loss, model, ds, y_oh)
    Flux.update!(opt_state, model, g)
end

@info "Training accuracy" acc = mean(Flux.onecold(model(ds), 1:2) .== y_bag)
```

## Developer notes

### Extension structure

The Mooncake rules for [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl) live in a Julia
package extension (`ext/MillMooncakeExt.jl`) that is loaded automatically when both `Mill` and
`Mooncake` are present. The extension is activated by the `[weakdeps]` / `[extensions]` entries
in `Project.toml`; no user action is required.

Each rule follows the Mooncake `rrule!!` convention:

```julia
@is_primitive DefaultCtx Tuple{typeof(segmented_sum_forw), AbstractMatrix, AbstractVector,
                                AbstractBags, Nothing}

function rrule!!(
    ::CoDual{typeof(segmented_sum_forw)},
    x::CoDual{<:AbstractMatrix},
    ψ::CoDual{<:AbstractVector},
    bags::CoDual{<:AbstractBags},
    w::CoDual,
)
    xp, ψp, bagsp, wp = primal(x), primal(ψ), primal(bags), primal(w)
    y = segmented_sum_forw(xp, ψp, bagsp, wp)
    ȳ = zero(y)                    # fdata: gradient accumulator for the output
    function pb!!(::NoRData)
        dx, dψ, _, dw = Mill.segmented_sum_back(ȳ, y, xp, ψp, bagsp, wp)
        tangent(x) .+= dx          # accumulate into input fdata in-place
        tangent(ψ) .+= dψ
        dw isa AbstractArray && (tangent(w) .+= dw)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, ȳ), pb!!
end
```

The key points:

* `@is_primitive` declares the function as a leaf: Mooncake will call our `rrule!!` rather than
  tracing into the function body.
* `CoDual(primal, fdata)` pairs each value with its heap-allocated cotangent buffer. For arrays,
  `fdata` is a same-shaped zero array; for scalars it is `NoFData`.
* The pullback **accumulates** (`+=`) into existing buffers. It never replaces them.
* The pullback returns one `NoRData()` per positional argument of `rrule!!`, including the
  leading function argument.
* Existing `_back` functions from the [`ChainRulesCore.jl`](https://github.com/JuliaDiff/ChainRulesCore.jl)
  rules are reused directly, so the derivative logic is not duplicated.

Rules that have a zero derivative (e.g. `_bagcount`) use the convenience macro:

```julia
@zero_derivative DefaultCtx Tuple{typeof(Mill._bagcount), Any, Any}
```

### Tangent types for imputing matrices

[`PreImputingMatrix`](@ref) and [`PostImputingMatrix`](@ref) are immutable structs that subtype
`AbstractMatrix`. Mooncake infers their tangent type from the struct fields, giving a
`Tangent{T, @NamedTuple{W::Matrix{F}, ψ::Vector{F}}}`. Inside a pullback, field gradients are
accessed with `Mooncake.get_tangent_field`:

```julia
dW = get_tangent_field(tangent(A), :W)   # returns the mutable gradient matrix
dψ = get_tangent_field(tangent(A), :ψ)   # returns the mutable gradient vector
dW .+= ...
dψ .+= ...
```

`get_tangent_field` handles the `PossiblyUninitTangent` wrapper that Mooncake may insert for
optional struct fields, so it is preferred over direct field access.

### Testing

The test suite in `test/mooncake.jl` compares Mooncake gradients against Zygote for every
registered primitive. To run it in isolation:

```sh
julia --project=test -e 'include("test/mooncake.jl")'
```
