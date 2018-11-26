function segmented_mean(x::Matrix, bags::Bags)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j,b) in enumerate(bags)
        for bi in b
            for i in 1:size(x, 1)
                o[i,j] += x[i,bi] / length(b)
            end
        end
    end
    o
end

function segmented_mean(x::Matrix, bags::Bags, w::Vector)
    o = zeros(eltype(x), size(x, 1), length(bags))
    @inbounds for (j,b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x, 1)
                o[i,j] += w[bi] * x[i,bi] / ws
            end
        end
    end
    o
end

function segmented_mean_back(Δ, x::TrackedMatrix, bags::Bags)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = similar(x)
    @inbounds for (j,b) in enumerate(bags)
        for bi in b
            for i in 1:size(x,1)
                dx[i,bi] = Δ[i,j] / length(b)
            end
        end
    end
    dx, nothing
end

function segmented_mean_back(Δ, x::TrackedMatrix, bags::Bags, w::Vector)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    dx = similar(x)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = w[bi] * Δ[i, j] / ws
            end
        end
    end
    dx, nothing, nothing
end

function segmented_mean_back(Δ, x::Matrix, bags::Bags, w::TrackedVector)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    w = Flux.data(w)
    n = segmented_mean(x, bags, w)
    dw = zero(w)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws
            end
        end
    end
    nothing, nothing, dw
end

function segmented_mean_back(Δ, x::TrackedMatrix, bags::Bags, w::TrackedVector)
    x = Flux.data(x)
    Δ = Flux.data(Δ)
    w = Flux.data(w)
    n = segmented_mean(x, bags, w)
    dx = similar(x)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dx[i, bi] = w[bi] * Δ[i, j] / ws
            end
        end
    end
    dw = zero(w)
    @inbounds for (j, b) in enumerate(bags)
        ws = sum(@view w[b])
        for bi in b
            for i in 1:size(x,1)
                dw[bi] += Δ[i, j] * (x[i, bi] - n[i, j]) / ws
            end
        end
    end
    dx, nothing, dw
end

