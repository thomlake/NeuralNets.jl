# -- Utiliy Functions -- #
using ..NeuralNets
vec2mat(b::Vector) = reshape(b, (size(b, 1), 1))

onehot(i::Int, d::Int) = (x = zeros(d); x[i] = 1; x)

argmax(x::Vector) = indmax(x)

function argmax(x::Matrix)
    n_rows, n_cols = size(x)
    imax = zeros(Int, n_cols)
    for j = 1:n_cols
        m = -Inf
        for i = 1:n_rows
            if x[i,j] > m
                m = x[i,j]
                imax[j] = i
            end
        end
    end
    return imax
end

argmax(x::Block) = argmax(x.x)

zscore(x, mu, sigma, sigma_min::FloatingPoint=1e-6) = (x .- mu) ./ max(sigma, sigma_min)

function zscore(x::Matrix, sigma_min::FloatingPoint=1e-6)
    mu = mean(x, 2)
    sigma = std(x, 2)
    return zscore(x, mu, sigma, sigma_min)
end

zscore(xs::Vector{Matrix}, mu, sigma, sigma_min::FloatingPoint=1e-6) = map(x->zscore(x, mu, sigma, sigma_min), xs)

function zscore(xs::Vector{Matrix}, sigma_min::FloatingPoint=1e-6)
    d = size(xs[1], 1)
    n = 0

    mu = zeros(d)
    for x in xs
        mu .+= sum(x, 2)
        n += size(x, 2)
    end
    mu ./= n

    sigma = zeros(d)
    for x in xs
        for j = 1:size(x, 2)
            for i = 1:d
                diff = (x[i,j] - mu[i])
                sigma[i] += diff * diff
            end
        end
    end
    sigma = sqrt(sigma ./ (n - 1))
    return zscore(xs, mu, sigma, 1e-6)
end

# -- Artificial Data Generation -- #
function randxor(T::Int)
    x_bits = rand(0:1, T)
    xs = [onehot(b + 1, 2) for b in x_bits]
    ys = (cumsum(x_bits) % 2) + 1
    return xs, ys
end

function randxor(r::UnitRange{Int}, n_samples::Int=1)
    X = Vector{Vector{Float64}}[]
    Y = Vector{Int}[]
    for i = 1:n_samples
        T = rand(r)
        xs, ys = randxor(T)
        push!(X, xs)
        push!(Y, ys)
    end
    return X, Y
end

function randblobs(n_classes::Int, n_dims::Int, n_samples::Int)
    mu = Vector[randn(n_dims) for i = 1:n_classes]
    sigma = Vector[abs(randn(n_dims)) for i = 1:n_classes]
    X, Y = zeros(n_dims, n_samples), zeros(Int, n_samples)
    for i = 1:n_samples
        Y[i] = rand(1:n_classes)
        X[:,i] = sigma[Y[i]] .* randn(n_dims) .+ mu[Y[i]]
    end
    return X, Y
end

