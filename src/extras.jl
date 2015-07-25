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
