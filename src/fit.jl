
function sgd!(nnet::NeuralNet, lr::Real)
    for theta in values(nnet.params)
        theta.x -= lr * theta.dx
        fill!(theta.dx, 0)
    end
end

function sgd!(nnet::NeuralNet, lr::Real, gradclip::Real)
    for theta in values(nnet.params)
        theta.x -= lr * max(min(theta.dx, gradclip), -gradclip)
        fill!(theta.dx, 0)
    end
end

function addcache!(nnet::NeuralNet, name)
    nnet.metadata[name] = [n => zero(p.x) for (n, p) in nnet.params]
end

function momentum!(nnet::NeuralNet, lr::Real, mu::Real)
    if !haskey(nnet.metadata, :momentum_cache)
        addcache!(nnet, :momentum_cache)
    end
    cached_grads = nnet.metadata[:momentum_cache]
    @assert length(nnet.params) == length(cached_grads)
    for (name, theta) in nnet.params
        cache = cached_grads[name]
        @assert size(theta) == size(cache)
        for i = 1:length(cache)
            cache[i] = mu * cache[i] + lr * theta.dx[i]
        end
        theta.x -= cache
        fill!(theta.dx, 0)
    end
end

function momentum!(nnet::NeuralNet, lr::Real, mu::Real, gradclip::Real)
    if !haskey(nnet.metadata, :momentum_cache)
        addcache!(nnet, :momentum_cache)
    end
    cached_grads = nnet.metadata[:momentum_cache]
    @assert length(nnet.params) == length(cached_grads)
    for (name, theta) in nnet.params
        cache = cached_grads[name]
        @assert size(theta) == size(cache)
        for i = 1:length(cache)
            cache[i] = mu * cache[i] + lr * theta.dx[i]
        end
        theta.x -= max(min(cache, gradclip), -gradclip)
        fill!(theta.dx, 0)
    end
end


function rmsprop!(nnet::NeuralNet, lr::Real, rho::Real, gradclip::Real)
    if !haskey(nnet.metadata, :rmsprop_cache)
        addcache!(nnet, :rmsprop_cache)
    end
    cached_grads = nnet.metadata[:rmsprop_cache]
    @assert length(nnet.params) == length(cached_grads)
    umrho = 1 - rho
    for (name, theta) in nnet.params
        cache = cached_grads[name]
        @assert size(theta) == size(cache)
        for i = 1:length(cache)
            cache[i] = rho * cache[i] + umrho * theta.dx[i] * theta.dx[i]
            theta.dx[i] = theta.dx[i] / sqrt(cache[i] + 1e-6)
        end
        theta.x -= lr * max(min(theta.dx, gradclip), -gradclip)
        fill!(theta.dx, 0)
    end
end

function rmsprop!(nnet::NeuralNet, lr::Real, rho::Real)
    if !haskey(nnet.metadata, :rmsprop_cache)
        addcache!(nnet, :rmsprop_cache)
    end
    cached_grads = nnet.metadata[:rmsprop_cache]
    @assert length(nnet.params) == length(cached_grads)
    umrho = 1 - rho
    for (name, theta) in nnet.params
        cache = cached_grads[name]
        @assert size(theta) == size(cache)
        for i = 1:length(cache)
            cache[i] = rho * cache[i] + umrho * theta.dx[i] * theta.dx[i]
            theta.dx[i] = theta.dx[i] / sqrt(cache[i] + 1e-6)
        end
        theta.x -= lr * theta.dx
        fill!(theta.dx, 0)
    end
end