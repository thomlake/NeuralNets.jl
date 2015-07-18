
function sgd(nnet::NeuralNet, lr::Real)
    for theta in nnet.params
        theta.x -= lr * theta.dx
        fill!(theta.dx, 0)
    end
end

function sgd(nnet::NeuralNet, lr::Real, gradclip::Real)
    for theta in nnet.params
        theta.x -= lr * max(min(theta.dx, gradclip), -gradclip)
        fill!(theta.dx, 0)
    end
end

function rmsprop(nnet::NeuralNet, cached_grads::Vector{Matrix}, lr::Real, rho::Real, gradclip::Real)
    @assert length(nnet.params) == length(cached_grads)
    umrho = 1 - rho
    for n = 1:length(nnet.params)
        theta = nnet.params[n]
        cache = cached_grads[n]
        @assert size(theta) == size(cache)
        for i = 1:length(cache)
            cache[i] = rho * cache[i] + umrho * theta.dx[i] * theta.dx[i]
            theta.dx[i] = theta.dx[i] / sqrt(cache[i] + 1e-6)
        end
        theta.x -= lr * max(min(theta.dx, gradclip), -gradclip)
        fill!(theta.dx, 0)
    end
end

function rmsprop(nnet::NeuralNet, cached_grads::Vector{Matrix}, lr::Real, rho::Real)
    @assert length(nnet.params) == length(cached_grads)
    umrho = 1 - rho
    for n = 1:length(nnet.params)
        theta = nnet.params[n]
        cache = cached_grads[n]
        @assert size(theta) == size(cache)
        for i = 1:length(cache)
            cache[i] = rho * cache[i] + umrho * theta.dx[i] * theta.dx[i]
            theta.dx[i] = theta.dx[i] / sqrt(cache[i] + 1e-6)
        end
        theta.x -= lr * theta.dx
        fill!(theta.dx, 0)
    end
end