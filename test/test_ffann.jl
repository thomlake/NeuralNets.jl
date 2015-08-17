using Base.Test
using NeuralNets

function build_model(sizes::Vector{Int})
    nnet = NeuralNet(Tuple{Symbol,Int})
    for i = 2:length(sizes)
        nnet[(:W,i-1)] = Orthonormal(sqrt(2), sizes[i], sizes[i-1])
        nnet[(:b,i-1)] = Zeros(sizes[i])
    end
    nnet.metadata[:depth] = length(sizes) - 1
    return nnet
end

function predict(nnet::NeuralNet, input::Matrix{Float64}, target::Vector{Int})
    d = nnet.metadata[:depth]
    h = Block(input)
    @autograd nnet begin
        for i = 1:d - 1
            W = nnet[(:W,i)]
            b = nnet[(:b,i)]
            h = relu(affine(W, h, b))
        end
        W = nnet[(:W,d)]
        b = nnet[(:b,d)]
        scores = affine(W, h, b)
        cost = nll_categorical(target, scores)
    end
    return cost
end

n_samples = 20
n_features = 10
n_outputs = 15
sizes = [n_features, 30, 50, n_outputs]
X = randn(n_features, n_samples)
Y = rand(1:n_outputs, n_samples)
nnet = build_model(sizes)
@test gradcheck(nnet, ()->predict(nnet, X, Y), verbose=false)
