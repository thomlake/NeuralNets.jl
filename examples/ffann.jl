import MNIST
using NeuralNets
const nnx = NeuralNets.Extras

function build_model(sizes::Vector{Int})
    nnet = NeuralNet((Symbol,Int))
    for i = 2:length(sizes)
        nnet[(:W,i-1)] = Orthonormal(sqrt(2), sizes[i], sizes[i-1])
        nnet[(:b,i-1)] = Zeros(sizes[i])
    end
    nnet.metadata[:depth] = length(sizes) - 1
    return nnet
end

function predict(nnet::NeuralNet, input::Matrix{Float64}, dp::Float64=0.5)
    d = nnet.metadata[:depth]
    h = Block(input)
    for i = 1:d - 1
        W = nnet[(:W,i)]
        b = nnet[(:b,i)]
        h = mult(1 - dp, relu(affine(W, h, b)))
    end
    W = nnet[(:W,d)]
    b = nnet[(:b,d)]
    scores = affine(W, h, b)
    return nnx.argmax(value(scores))
end 

function predict(nnet::NeuralNet, input::Matrix{Float64}, target::Vector{Int}, dp::Float64=0.5)
    d = nnet.metadata[:depth]
    h = Block(input)
    @autograd nnet begin
        for i = 1:d - 1
            W = nnet[(:W,i)]
            b = nnet[(:b,i)]
            h = dropout(dp, relu(affine(W, h, b)))
        end
        W = nnet[(:W,d)]
        b = nnet[(:b,d)]
        scores = affine(W, h, b)
        cost = nll_categorical(target, scores)
    end
    return cost, nnx.argmax(value(scores))
end

function image_string(x::Vector, symbols::Vector{Char}=['-', '+'])
    s = isqrt(length(x))
    img = reshape(x, (s, s))
    rows = ASCIIString[]
    for i in 1:s
        push!(rows, join(symbols[int(img[i,:] .> 0) + 1]))
    end
    return join(rows, '\n')
end

function test()
    rawX, rawY = MNIST.traindata()
    trX = nnx.zscore(rawX)[:,1:10]
    trY = int(rawY + 1)[1:10]
    @assert all(isfinite(trX))
    @assert all(isfinite(trY))

    sizes = [size(trX, 1), 10, 5, 10, maximum(trY)]
    nnet = build_model(sizes)

    f() = predict(nnet, trX, trY, 0.0)[1]
    gradcheck(nnet, f)
end

function show_example_data()
    rawX, rawY = MNIST.traindata()
    X = nnx.zscore(rawX)
    @assert all(isfinite(X))
    for i = 1:10
        println(int(rawY[i]))
        println(image_string(X[:,i]))
    end
end

function fit()
    srand(123)
    rawX, rawY = MNIST.traindata()
    const mu = mean(rawX, 2)
    const sigma = std(rawX, 2)
    trX = nnx.zscore(rawX, mu, sigma)
    trY = int(rawY + 1)
    @assert all(isfinite(trX))
    @assert all(isfinite(trY))
    
    sizes = [size(trX, 1), 200, 200, 200, maximum(trY)]
    nnet = build_model(sizes)

    n_train = size(trX, 2)
    batch_size = 100
    indices = collect(1:n_train)
    tol = 1e-3
    bestcost = Inf
    patience = 5
    frustration = 0
    epochs = 0
    function statusmsg(cost, error)
        n = lpad(epochs, 3)
        c = rpad(round(cost / n_train, 3), 5, " ")
        b = rpad(round(bestcost / n_train, 3), 5, " ")
        e = rpad(round(error / n_train, 3), 5, " ")
        println("[epoch => $n, frustration => $frustration, bestcost => $b, cost => $c, error => $e ($error of $n_train)]")
    end

    println("fitting...")
    while frustration < patience
        epochs += 1
        cost = 0
        error = 0
        shuffle!(indices)
        for i = 1:batch_size:n_train
            idx = indices[i:min(i + batch_size - 1, end)]
            X, Y = trX[:,idx], trY[idx]
            cost_batch, Y_pred = predict(nnet, X, Y)
            cost += cost_batch
            error += sum(Y .!= Y_pred)
            backprop(nnet)
            rmsprop!(nnet, 0.1 / length(idx), 0.8)
        end
        statusmsg(cost, error)
        frustration = tol > (bestcost - cost) ? frustration + 1 : 0
        bestcost = min(bestcost, cost)
    end
    println("converged after $epochs epochs")
    println("testing...")
    rawX, rawY = MNIST.testdata()
    teX = nnx.zscore(rawX, mu, sigma)
    teY = int(rawY + 1)
    n_test = length(teY)
    Y_pred = predict(nnet, teX)
    error = sum(teY .!= Y_pred)
    println("test error => $(error / n_test) ($error of $n_test)")
end

# test()
fit()
# show_example_data()