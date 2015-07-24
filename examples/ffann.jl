using SoItGoes
using NeuralNets
import MNIST

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
    return vec(mapslices(indmax, value(scores), 1))
end 

function predict(nnet::NeuralNet, input::Matrix{Float64}, target::Vector{Int}, dp::Float64=0.5)
    d = nnet.metadata[:depth]
    h = Block(input)
    @grad nnet begin
        for i = 1:d - 1
            W = nnet[(:W,i)]
            b = nnet[(:b,i)]
            h = dropout(dp, relu(affine(W, h, b)))
            # h = relu(affine(W, h, b))
        end
        W = nnet[(:W,d)]
        b = nnet[(:b,d)]
        scores = affine(W, h, b)
        cost = nll_categorical(target, scores)
    end
    return cost, vec(mapslices(indmax, value(scores), 1))
end

function image_string(x::Vector, symbols::Vector{Char}=['-', '+'])
    s = isqrt(length(x))
    img = reshape(x, (s, s))
    rows = ASCIIString[]
    for i in 1:s
        push!(rows, join(symbols[int(img[i,:] .> 0) + 1]))
        # push!(rows, join([x > 0 ? " $(round(x, 1)) " : " --- " for x in img[i,:]]))
    end
    join(rows, '\n')
end

function test()
    rawX, rawY = MNIST.traindata()
    mu = mean(rawX, 2)
    sigma = std(rawX, 2)
    trX = normalize(rawX, mu, sigma)[:,1:10]
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
    mu = mean(rawX, 2)
    sigma = std(rawX, 2) + 1e-5
    X = normalize(rawX, mu, sigma)
    for i = 1:10
        println(int(rawY[i]))
        println(image_string(X[:,i]))
    end
end

function fit()
    srand(123)
    rawX, rawY = MNIST.traindata()
    mu = mean(rawX, 2)
    sigma = max(std(rawX, 2), 1e-6)
    trX = normalize(rawX, mu, sigma)
    trY = int(rawY + 1)
    
    sizes = [size(trX, 1), 200, 200, 200, maximum(trY)]
    nnet = build_model(sizes)

    n_train = size(trX, 2)
    batch_size = 100
    indices = collect(1:n_train)
    ffmt(f) = rpad(round(f, 3), 5, "0")
    tol = 1e-3
    
        loss = 0
        error = 0
        shuffle!(indices)
        for i = 1:batch_size:n_train
            idx = indices[i:min(i + batch_size - 1, end)]
            X, Y = trX[:,idx], trY[idx]
            loss_batch, Y_pred = predict(nnet, X, Y)
            loss += loss_batch
            error += sum(Y .!= Y_pred)
            backprop(nnet)
            sgd(nnet, 0.01 / length(idx))
        end
        println("[epoch => $(lpad(epoch, 3)),  loss => $(ffmt(loss / n_train)), error => $(ffmt(error / n_train)) ($error of $n_train)]")
    end

    rawX, rawY = MNIST.testdata()
    teX, teY = normalize(rawX, mu, sigma), int(rawY + 1)
    n_test = length(teY)
    Y_pred = predict(nnet, teX)
    error = sum(teY .!= Y_pred)
    println("test error => $(ffmt(error / n_test)) ($error of $n_test)")
end

test()
# fit()
# show_example_data()