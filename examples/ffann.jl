using SoItGoes
using NeuralNets
import MNIST

function Model(n_in::Int, n_hid1::Int, n_hid2::Int, n_out::Int)
    W1 = Param(:Orthonormal, sqrt(2), n_hid1, n_in)
    W2 = Param(:Orthonormal, sqrt(2), n_hid2, n_hid1)
    U = Param(:Orthonormal, sqrt(2), n_out, n_hid2)
    bh1 = Param(n_hid1)
    bh2 = Param(n_hid2)
    bo = Param(n_out)
    G = Graph()
    dp = 0.5
    
    function feedforward(X::Matrix{Float64})
        x = Block(X)
        h1 = mult(1 - dp, relu(G, add(G, linear(G, W1, x), bh1)))
        h2 = mult(1 - dp, relu(G, add(G, linear(G, W2, h1), bh2)))
        Y_pred = add(G, linear(G, U, h2), bo)
        vec(mapslices(indmax, value(Y_pred), 1))
    end

    function feedforward(X::Matrix{Float64}, Y_true::Vector{Int})
        x = Block(X)
        h1 = dropout(relu(G, add(G, linear(G, W1, x), bh1)), dp)
        h2 = dropout(relu(G, add(G, linear(G, W2, h1), bh2)), dp)
        Y_pred = add(G, linear(G, U, h2), bo)
        loss = catloss(Y_true, Y_pred)
        loss, vec(mapslices(indmax, value(Y_pred), 1))
    end

    NeuralNet(G, [:W1=>W1, :W2=>W2, :U=>U, :bh1=>bh1, :bh2=>bh2, :bo=>bo], feedforward)
end

function image_string(x::Vector, symbols::Vector{Char}=['-', '+'])
    s = isqrt(length(x))
    img = reshape(x, (s, s))
    rows = ASCIIString[]
    for i in 1:s
        # push!(rows, join(symbols[int(img[i,:] .> 0) + 1]))
        push!(rows, join([x > 0 ? " $(round(x, 1)) " : " --- " for x in img[i,:]]))
    end
    join(rows, '\n')
end

function test()
    rawX, rawY = MNIST.traindata()
    mu = mean(rawX, 2)
    sigma = std(rawX, 2)
    trX = normalize(rawX, mu, sigma)[:,1:10]
    trY = int(rawY + 1)[1:10]
    nnet = Model(size(trX, 1), 20, 10, maximum(trY))
    fwd() = nnet.feedforward(trX, trY)[1]
    gradcheck(nnet, fwd)
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
    
    nnet = Model(size(trX, 1), 500, 200, maximum(trY))

    n_train = size(trX, 2)
    batch_size = 100
    indices = collect(1:n_train)
    fmt(f) = rpad(round(f, 3), 5, "0")

    for epoch = 1:100
        loss = 0
        error = 0
        shuffle!(indices)
        for i = 1:batch_size:n_train
            idx = indices[i:min(i + batch_size - 1, end)]
            X, Y = trX[:,idx], trY[idx]
            loss_batch, Y_pred = nnet.feedforward(X, Y)
            loss += loss_batch
            error += sum(Y .!= Y_pred)
            backprop(nnet)
            sgd(nnet, 0.01 / length(idx))
        end
        println("[epoch => $(lpad(epoch, 3)),  loss => $(fmt(loss / n_train)), error => $(fmt(error / n_train)) ($error of $n_train)]")
    end

    rawX, rawY = MNIST.testdata()
    teX, teY = normalize(rawX, mu, sigma), int(rawY + 1)
    n_test = length(teY)
    Y_pred = nnet.feedforward(teX)
    error = sum(teY .!= Y_pred)
    println("test error => $(fmt(error / n_test)) ($error of $n_test)")
end

# test()
fit()
# show_example_data()