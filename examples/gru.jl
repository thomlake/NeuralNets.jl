using NeuralNets
const nnx = NeuralNets.Extras

function build_model(n_in::Int, n_hid::Int, n_out::Int)
    nnet = NeuralNet()
    nnet[:Wr] = Orthonormal(1.2, n_hid, n_in)
    nnet[:Wz] = Orthonormal(1.2, n_hid, n_in)
    nnet[:Wc] = Orthonormal(1.2, n_hid, n_in)

    nnet[:Ur] = Orthonormal(1.2, n_hid, n_hid)
    nnet[:Uz] = Orthonormal(1.2, n_hid, n_hid)
    nnet[:Uc] = Orthonormal(1.2, n_hid, n_hid)

    nnet[:W_out] = Orthonormal(1.2, n_out, n_hid)

    nnet[:h0] = Zeros(n_hid)
    nnet[:br] = Zeros(n_hid)
    nnet[:bz] = Zeros(n_hid)
    nnet[:bc] = Zeros(n_hid)
    nnet[:b_out] = Zeros(n_out)
    
    return nnet
end

function predict(nnet::NeuralNet, input::Vector, target::Vector{Int})
    Wr = nnet[:Wr]
    Wz = nnet[:Wz]
    Wc = nnet[:Wc]
    Ur = nnet[:Ur]
    Uz = nnet[:Uz]
    Uc = nnet[:Uc]
    W_out = nnet[:W_out]
    h0 = nnet[:h0]
    br = nnet[:br]
    bz = nnet[:bz]
    bc = nnet[:bc]
    b_out = nnet[:b_out]

    cost = 0.0
    predictions = Int[]
    x = map(Block, input)
    @grad nnet begin
        r = sigmoid(add(add(linear(Wr, x[1]), linear(Ur, h0)), br))
        z = sigmoid(add(add(linear(Wz, x[1]), linear(Uz, h0)), bz))
        c = tanh(add(add(linear(Wc, x[1]), mult(r, linear(Uc, h0))), bc))
        h = add(mult(z, h0), mult(minus(1.0, z), c))
        scores = affine(W_out, h, b_out)
        cost += nll_categorical(target[1], scores)
        append!(predictions, nnx.argmax(value(scores)))
        for t = 2:length(input)
            r = sigmoid(add(add(linear(Wr, x[t]), linear(Ur, h)), br))
            z = sigmoid(add(add(linear(Wz, x[t]), linear(Uz, h)), bz))
            c = tanh(add(add(linear(Wc, x[t]), mult(r, linear(Uc, h))), bc))
            h = add(mult(z, h0), mult(minus(1.0, z), c))
            scores = affine(W_out, h, b_out)
            cost += nll_categorical(target[t], scores)
            append!(predictions, nnx.argmax(value(scores)))
        end
    end
    return cost, predictions
end

function predict(nnet::NeuralNet, input::Vector)
    Wr = nnet[:Wr]
    Wz = nnet[:Wz]
    Wc = nnet[:Wc]
    Ur = nnet[:Ur]
    Uz = nnet[:Uz]
    Uc = nnet[:Uc]
    W_out = nnet[:W_out]
    h0 = nnet[:h0]
    br = nnet[:br]
    bz = nnet[:bz]
    bc = nnet[:bc]
    b_out = nnet[:b_out]

    predictions = Int[]
    x = map(Block, input)
    r = sigmoid(add(add(linear(Wr, x[1]), linear(Ur, h0)), br))
    z = sigmoid(add(add(linear(Wz, x[1]), linear(Uz, h0)), bz))
    c = tanh(add(add(linear(Wc, x[1]), mult(r, linear(Uc, h0))), bc))
    h = add(mult(z, h0), mult(minus(1.0, z), c))
    scores = affine(W_out, h, b_out)
    append!(predictions, nnx.argmax(value(scores)))
    for t = 2:length(input)
        r = sigmoid(add(add(linear(Wr, x[t]), linear(Ur, h)), br))
        z = sigmoid(add(add(linear(Wz, x[t]), linear(Uz, h)), bz))
        c = tanh(add(add(linear(Wc, x[t]), mult(r, linear(Uc, h))), bc))
        h = add(mult(z, h0), mult(minus(1.0, z), c))
        scores = affine(W_out, h, b_out)
        append!(predictions, nnx.argmax(value(scores)))
    end
    return predictions
end

function xor_sample(T::Int)
    x_bits = rand(0:1, T)
    xs = [nnx.onehot(b + 1, 2) for b in x_bits]
    ys = (cumsum(x_bits) % 2) + 1
    return xs, ys
end

function check_grads()
    n_in = 2
    n_hid = 5
    n_out = 2
    xs, ys = xor_sample(20)
    nnet = build_model(n_in, n_hid, n_out)    
    f() = predict(nnet, xs, ys)[1]
    gradcheck(nnet, f)
end

function fitrnn()
    n_in = 2
    n_hid = 10
    n_out = 2
    n_train = 50
    trX, trY = {}, {}
    minlen, maxlen = typemax(Int), typemin(Int)
    for i = 1:n_train
        T = rand(5:20)
        xs, ys = xor_sample(T)
        minlen = min(minlen, T)
        maxlen = max(maxlen, T)
        push!(trX, xs)
        push!(trY, ys)
    end
    
    nnet = build_model(n_in, n_hid, n_out)
    indices = collect(1:n_train)

    for epoch = 1:50
        errors = 0
        shuffle!(indices)
        for i in indices
            Y_pred = predict(nnet, trX[i], trY[i])[2]
            errors += sum(trY[i] .!= Y_pred)
            backprop(nnet)
            rmsprop!(nnet, 0.01, 0.9, 3)
        end
    end
    println("[train]")
    println("  min seq length => $minlen")
    println("  max seq length => $maxlen")
    println("  total errors => $errors")

    n_test = 20
    errors = 0
    minlen, maxlen = typemax(Int), typemin(Int)
    for i = 1:n_test
        T = rand(100:200)
        xs, ys = xor_sample(T)
        minlen = min(minlen, T)
        maxlen = max(maxlen, T)
        ps = predict(nnet, xs)
        errors += sum(ys .!= ps)
    end
    println("[test]")
    println("  min seq length => $minlen")
    println("  max seq length => $maxlen")
    println("  total errors => $errors")
end

# check_grads()
fitrnn()