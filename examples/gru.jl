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
    @paramdef nnet Wr Wz Wc Ur Uz Uc W_out h0 br bz bc b_out
    cost = 0.0
    predictions = Int[]
    x = map(Block, input)
    @autograd nnet begin
        r = sigmoid(add(linear(Wr, x[1]), linear(Ur, h0), br))
        z = sigmoid(add(linear(Wz, x[1]), linear(Uz, h0), bz))
        c = tanh(add(linear(Wc, x[1]), mult(r, linear(Uc, h0)), bc))
        h = add(mult(z, h0), mult(minus(1.0, z), c))
        scores = affine(W_out, h, b_out)
        cost += nll_categorical(target[1], scores)
        append!(predictions, nnx.argmax(value(scores)))
        for t = 2:length(input)
            r = sigmoid(add(linear(Wr, x[t]), linear(Ur, h), br))
            z = sigmoid(add(linear(Wz, x[t]), linear(Uz, h), bz))
            c = tanh(add(linear(Wc, x[t]), mult(r, linear(Uc, h)), bc))
            h = add(mult(z, h0), mult(minus(1.0, z), c))
            scores = affine(W_out, h, b_out)
            cost += nll_categorical(target[t], scores)
            append!(predictions, nnx.argmax(value(scores)))
        end
    end
    return cost, predictions
end

function predict(nnet::NeuralNet, input::Vector)
    @paramdef nnet Wr Wz Wc Ur Uz Uc W_out h0 br bz bc b_out
    predictions = Int[]
    x = map(Block, input)
    r = sigmoid(add(linear(Wr, x[1]), linear(Ur, h0), br))
    z = sigmoid(add(linear(Wz, x[1]), linear(Uz, h0), bz))
    c = tanh(add(linear(Wc, x[1]), mult(r, linear(Uc, h0)), bc))
    h = add(mult(z, h0), mult(minus(1.0, z), c))
    scores = affine(W_out, h, b_out)
    append!(predictions, nnx.argmax(value(scores)))
    for t = 2:length(input)
        r = sigmoid(add(linear(Wr, x[t]), linear(Ur, h), br))
        z = sigmoid(add(linear(Wz, x[t]), linear(Uz, h), bz))
        c = tanh(add(linear(Wc, x[t]), mult(r, linear(Uc, h)), bc))
        h = add(mult(z, h0), mult(minus(1.0, z), c))
        scores = affine(W_out, h, b_out)
        append!(predictions, nnx.argmax(value(scores)))
    end
    return predictions
end

function check_grads()
    n_in = 2
    n_hid = 5
    n_out = 2
    xs, ys = nnx.randxor(20)
    nnet = build_model(n_in, n_hid, n_out)    
    f() = predict(nnet, xs, ys)[1]
    gradcheck(nnet, f)
end

function fitrnn()
    n_in = 2
    n_hid = 10
    n_out = 2
    n_train = 50
    trX, trY = nnx.randxor(5:20, n_train)
    minlen = minimum(map(length, trX))
    maxlen = maximum(map(length, trX))
    
    nnet = build_model(n_in, n_hid, n_out)
    indices = collect(1:n_train)
    epoch = 0
    max_epochs = 50
    println("fitting...")
    while epoch < max_epochs
        epoch += 1
        errors = 0
        shuffle!(indices)
        for i in indices
            Y_pred = predict(nnet, trX[i], trY[i])[2]
            errors += sum(trY[i] .!= Y_pred)
            backprop(nnet)
            rmsprop!(nnet, 0.01, 0.9, 3)
        end
        if epoch % 5 == 0
            println("epoch => $epoch, errors => $errors")
        end
        errors == 0 ? break : nothing
    end
    println("[train]")
    println("  number of epochs => $epoch")
    println("  min seq length => $minlen")
    println("  max seq length => $maxlen")
    println("  total errors => $errors")

    n_test = 20
    teX, teY = nnx.randxor(100:200, n_test)
    minlen = minimum(map(length, trX))
    maxlen = maximum(map(length, trX))
    errors = 0
    for i = 1:n_test
        Y_pred = predict(nnet, teX[i])
        errors += sum(teY[i] .!= Y_pred)
    end
    println("[test]")
    println("  min seq length => $minlen")
    println("  max seq length => $maxlen")
    println("  total errors => $errors")
end

check_grads()
fitrnn()