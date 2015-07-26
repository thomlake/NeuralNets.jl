using NeuralNets
const nnx = NeuralNets.Extras

function build_model(n_in::Int, n_hid::Int, n_out::Int)
    nnet = NeuralNet()
    nnet[:W] = Orthonormal(1.2, n_hid, n_in)
    nnet[:U] = Orthonormal(1.2, n_hid, n_hid)
    nnet[:V] = Orthonormal(1.2, n_out, n_hid)
    nnet[:h0] = Zeros(n_hid)
    nnet[:b1] = Zeros(n_hid)
    nnet[:b2] = Zeros(n_out)
    return nnet
end

function predict(nnet::NeuralNet, input::Vector, target::Vector{Int}, h::Block=nnet[:h0], t::Int=1)
    @paramdef nnet W U V b1 b2
    c = 0.0
    x = Block(input[t])
    @grad nnet begin
        h = tanh(add(linear(W, x), linear(U, h), b1))
        scores = affine(V, h, b2)
        c = nll_categorical(target[t], scores)
    end
    p = nnx.argmax(value(scores))
    if t == length(input)
        return (c, [p])
    else
        c_rest, p_rest = predict(nnet, input, target, h, t + 1)
        return (c + c_rest, [p, p_rest])
    end
end

function predict(nnet::NeuralNet, input::Vector, h::Block=nnet[:h0], t::Int=1)
    @paramdef nnet W U V b1 b2
    x = Block(input[t])
    h = tanh(add(linear(W, x), linear(U, h), b1))
    scores = affine(V, h, b2)
    p = nnx.argmax(value(scores))
    if t == length(input)
        return [p]
    else
        return [p, predict(nnet, input, h, t + 1)]
    end
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
    n_hid = 5
    n_out = 2
    n_train = 50
    trX, trY = {}, {}
    minlen, maxlen = typemax(Int), typemin(Int)
    for i = 1:n_train
        T = rand(5:20)
        xs, ys = nnx.randxor(T)
        minlen = min(minlen, T)
        maxlen = max(maxlen, T)
        push!(trX, xs)
        push!(trY, ys)
    end
    
    nnet = build_model(n_in, n_hid, n_out)
    indices = collect(1:n_train)
    epoch = 0
    max_epochs = 100
    println("fitting...")
    while epoch < max_epochs
        epoch += 1
        errors = 0
        shuffle!(indices)
        for i in indices
            Y_pred = predict(nnet, trX[i], trY[i])[2]
            errors += sum(trY[i] .!= Y_pred)
            backprop(nnet)
            rmsprop!(nnet, 0.01, 0.9, 1)
        end
        if epoch % 10 == 0
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
    errors = 0
    minlen, maxlen = typemax(Int), typemin(Int)
    for i = 1:n_test
        T = rand(100:200)
        xs, ys = nnx.randxor(T)
        minlen = min(minlen, T)
        maxlen = max(maxlen, T)
        ps = predict(nnet, xs)
        errors += sum(ys .!= ps)
    end
    println("[test]")
    println("  min seq length => $minlen")
    println("  max seq length => $maxlen")
    println("  total errors => $errors")
    println("  average errors per sequence => $(errors / n_test)")
end

check_grads()
fitrnn()