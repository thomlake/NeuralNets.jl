using NeuralNets

function randadd(r::UnitRange{Int})
    steps = rand(r)
    i1 = rand(1:steps)
    i2 = i1
    while i1 == i2
        i2 = rand(1:steps)
    end
    n1 = rand(1:10)
    n2 = rand(1:10)
    output = float(n1 + n2)
    input = Vector{Float64}[]

    for i = 1:steps
        x = if i == i1
            [n1, 1]
        elseif i == i2
            [n2, 1]
        else
            [rand(1:10), 0]
        end
        push!(input, x)
    end

    return input, output
end

function dataset(r::UnitRange{Int}, n_samples::Int)
    x, y = randadd(r)
    X, Y = typeof(x)[x], typeof(y)[y]
    for i = 2:n_samples
        x, y = randadd(r)
        push!(X, x)
        push!(Y, y)
    end
    X, Y
end

function build_model(n_in::Int, n_hid::Int, n_out::Int)
    nnet = NeuralNet()
    nnet[:w] = Orthonormal(sqrt(2), n_hid, n_in)
    nnet[:u] = Orthonormal(2, 1, n_hid)
    nnet[:v] = Orthonormal(1, n_out, n_hid)
    nnet[:bh] = Zeros(n_hid)
    nnet[:bs] = Zeros(1)
    nnet[:bo] = Zeros(n_out)
    println(typeof(nnet))
    return nnet
end

function predict(nnet::NeuralNet, input::Vector)
    @paramdef nnet w u v bh bs bo
    n = length(input)
    x = map(Block, input)
    h, s = Array(Block, n), Array(Block, n)
    for i = 1:n
        h[i] = relu(affine(w, x[i], bh))
        s[i] = affine(u, h[i], bs)
    end
    a = decat(softmax(concat(s)))
    z = add(map(i->mult(a[i], h[i]), 1:n)...)
    prediction = affine(v, z, bo)
    return value(prediction)[1], map(i->value(a[i])[1], 1:n)
end

function predict(nnet::NeuralNet, input::Vector, target::FloatingPoint)
    @paramdef nnet w u v bh bs bo
    n = length(input)
    x = map(Block, input)
    h, s = Array(Block, n), Array(Block, n)
    @autograd nnet begin
        for i = 1:n
            h[i] = relu(affine(w, x[i], bh))
            s[i] = affine(u, h[i], bs)
        end
        a = decat(softmax(concat(s)))
        z = add(map(i->mult(a[i], h[i]), 1:n)...)
        prediction = affine(v, z, bo)
        cost = nll_normal(target, prediction)
    end
    return cost
end

function testgrads()
    n_in = 2
    n_out = 1
    x, y = randadd(5:5)
    nnet = build_model(n_in, 7, n_out)
    f() = predict(nnet, x, y)
    gradcheck(nnet, f)
end

function main()
    n_train = 100
    n_in = 2
    n_hid = 10
    n_out = 1
    nnet = build_model(n_in, n_hid, n_out)
    trX, trY = dataset(5:20, n_train)
    indices = collect(1:n_train)

    for epoch = 1:200
        shuffle!(indices)
        cost = 0.0
        for i in indices
            cost += predict(nnet, trX[i], trY[i])
            backprop(nnet)
            rmsprop!(nnet, 0.005 / length(trX[i]), 0.9, 15)
        end
        epoch % 50 == 0 && println("epoch => $epoch, cost => $cost")
    end

    n_test = 20
    error = 0.0
    for i = 1:n_test
        x, y_true = randadd(20:50)
        y_pred, a = predict(nnet, x)
        error += (y_true - y_pred)^2
        for i = 1:length(x)
            inp = "[$(lpad(int(x[i][1]), 2)) $(int(x[i][2]))]"
            atn = "$(repeat("-", iround(10 * a[i])))"
            println("$inp $atn")
        end
        println("$(iround(y_true)) : $(iround(y_pred))")
        println()
    end
    println("error => $error, avg error $(error / n_test)")
end

testgrads()
main()
