using NeuralNets

# function randadd(r::UnitRange{Int})
#     steps = rand(r)
#     t1 = rand(1:ifloor(steps / 3))
#     t2 = rand(steps - iceil(steps / 3):steps)
#     n1 = rand(1:10)
#     n2 = rand(1:10)
#     input = zeros(2, steps)
#     input[1,:] = rand(1:10, steps)
#     input[1,t1] = n1
#     input[2,t1] = 1
#     input[1,t2] = n2
#     input[2,t2] = 1
#     output = zeros(steps)
#     output[t2:end] = n1 + n2
#     return Vector{Float64}[input[:,t] for t = 1:steps], Vector{Float64}[[output[t]] for t = 1:steps]
# end


function randadd(r::UnitRange{Int})
    steps = rand(r)
    i1 = rand(1:steps)
    i2 = i1
    while i1 == i2
        i2 = rand(1:steps)
    end
    n1 = rand(1:10)
    n2 = rand(1:10)
    output = [float(n1 + n2)]
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
    X, Y = Vector{eltype(x)}[x], Vector{eltype(y)}[y]
    for i = 2:n_samples
        x, y = randadd(r)
        push!(X, x)
        push!(Y, y)
    end
    X, Y
end


function build_model(n_in::Int, n_hid::Int, n_out::Int)
    nnet = NeuralNet()
    nnet[:W] = Orthonormal(sqrt(2), n_hid, n_in)
    nnet[:U] = Orthonormal(1, 1, n_hid)
    nnet[:V] = Orthonormal(1, n_out, n_hid)
    nnet[:b1] = Zeros(n_hid)
    nnet[:b2] = Zeros(1)
    nnet[:b3] = Zeros(n_out)
    return nnet
end

function predict(nnet::NeuralNet, input::Vector)
    @paramdef nnet W U V b1 b2 b3
    len = length(input)
    x = map(Block, input)
    H = Block[]
    S = Block[]
    for i = 1:len
        h = relu(affine(W, x[i], b1))
        s = affine(U, h, b2)
        push!(H, h)
        push!(S, s)
    end
    A = decat(softmax(concat(S)))
    z = add([mult(A[i], H[i]) for i = 1:len]...)
    prediction = affine(V, z, b3)
    return value(prediction)
end            

function predict(nnet::NeuralNet, input::Vector, target)
    @paramdef nnet W U V b1 b2 b3
    len = length(input)
    x = map(Block, input)
    H = Block[]
    S = Block[]
    @autograd nnet begin
        for i = 1:len
            h = relu(affine(W, x[i], b1))
            s = affine(U, h, b2)
            push!(H, h)
            push!(S, s)
        end
        A = decat(softmax(concat(S)))
        z = add([mult(A[i], H[i]) for i = 1:len]...)
        prediction = affine(V, z, b3)
        cost = nll_normal(target, prediction)
    end
    return cost, value(prediction)
end          

function check_grads()
    n_in = 2
    n_out = 2
    x, y = randxor(5:5)
    nnet = build_model(n_in, 7, n_out)
    f() = predict(nnet, x, y)[1]
    gradcheck(nnet, f, tol=0.1)
end

function main()
    n_train = 100
    n_in = 2
    n_hid = 10
    n_out = 1
    nnet = build_model(n_in, n_hid, n_out)
    trX, trY = dataset(5:15, n_train)
    indices = collect(1:n_train)

    for epoch = 1:1000
        shuffle!(indices)
        cost = 0.0
        for i in indices
            c, y_pred = predict(nnet, trX[i], trY[i])
            cost += c
            backprop(nnet)
            rmsprop!(nnet, 0.001 / length(trX[i]), 0.9, 5)
        end
        epoch % 50 == 0 && println("epoch => $epoch, cost => $cost")
    end

    n_test = 20
    error = 0
    teX, teY = dataset(5:15, n_test)
    for i = 1:n_test
        y_pred = predict(nnet, teX[i])
        error += sumabs2(teY[i] - y_pred)
        println((teY[i], y_pred))
    end
    println("error => $error, avg error $(error / n_test)")

end

# check_grads()
main()