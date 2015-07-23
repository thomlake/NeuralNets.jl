type Block
    x::Matrix
    dx::Matrix
end

Block(n_rows::Int, n_cols::Int) = Block(zeros(n_rows, n_cols), zeros(n_rows, n_cols))

Block(n_rows::Int) = Block(zeros(n_rows, 1), zeros(n_rows, 1))

Block{F<:FloatingPoint}(X::Matrix{F}) = Block(X, zero(X))

Block{F<:FloatingPoint}(x::Vector{F}) = Block(vec2mat(x), zeros(length(x), 1))

Base.zero(b::Block) = Block(zero(b.x), zero(b.dx))

Base.size(b::Block) = size(b.x)

Base.size(b::Block, i::Int) = size(b.x, i)

value(b::Block) = b.x

type NeuralNet
    params::Dict{Symbol,Block}
    bpstack::Vector{Function}
end

NeuralNet() = NeuralNet(Dict{Symbol,Block}(), Function[])

NeuralNet(params::Dict{Symbol,Block}) = NeuralNet(params, Function[])

function NeuralNet(a::Array{Block})
    params = Dict{Symbol,Block}()
    for i = 1:length(a)
        params[symbol("anonymous_param_$i")] = a[i]
    end
    return NeuralNet(params)
end

Base.getindex(nnet::NeuralNet, name::Symbol) = nnet.params[name]

Base.setindex!(nnet::NeuralNet, block::Block, name::Symbol) = nnet.params[name] = block

function backprop(nnet::NeuralNet)
    for i = length(nnet.bpstack):-1:1
        nnet.bpstack[i]()
    end
    empty!(nnet.bpstack)
end
