type Block
    x::Matrix
    dx::Matrix
end

Block(n_rows::Int, n_cols::Int) = Block(zeros(n_rows, n_cols), zeros(n_rows, n_cols))

Block(n_rows::Int) = Block(zeros(n_rows, 1), zeros(n_rows, 1))

Block{F<:FloatingPoint}(X::Matrix{F}) = Block(X, zero(X))

Block{F<:FloatingPoint}(x::Vector{F}) = Block(Extras.vec2mat(x), zeros(length(x), 1))

Base.zero(b::Block) = Block(zero(b.x), zero(b.dx))

Base.size(b::Block) = size(b.x)

Base.size(b::Block, i::Int) = size(b.x, i)

value(b::Block) = b.x

type NeuralNet{T}
    params::Dict{T,Block}
    bpstack::Vector{Function}
    metadata::Dict{Any,Any}
end

NeuralNet{T}(params::Dict{T,Block}) = NeuralNet(params, Function[], Dict())

NeuralNet() = NeuralNet(Dict{Any,Block}())

NeuralNet(T::Type) = NeuralNet(Dict{T,Block}())

function NeuralNet(a::Array{Block})
    params = Dict{Symbol,Block}()
    for i = 1:length(a)
        params[symbol("anonymous_param_$i")] = a[i]
    end
    return NeuralNet(params)
end

Base.getindex(nnet::NeuralNet, name) = nnet.params[name]

Base.setindex!(nnet::NeuralNet, block::Block, name) = nnet.params[name] = block

function backprop(nnet::NeuralNet)
    for i = length(nnet.bpstack):-1:1
        nnet.bpstack[i]()
    end
    empty!(nnet.bpstack)
end
