
vec2mat(b::Vector) = reshape(b, (size(b, 1), 1))
onehot(i::Int, d::Int) = (x = zeros(d); x[i] = 1; x)

type Graph
    dobackprop::Bool
    backward::Vector{Function}
end

Graph() = Graph(true, Function[])

function backprop(G::Graph)
    for i = length(G.backward):-1:1
        G.backward[i]()
    end
    empty!(G.backward)
end

# abstract Block{F<:FloatingPoint}

type Block{F<:FloatingPoint}
    x::Matrix{F}
    dx::Matrix{F}
end

Block(n_rows::Int, n_cols::Int) = Block(zeros(n_rows, n_cols), zeros(n_rows, n_cols))
Block(n_rows::Int) = Block(zeros(n_rows, 1), zeros(n_rows, 1))
Block{F<:FloatingPoint}(X::Matrix{F}) = Block(X, zero(X))
Block{F<:FloatingPoint}(x::Vector{F}) = Block(vec2mat(x), zeros(length(x), 1))

Base.zero(b::Block) = Block(zero(b.x), zero(b.dx))

Base.size(b::Block) = size(b.x)
Base.size(b::Block, i::Int) = size(b.x, i)

input = Block

type NeuralNet
    G::Graph
    params::Array{Block}
    feedforward::Function
end

function backprop(nnet::NeuralNet)
    G = nnet.G
    for i = length(G.backward):-1:1
        G.backward[i]()
    end
    empty!(G.backward)
end

function dropout(block::Block, p::FloatingPoint)
    for j = 1:size(block, 2)
        for i = 1:size(block, 1)
            if rand() < p
                block.x[i,j] = 0
            end
        end
    end
    block
end

function dropout(G::Graph, block::Block, p::FloatingPoint)
    dropout(block, p)
end

function bwd_tanh(outblock::Block, inblock::Block)
    inblock.dx += (1 .- (outblock.x .* outblock.x)) .* outblock.dx
end

function Base.tanh(G::Graph, inblock::Block)
    outblock = Block(tanh(inblock.x))
    if G.dobackprop
        push!(G.backward, () -> bwd_tanh(outblock, inblock))
    end
    outblock
end

function bwd_sigmoid(outblock::Block, inblock::Block)
    inblock.dx += outblock.x .* (1 - outblock.x) .* outblock.dx
end

function sigmoid(G::Graph, inblock::Block)
    outblock = Block(1 ./ (1 .+ exp(-inblock.x)))
    if G.dobackprop
        push!(G.backward, () -> bwd_sigmoid(outblock, inblock))
    end
    outblock
end

function bwd_relu(outblock::Block, inblock::Block)
    inblock.dx += outblock.dx .* (outblock.x .> 0)
end

function relu(G::Graph, inblock::Block)
    outblock = Block(max(inblock.x, 0))
    if G.dobackprop
        push!(G.backward, () -> bwd_relu(outblock, inblock))
    end
    outblock
end

function bwd_softmax(outblock::Block, inblock::Block)
    for n = 1:size(outblock, 2)
        for i = 1:size(outblock, 1)
            for j = 1:size(outblock, 1)
                if i == j
                    inblock.dx[i,n] += outblock.x[i,n] * (1 - outblock.x[j,n]) * outblock.dx[j,n]
                else
                    inblock.dx[i,n] -= outblock.x[i,n] * outblock.x[j,n] * outblock.dx[j,n]
                end
            end
        end
    end
end

function softmax(G::Graph, inblock::Block)
    a = exp(inblock.x .- maximum(inblock.x, 1))
    outblock = Block(a ./ sum(a, 1))
    if G.dobackprop
        push!(G.backward, () -> bwd_softmax(outblock, inblock))
    end
    outblock
end

function softmax(inblock::Block)
    a = exp(inblock.x .- maximum(inblock.x, 1))
    Block(a ./ sum(a, 1))
end

function bwd_maxout(outblock::Block, inblock::Block)
    for j = 1:size(inblock, 2)
        i =indmax(inblock.x[:,j])
        inblock.dx[i,j] += outblock.dx[i,j]
    end
end

function maxout(G::Graph, inblock::Block)
    outblock = zero(inblock)
    for j = 1:size(inblock, 2)
        i = indmax(inblock.x[:,j])
        outblock.x[i,j] = inblock.x[i,j]
    end
    if G.dobackprop
        push!(G.backward, () -> bwd_maxout(outblock, inblock))
    end
    outblock
end

function bwd_mult(outblock::Block, inblock1::Block, inblock2::Block)
    inblock1.dx += outblock.dx .* inblock2.x
    inblock2.dx += outblock.dx .* inblock1.x
end

function mult(G::Graph, inblock1::Block, inblock2::Block)
    outblock = Block(inblock1.x .* inblock2.x)
    if G.dobackprop
        push!(G.backward, () -> bwd_mult(outblock, inblock1, inblock2))
    end
    outblock
end

function mult(G::Graph, a::FloatingPoint, inblock::Block)
    outblock = Block(a .* inblock.x)
    if G.dobackprop
        push!(G.backward, () -> inblock.dx += c .* outblock.dx)
    end
    outblock
end

function mult(a::FloatingPoint, inblock::Block)
    Block(a .* inblock.x)
end

function bwd_linear(params::Block, outblock::Block, inblock::Block)
    inblock.dx .+= At_mul_B(params.x, outblock.dx)
    params.dx .+= A_mul_Bt(outblock.dx, inblock.x)
end

function linear(G::Graph, params::Block, inblock::Block)
    @assert size(params, 2) == size(inblock, 1)
    outblock = Block(size(params, 1), size(inblock, 2))
    A_mul_B!(outblock.x, params.x, inblock.x)
    if G.dobackprop
        push!(G.backward, () -> bwd_linear(params, outblock, inblock))
    end
    outblock
end

function add_mat_mat(G::Graph, inblock1::Block, inblock2::Block)
    @assert size(inblock1, 1) == size(inblock2, 1)
    @assert size(inblock1, 2) == size(inblock2, 2)

    outblock = zero(inblock1)
    copy!(outblock.x, inblock1.x)
    outblock.x .+= inblock2.x
    if G.dobackprop
        push!(G.backward, () -> inblock1.dx .+= outblock.dx)
        push!(G.backward, () -> inblock2.dx .+= outblock.dx)
    end
    outblock
end

function add_mat_vec(G::Graph, inblock1::Block, inblock2::Block)
    @assert size(inblock1, 1) == size(inblock2, 1)
    @assert size(inblock2, 2) == 1

    outblock = Block(copy(inblock1.x))
    outblock.x .+= inblock2.x
    if G.dobackprop
        push!(G.backward, () -> inblock1.dx .+= outblock.dx)
        push!(G.backward, () -> inblock2.dx .+= sum(outblock.dx, 2))
    end
    outblock
end

function add(G::Graph, inblock1::Block, inblock2::Block)
    if size(inblock1, 2) > 1 && size(inblock2, 2) == 1
        add_mat_vec(G, inblock1, inblock2)
    else
        add_mat_mat(G, inblock1, inblock2)
    end
end

function minus_mat_mat(G::Graph, inblock1::Block, inblock2::Block)
    @assert size(inblock1, 1) == size(inblock2, 1)
    @assert size(inblock1, 2) == size(inblock2, 2)

    outblock = Block(copy(inblock1.x))
    outblock.x .-= inblock2.x
    if G.dobackprop
        push!(G.backward, () -> inblock1.dx -= outblock.dx)
        push!(G.backward, () -> inblock2.dx -= outblock.dx)
    end
    outblock
end

function minus_mat_vec(G::Graph, inblock1::Block, inblock2::Block)
    @assert size(inblock1, 1) == size(inblock2, 1)
    @assert size(inblock2, 2) == 1

    outblock = Block(copy(inblock1.x))
    outblock.x .-= inblock2.x
    if G.dobackprop
        push!(G.backward, () -> inblock1.dx -= outblock.dx)
        push!(G.backward, () -> inblock2.dx -= sum(outblock.dx, 2))
    end
    outblock
end

function minus(G::Graph, inblock1::Block, inblock2::Block)
    if size(inblock1, 2) > 1 && size(inblock2, 2) == 1
        add_mat_vec(G, inblock1, inblock2)
    else
        add_mat_mat(G, inblock1, inblock2)
    end
end

function minus(G::Graph, a::FloatingPoint, inblock::Block)
    outblock = Block(a .- inblock.x)
    if G.dobackprop
        push!(G.backward, () -> inblock.dx -= outblock.dx)
    end
    outblock
end

function bwd_concat(outblock::Block, inblocks::Vector{Block})
    i = 1
    for b = 1:length(inblocks)
        j = size(inblocks[b], 1)
        inblocks[b].dx .+= outblock.dx[i:i + j - 1,:]
        i += j
    end
end

function concat(G::Graph, inblocks::Vector{Block})
    first_size = size(inblocks[1], 2)
    for b = 2:length(inblocks)
        @assert first_size == size(inblocks[b], 2)
    end
    outblock = Block(vcat(map(inblock->inblock.x, inblocks)...))
    if G.dobackprop
        push!(G.backward, () -> bwd_concat(outblock, inblocks))
    end
    outblock
end

function mseloss{F<:FloatingPoint}(target::Array{F}, output::Block)
    output.dx += output.x .- target
    0.5 * sum(output.dx .* output.dx)
end

function catloss(target::Vector{Int}, output::Block)
    @assert size(output, 2) == length(target)

    s = softmax(output)
    output.dx .+= s.x
    for i = 1:size(output, 2)
        output.dx[target[i],i] -= 1
    end
    nll = 0.0
    for i = 1:length(target)
        nll -= log(s.x[target[i],i])
    end
    nll
end

function catloss(target::Int, output::Block)
    @assert size(output, 2) == 1

    s = softmax(output)
    output.dx += s.x
    output.dx[target] -= 1
    -log(s.x[target])
end

