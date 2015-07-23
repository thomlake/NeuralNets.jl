# -- Dropout -- #
function dropout(block::Block, p::FloatingPoint)
    for j = 1:size(block, 2)
        for i = 1:size(block, 1)
            if rand() < p
                block.x[i,j] = 0
            end
        end
    end
    return block
end

# -- Tanh -- #
Base.tanh(inblock::Block) = Block(tanh(inblock.x))

function bwd_tanh(outblock::Block, inblock::Block)
    inblock.dx += (1 .- (outblock.x .* outblock.x)) .* outblock.dx
end

function Base.tanh(nnet::NeuralNet, inblock::Block)
    outblock = tanh(inblock)
    push!(nnet.bpstack, () -> bwd_tanh(outblock, inblock))
    outblock
end

# -- Sigmoid -- #
sigmoid(inblock::Block) = Block(1 ./ (1 .+ exp(-inblock.x)))

function bwd_sigmoid(outblock::Block, inblock::Block)
    inblock.dx += outblock.x .* (1 - outblock.x) .* outblock.dx
end

function sigmoid(nnet::NeuralNet, inblock::Block)
    outblock = sigmoid(inblock)
    push!(nnet.bpstack, () -> bwd_sigmoid(outblock, inblock))
    outblock
end

# -- ReLU -- #
relu(inblock::Block) = Block(max(inblock.x, 0))

function bwd_relu(outblock::Block, inblock::Block)
    inblock.dx += outblock.dx .* (outblock.x .> 0)
end

function relu(nnet::NeuralNet, inblock::Block)
    outblock = relu(inblock)
    push!(nnet.bpstack, () -> bwd_relu(outblock, inblock))
    outblock
end

# -- Softmax -- #
function softmax(inblock::Block)
    a = exp(inblock.x .- maximum(inblock.x, 1))
    a ./= sum(a, 1)
    Block(a)
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

function softmax(nnet::NeuralNet, inblock::Block)
    outblock = softmax(inblock)
    push!(nnet.bpstack, () -> bwd_softmax(outblock, inblock))
    outblock
end

# -- Maxout -- #
function maxout(inblock::Block)
    outblock = zero(inblock)
    for j = 1:size(inblock, 2)
        i = indmax(inblock.x[:,j])
        outblock.x[i,j] = inblock.x[i,j]
    end
    outblock
end

function bwd_maxout(outblock::Block, inblock::Block)
    for j = 1:size(inblock, 2)
        i =indmax(inblock.x[:,j])
        inblock.dx[i,j] += outblock.dx[i,j]
    end
end

function maxout(nnet::NeuralNet, inblock::Block)
    outblock = maxout(inblock)
    push!(nnet.bpstack, () -> bwd_maxout(outblock, inblock))
    outblock
end

# -- Multiply (scalar by block) -- #
mult(a::FloatingPoint, inblock::Block) = Block(a .* inblock.x)

function bwd_mult(outblock::Block, a::FloatingPoint, inblock::Block)
    inblock.dx += a .* outblock.dx
end

function mult(nnet::NeuralNet, a::FloatingPoint, inblock::Block)
    outblock = mult(a, inblock)
    push!(nnet.bpstack, () -> bwd_mult(a, inblock))
    outblock
end

# -- Multiply (block by block) -- #
mult(inblock1::Block, inblock2::Block) = Block(inblock1.x .* inblock2.x)

function bwd_mult(outblock::Block, inblock1::Block, inblock2::Block)
    inblock1.dx += outblock.dx .* inblock2.x
    inblock2.dx += outblock.dx .* inblock1.x
end

function mult(nnet::NeuralNet, inblock1::Block, inblock2::Block)
    outblock = mult(inblock1, inblock2)
    push!(nnet.bpstack, () -> bwd_mult(outblock, inblock1, inblock2))
    outblock
end

# -- Linear -- #
function linear(w::Block, inblock::Block)
    @assert size(w, 2) == size(inblock, 1)

    outblock = Block(size(w, 1), size(inblock, 2))
    A_mul_B!(outblock.x, w.x, inblock.x)
    outblock
end

function bwd_linear(w::Block, outblock::Block, inblock::Block)
    inblock.dx .+= At_mul_B(w.x, outblock.dx)
    w.dx .+= A_mul_Bt(outblock.dx, inblock.x)
end

function linear(nnet::NeuralNet, w::Block, inblock::Block)
    outblock = linear(w, inblock)
    push!(nnet.bpstack, () -> bwd_linear(w, outblock, inblock))
    outblock
end

# -- Affine (for convenience) -- #
affine(w::Block, x::Block, b::Block) = add(linear(w, x), b)

affine(nnet::NeuralNet, w::Block, x::Block, b::Block) = add(nnet, linear(nnet, w, x), b)

# -- Add (2d to 2d) -- #
function add_mat_mat(inblock1::Block, inblock2::Block)
    @assert size(inblock1) == size(inblock2)

    outblock = copy(inblock.x)
    outblock.x .+= inblock2.x
    outblock
end

function bwd_add_mat_mat(outblock::Block, inblock1::Block, inblock2::Block)
    inblock1.dx .+= outblock.dx
    inblock2.dx .+= outblock.dx
end

function add_mat_mat(nnet::NeuralNet, inblock1::Block, inblock2::Block)
    outblock = add_mat_mat(inblock1, inblock2)
    push!(nnet.bpstack, () -> bwd_add_mat_mat(outblock, inblock1, inblock2))
    outblock
end

# -- Add (2d to 1d) -- #
function add_mat_vec(inblock1::Block, inblock2::Block)
    @assert size(inblock1, 1) == size(inblock2, 1)
    @assert size(inblock2, 2) == 1

    outblock = Block(copy(inblock1.x))
    outblock.x .+= inblock2.x
    outblock
end

function bwd_add_mat_vec(outblock::Block, inblock1::Block, inblock2::Block)
    inblock1.dx .+= outblock.dx
    inblock2.dx .+= sum(outblock.dx, 2)
end

function add_mat_vec(nnet::NeuralNet, inblock1::Block, inblock2::Block)
    outblock = add_mat_vec(inblock1, inblock2)
    push!(nnet.bpstack, () -> bwd_add_mat_vec(outblock, inblock1, inblock2))
    outblock
end

# -- Add (block dispatch) -- #
function add(inblock1::Block, inblock2::Block)
    if size(inblock1, 2) > 1 && size(inblock2, 2) == 1
        add_mat_vec(inblock1, inblock2)
    else
        add_mat_mat(inblock1, inblock2)
    end
end

function add(nnet::NeuralNet, inblock1::Block, inblock2::Block)
    if size(inblock1, 2) > 1 && size(inblock2, 2) == 1
        add_mat_vec(nnet, inblock1, inblock2)
    else
        add_mat_mat(nnet, inblock1, inblock2)
    end
end

# -- Minus (2d by 2d) -- #
function minus_mat_mat(inblock1::Block, inblock2::Block)
    @assert size(inblock1) == size(inblock2)

    outblock = Block(copy(inblock1.x))
    outblock.x .-= inblock2.x
    outblock
end

function bwd_minus_mat_mat(outblock::Block, inblock1::Block, inblock2::Block)
    inblock1.dx -= outblock.dx
    inblock2.dx -= outblock.dx
end

function minus_mat_mat(nnet::NeuralNet, inblock1::Block, inblock2::Block)
    outblock = minus_mat_mat(inblock1, inblock2)
    push!(nnet.bpstack, () -> bwd_minus_mat_mat(outblock, inblock1, inblock2))
    outblock
end

# -- Minus (2d by 1d) -- #
function minus_mat_vec(inblock1::Block, inblock2::Block)
    @assert size(inblock1, 1) == size(inblock2, 1)
    @assert size(inblock2, 2) == 1

    outblock = Block(copy(inblock1.x))
    outblock.x .-= inblock2.x
    outblock
end

function bwd_minus_mat_vec(outblock::Block, inblock1::Block, inblock2::Block)
    inblock1.dx -= outblock.dx
    inblock2.dx -= sum(outblock.dx, 2)
end

function minus_mat_vec(nnet::NeuralNet, inblock1::Block, inblock2::Block)
    outblock = minus_mat_vec(inblock1, inblock2)
    push!(nnet.bpstack, () -> bwd_minus_mat_mat(outblock, inblock1, inblock2))
    outblock
end

# -- Minus (block dispatch) --#
function minus(inblock1::Block, inblock2::Block)
    if size(inblock1, 2) > 1 && size(inblock2, 2) == 1
        add_mat_vec(inblock1, inblock2)
    else
        add_mat_mat(inblock1, inblock2)
    end
end

function minus(nnet::NeuralNet, inblock1::Block, inblock2::Block)
    if size(inblock1, 2) > 1 && size(inblock2, 2) == 1
        add_mat_vec(nnet, inblock1, inblock2)
    else
        add_mat_mat(nnet, inblock1, inblock2)
    end
end

# -- Minus (scalar from block) -- #
minus(a::FloatingPoint, inblock::Block) = Block(a .- inblock.x)

function bwd_minus(outblock::Block, inblock::Block)
    inblock.dx -= outblock.dx
end

function minus(nnet::NeuralNet, a::FloatingPoint, inblock::Block)
    outblock = minus(a, inblock)
    push!(nnet.bpstack, () -> bwd_minus(outblock, inblock))
    outblock
end

# -- Concat --#
function concat(inblocks::Vector{Block})
    first_size = size(inblocks[1], 2)
    for b = 2:length(inblocks)
        @assert first_size == size(inblocks[b], 2)
    end
    Block(vcat(map(inblock->inblock.x, inblocks)...))
end

function bwd_concat(outblock::Block, inblocks::Vector{Block})
    i = 1
    for b = 1:length(inblocks)
        j = size(inblocks[b], 1)
        inblocks[b].dx .+= outblock.dx[i:i + j - 1,:]
        i += j
    end
end

function concat(nnet::NeuralNet, inblocks::Vector{Block})
    outblock = concat(inblocks)
    push!(nnet.bpstack, () -> bwd_concat(outblock, inblocks))
    outblock
end