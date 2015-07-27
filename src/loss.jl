
function nll_normal{F<:FloatingPoint}(target::Array{F}, output::Block)
    @assert size(target, 1) == size(output, 1)
    @assert size(target, 2) == size(output, 2)

    output.dx += output.x .- target
    0.5 * sum(output.dx .* output.dx)
end

function nll_normal(target::FloatingPoint, output::Block)
    @assert size(output) == (1,1)
    output.dx += output.x .- target
    0.5 * output.dx[1] .* output.dx[1]
end

function nll_categorical{I<:Integer}(target::Vector{I}, output::Block)
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

function nll_categorical(target::Integer, output::Block)
    @assert size(output, 2) == 1

    s = softmax(output)
    output.dx += s.x
    output.dx[target] -= 1
    -log(s.x[target])
end

