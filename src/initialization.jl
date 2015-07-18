
#-- Normal --#
function init_normal(n_out::Int, n_in::Int)
    Block(randn(n_out, n_in))
end

function init_normal(sigma::Real, n_out::Int, n_in::Int)
    Block(sigma * randn(n_out, n_in))
end

#-- Uniform --#
function init_uniform(lower::Real, upper::Real, n_out::Int, n_in::Int)
    Block((upper - lower) .* rand(n_out, n_in) .+ lower)
end

#-- Glorot --#
function init_glorot(n_out::Int, n_in::Int)
    g = sqrt(6.0) / sqrt(n_out + n_in)
    init_uniform(-g, g, n_out, n_in)
end

#-- Orthonormal --#
function init_orthonormal(g::Real, n_out::Int, n_in::Int)
    n = max(n_out, n_in)
    W = g .* svd(randn(n, n))[1]
    Block(W[1:n_out,1:n_in])
end

function init_orthonormal(n_out::Int, n_in::Int)
    n = max(n_out, n_in)
    W = svd(randn(n, n))[1]
    Block(W[1:n_out,1:n_in])
end

#-- Sparse -- #
function init_sparse(n::Int, n_out::Int, n_in::Int)
    n_zero = n_in - min(n, n_in)
    W = randn(n_out, n_in)
    for i = 1:n_out
        jz = collect(1:n_in)
        shuffle!(jz)
        for j = 1:n_zero
            W[i,jz[j]] = 0
        end
    end
    Block(W)
end

param_jump_table = [
    :Normal => init_normal,
    :Uniform => init_uniform,
    :Glorot => init_glorot,
    :Orthonormal => init_orthonormal,
    :Sparse => init_sparse,
]

function register_param_initializer!(symbol::Symbol, f::Function)
    param_jump_table[symbol] = f
end

function Param(symbol::Symbol, args...)
    param_jump_table[symbol](args...)
end

Param(args...) = Block(zeros(args...))
