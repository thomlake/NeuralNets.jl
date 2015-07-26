
nnet_ops = Symbol[
    :tanh,
    :sigmoid,
    :relu,
    :softmax,
    :maxout,
    :mult,
    :linear,
    :affine,
    :add,
    :minus,
    :concat
]

function grad_parser(nnet::Symbol, expr::Expr)
    if expr.head == :call && expr.args[1] in nnet_ops
        insert!(expr.args, 2, nnet)
    end
    for arg in expr.args
        try
            grad_parser(nnet, arg)
        end
    end
    return expr
end

macro grad(nnet::Symbol, expr::Expr)
    grad_parser(nnet, esc(expr))
end

macro paramdef(coll::Symbol, S::Symbol...)
    r = Expr(:block)
    for s in S
        if typeof(s) <: Symbol
            v = s
            k = Expr(:quote,s)
            push!(r.args, Expr(:(=), v, :($coll[$k])))
        elseif s.head == :tuple
            v = symbol(join(s.args, "_"))
            k = tuple(s.args[1], s.args[2:end]...)
            push!(r.args, Expr(:(=), v, :($coll[$k])))
        else
            error("unable to define variable from: $(s)")
        end
    end
    return esc(r)
end

