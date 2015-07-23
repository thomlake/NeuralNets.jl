
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
# function grad_parser(nnet::Symbol, expr::Expr)
#     if expr.head == :call && expr.args[1] in nnet_ops    
#         insert!(expr.args, 2, nnet)
#     end
#     for arg in expr.args
#         try
#             grad_parser(nnet, arg)
#         end
#     end
#     expr
# end

# macro grad(nnet::Symbol, expr::Expr)
#     texpr = grad_parser(nnet, expr)
#     return texpr
# end

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
