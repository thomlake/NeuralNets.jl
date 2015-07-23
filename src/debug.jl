const NNET_JL_DEBUG_LEVEL = int(get(ENV, "NNET_JL_DEBUG_LEVEL", 1))

macro nnet_assert(expr::Expr)
    # if NNET_JL_DEBUG_LEVEL > 0
    #     return @assert(expr)
    # end
    nothing
end
