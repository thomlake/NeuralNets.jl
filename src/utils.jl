vec2mat(b::Vector) = reshape(b, (size(b, 1), 1))

onehot(i::Int, d::Int) = (x = zeros(d); x[i] = 1; x)

argmax(x::Vector) = indmax(x)

function argmax(x::Matrix)
    n_rows, n_cols = size(x)
    imax = zeros(Int, n_cols)
    for j = 1:n_cols
        m = -Inf
        for i = 1:n_rows
            if x[i,j] > m
                m = x[i,j]
                imax[j] = i
            end
        end
    end
    return imax
end
