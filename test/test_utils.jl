using Base.Test
using NeuralNets
const nnx = NeuralNets.Extras

# vec2mat tests
v = randn(20)
m = nnx.vec2mat(v)
@test size(m) == (20, 1)
@test all(v .== m)
@test_throws(MethodError, nnx.vec2mat(randn(3, 4)))
@test_throws(MethodError, nnx.vec2mat(1))

# test onehot
d = 20
for i = 1:d
    v = nnx.onehot(i, d)
    for j = 1:d
        @test i == j ? v[j] == 1 : v[j] == 0
    end
end

# test argmax
v = randn(20)
vmax, imax = -Inf, 0
for i = 1:length(v)
    if v[i] > vmax
        vmax = v[i]
        imax = i
    end
end
@test imax == nnx.argmax(v)

m = randn(20, 30)
imax = nnx.argmax(m)
@test size(imax) == (30,)
for j = 1:size(m, 2)
    @test nnx.argmax(m[:,j]) == imax[j]
end

@test_throws(MethodError, nnx.argmax(randn(3, 4, 5)))
@test_throws(MethodError, nnx.argmax(randn()))

# test zscore matrix
x = randn(5, 20)
mu = mean(x, 2)
sigma = std(x, 2)
z = (x .- mu) ./ sigma
@test_approx_eq nnx.zscore(x) z
@test_approx_eq nnx.zscore(x, mu, sigma) z

# test zscore matrix with std == 0 on the last row
x[end,:] = 0
mu = mean(x, 2)
sigma = std(x, 2)
z = (x .- mu) ./ sigma
@test sigma[end] .== 0
@test all(isnan(z[end,:]))
@test_approx_eq nnx.zscore(x)[1:end-1,:] z[1:end-1,:]
@test all(nnx.zscore(x)[end,:] .== 0)
@test_approx_eq nnx.zscore(x, mu, sigma)[1:end-1,:] z[1:end-1,:]

# test zscore sequence
xs = Matrix[randn(3, rand(4:10)) for i = 1:5]
X = hcat(xs...)
mu = mean(X, 2)
sigma = std(X, 2)
zs = nnx.zscore(xs)
for t = 1:length(zs)
    z = (xs[t] .- mu) ./ sigma
    @test_approx_eq zs[t] z 
end

# test zscore sequence with std == 0 on the last row
xs = Matrix[randn(3, rand(4:10)) for i = 1:5]
for x in xs
    x[end,:] = 0
end
X = hcat(xs...)
mu = mean(X, 2)
sigma = std(X, 2)
@test sigma[end] .== 0
zs = nnx.zscore(xs)
for t = 1:length(zs)
    z = (xs[t] .- mu) ./ sigma
    @test all(isnan(z[end,:]))
    @test all(zs[t][end,:] .== 0)
    @test_approx_eq zs[t][1:end-1,:] z[1:end-1,:]
end
