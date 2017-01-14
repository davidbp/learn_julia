
sigmoid(x::Float64) = 1. / (1. + exp(-x))

type RBM_col{T <: Real}
    W::Matrix{T}
    vis_bias::Vector{T}
    hid_bias::Vector{T}
    n_vis::Int64
    n_hid::Int64
    trained::Bool
end

function initializeRBM_col(n_vis::Int64, n_hid::Int64; sigma=0.01, T=Float64)

    return RBM_col{T}(sigma*randn(n_hid,n_vis),  # weight matrix
                      zeros(n_vis),              # visible vector
                      zeros(n_hid),              # Hidden vector
                      n_vis,                     # num visible units
                      n_hid,                     # num hidden unnits
                      false)                     # trained


end

function contrastive_divergence_col_K(Xbatch, rbm, K::Int64, lr::Float64)

    batch_size = size(Xbatch)[2]

    Delta_W = zeros(size(rbm.W))
    Delta_b = zeros(size(rbm.vis_bias))
    Delta_c = zeros(size(rbm.hid_bias))

    hneg = similar(rbm.hid_bias)
    b1 = similar(rbm.W * Xbatch[:,1])
    b2 = similar(rbm.W' * hneg)
    ehp = similar(rbm.hid_bias)
    ehn = similar(rbm.hid_bias)
    xneg = similar(Xbatch[:,1])
    @inbounds for i in 1:batch_size
        x = @view Xbatch[:,i]
        xneg .= @view Xbatch[:,i]

        for k in 1:K
            A_mul_B!(b1, rbm.W, xneg)
            hneg .= sigmoid.(b1 .+ rbm.hid_bias) .> rand.()
            At_mul_B!(b2, rbm.W, hneg)
            xneg .= sigmoid.(b2 .+ rbm.vis_bias) .> rand.()
        end

        A_mul_B!(b1, rbm.W, x)
        ehp .= sigmoid.(b1 .+ rbm.hid_bias)
        A_mul_B!(b1, rbm.W, xneg)
        ehn .= sigmoid.(b1 .+ rbm.hid_bias)

        ### kron vs no kron???
        Delta_W .+= lr .* (ehp .* x' .- ehn .* xneg')
        Delta_b .+= lr .* (x .- xneg)
        Delta_c .+= lr .* (ehp .- ehn)

    end

    rbm.W .+= Delta_W ./ batch_size;
    rbm.vis_bias .+= Delta_b ./ batch_size;
    rbm.hid_bias .+= Delta_c ./ batch_size;

    return
end


using BenchmarkTools

X_train = rand(5000,784);
X_batch_col = X_train'[:,1:200];
rbm = initializeRBM_col(784, 225)

print("\n This RBM took\n\t")
print(@time contrastive_divergence_col_K(X_batch_col, rbm, 1, 0.01))

print("\n This RBM benchmark\n\t")
print(@benchmark contrastive_divergence_col_K(X_batch_col, rbm, 1, 0.01))
print("\n\n")
