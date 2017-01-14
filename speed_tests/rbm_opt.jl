
function sigmoid(vector::Array{Float64})
    return 1./(1 + e.^(-vector))
end

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
    b1 = similar(rbm.W * Xbatch[:, 1])
    b2 = similar(rbm.W' * hneg)
    ehp = similar(rbm.hid_bias)
    ehn = similar(rbm.hid_bias)
    xneg = similar(Xbatch[:, 1])

    @inbounds for i in 1:batch_size
        x =  @view Xbatch[:,i]
        xneg =  @view Xbatch[:,i]

        for k in 1:K
            hneg = sigmoid( rbm.W * xneg .+ rbm.hid_bias) .> rand(rbm.n_hid)
            xneg = sigmoid( rbm.W' * hneg .+ rbm.vis_bias) .> rand(rbm.n_vis)
        end

        ehp = sigmoid(rbm.W * x + rbm.hid_bias)
        ehn = sigmoid(rbm.W * xneg + rbm.hid_bias)
 
        Delta_W .+= lr .* ( x * ehp' .- xneg * ehn')'
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
print("\n This RBM used scalar product\n\t")

# warmup 
contrastive_divergence_col_K(X_batch_col[:,1:10], rbm, 1, 0.01)

print(@time contrastive_divergence_col_K(X_batch_col, rbm, 1, 0.01))
print("\n")
print(@benchmark contrastive_divergence_col_K(X_batch_col, rbm, 1, 0.01))
print("\n")



