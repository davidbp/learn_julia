
module RBM_Models

using Distributions
using BLAS
import Base.show

#### methods to export when this module is loaded
export RBM, initialize_RBM, CDK, initialize_CDK, fit!, partial_fit!

function sigmoid(x::Float32)
    return 1/(1 + exp(-x))
end

type RBM{T <: Real}
    n_vis::Int
    n_hid::Int
    W::Matrix{T}  
    vis_bias::Vector{T}     
    hid_bias::Vector{T}   
    trained::Bool
    n_epochs_trained::Int
end

function initialize_RBM(n_vis, n_hid, sigma, T)
    
    return RBM{T}(n_vis,                                   # num visible units 
                  n_hid,                                   # num hidden unnits
                  rand(Normal(0,sigma), n_hid, n_vis),     # weight matrix
                  zeros(n_vis),                            # visible vector  
                  zeros(n_hid),                            # Hidden vector
                  false,0)                                 # trained
end

function Base.show{T}(io::IO, rbm::RBM{T})
    n_vis = size(rbm.vis_bias, 1)
    n_hid = size(rbm.hid_bias, 1)
    trained = rbm.trained
    print(io, "RBM{$T}(n_vis=$n_vis, n_hid=$n_hid, trained=$trained)")
end


type CDK{T}
    K::Int
    batch_size::Int
    
    # Placeholders needed for the gradients of the parameters of the RBM
    grad_W::Matrix{T}         
    grad_vis_bias::Vector{T}     
    grad_hid_bias::Vector{T}   
    
    # Placeholders needed for performing CDK in a minibatch
    H::Matrix{T}
    V_hat::Matrix{T}
    H_hat::Matrix{T}
    rec_error::Float64 # This is probably irrelevant, allo
    
    # Placeholders needed for performing sampling in a minibatch
    V_sampling::Matrix{T}
    H_sampling::Matrix{T}   
    H_aux::Matrix{T}  
    V_aux::Matrix{T}  

end

function initialize_CDK(rbm::RBM, K, batch_size)
    """
    This function initializes a CDK type that will be used as placeholder for the
    memory needed for the gibbs sampling process needed at every minibatch update.
    """
    T = eltype(rbm.vis_bias)
    grad_W = zeros(T, size(rbm.W))
    grad_vis_bias = zeros(T, size(rbm.vis_bias))
    grad_hid_bias = zeros(T, size(rbm.hid_bias))
    V_hat = zeros(T, rbm.n_vis, batch_size)
    H_hat = zeros(T, rbm.n_hid, batch_size)
    H = zeros(T, rbm.n_hid, batch_size)
    V_sampling = zeros(T, rbm.n_vis, batch_size)
    H_sampling = zeros(T, rbm.n_hid, batch_size)
    H_aux = zeros(T, rbm.n_hid, batch_size)
    V_aux = zeros(T, rbm.n_vis, batch_size)

    cdk = CDK(K, batch_size, 
              grad_W, grad_vis_bias,grad_hid_bias,
              H, V_hat, H_hat, 0.,
              V_sampling, H_sampling, H_aux, V_aux)
    return cdk
end

function update_params!(rbm::RBM, opt::CDK, lr)
    rbm.W .+= lr .* opt.grad_W 
    rbm.vis_bias .+= lr .* opt.grad_vis_bias
    rbm.hid_bias .+= lr .* opt.grad_hid_bias
end

function fit!(rbm::RBM, 
              X::Matrix, 
              batch_size::Integer,
              n_epochs::Integer,
              lr::Real,
              shuffle_data::Bool,
              opt)
    
    T = eltype(X)
    lr = T(lr)
    n_samples = size(X)[2]
    indicies = [x:min(x + batch_size-1, n_samples) for x in 1:batch_size:n_samples]
    sample_perm = Vector(1:n_samples)
    n_minibatches = T(length(indicies))
    rec_errors = Vector{T}([])
    
    for epoch in 1:n_epochs
        rec_error = Float32(0.)
        
        # should  it be more efficient to Shuffle indicies not the whole data?
        # then access is not contiguous though
        if shuffle_data==true
            shuffle!(sample_perm)
            X .= X[:,sample_perm]
        end
        
        for minibatch_ind in indicies          
            partial_fit!(rbm, X[:, minibatch_ind], lr, opt)
            rec_error += opt.rec_error
        end
        
        push!(rec_errors, rec_error/n_minibatches)
        rbm.n_epochs_trained +=1
        print(rec_errors[end], "\n")
    end
    rbm.trained = true
    return rec_errors
end

function partial_fit!(rbm::RBM, X::Matrix,  lr::Real, opt::CDK)
    compute_grad!(rbm, X, opt)
    update_params!(rbm, opt, lr)    
end

function MSE(X, V_hat)
    aux = 0.
    for (x,y) in zip(X , V_hat)
        aux += (x - y)^2
    end
    return sqrt(aux)
end

function A_mul_B_plus_C!(C,A,B)
    BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, C);
end

function compute_grad!(rbm::RBM, X::Matrix,  opt::CDK)
    
    T = eltype(rbm.vis_bias)
    batch_size = size(X)[2]
    
    # Perform gibbs sampling to compute the negative phase
    for k in 1:opt.K
        rand!(opt.V_sampling)
        if k==1
            opt.H .= sigmoid.(A_mul_B!(opt.H_aux, rbm.W, X) .+ rbm.hid_bias)
            opt.V_hat .= sigmoid.(At_mul_B!(opt.V_hat, rbm.W, opt.H) .+ rbm.vis_bias) .> opt.V_sampling
            opt.H_hat .= sigmoid.(A_mul_B!(opt.H_aux, rbm.W, opt.V_hat) .+ rbm.hid_bias) 
        else
            opt.V_hat .= sigmoid.(At_mul_B!(opt.V_hat, rbm.W, opt.H_hat) .+ rbm.vis_bias) .> opt.V_sampling
            opt.H_hat .= sigmoid.(A_mul_B!(opt.H_aux, rbm.W, opt.V_hat) .+ rbm.hid_bias) 
        end
    end
    opt.grad_W .=  (A_mul_Bt!(opt.grad_W, opt.H, X) .- A_mul_Bt!(opt.grad_W, opt.H_hat, opt.V_hat))./ batch_size; 
    opt.grad_vis_bias .= sum!(opt.grad_vis_bias, X .- opt.V_hat)./ batch_size;
    opt.grad_hid_bias .= squeeze(sum((opt.H .- opt.H_hat), 2),2)./ batch_size;
    #opt.rec_error = MSE(X, (@view opt.V_hat))
    #opt.rec_error = MSE(X,  opt.V_hat)

    aux = 0.
    for (x,y) in zip(X , opt.V_hat)
        aux += (x - y)^2
    end
    opt.rec_error = sqrt(aux)

end

#function compute_grad_with_dot2!(rbm::RBM, X::Matrix,  opt::CDK)
#
#    T = eltype(rbm.vis_bias)
#    batch_size = size(X)[2]
#    
#    # Perform gibbs sampling to compute the negative phase
#    for k in 1:opt.K
#        rand!(opt.V_sampling)
#        if k==1       
#            opt.H .= sigmoid.(A_mul_B!(opt.H_aux, rbm.W, X) .+ rbm.hid_bias)
#            opt.V_hat .= sigmoid.(rbm.W'* opt.H .+ rbm.vis_bias) .> opt.V_sampling
#            opt.H_hat .= sigmoid.(A_mul_B!(opt.H_aux, rbm.W, opt.V_hat)  .+ rbm.hid_bias) 
#        else
#            opt.V_hat .= sigmoid.(rbm.W'* opt.H_hat .+ rbm.vis_bias) .> opt.V_sampling
#            opt.H_hat .= sigmoid.(A_mul_B!(opt.H_aux, rbm.W, opt.V_hat)  .+ rbm.hid_bias) 
#        end        
#    end   
#    opt.grad_W .=  (opt.H * X' .-  opt.H_hat * opt.V_hat')./ batch_size; 
#    opt.grad_vis_bias .= squeeze(sum((X .- opt.V_hat), 2),2)./ batch_size;
#    opt.grad_hid_bias .= squeeze(sum((opt.H .- opt.H_hat), 2),2)./ batch_size;
#    opt.rec_error = sqrt(sum((X .- opt.V_hat).^2))
#end

end
