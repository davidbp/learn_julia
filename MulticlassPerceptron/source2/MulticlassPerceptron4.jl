
module MulticlassPerceptron4

export predict, MPerceptron, fit!

type MPerceptron{T}
    W::Array{T}
    b::Array{T}
    n_classes::Int
    n_features::Int
end

MPerceptron(T::Type, n_classes::Int, n_features::Int) = MPerceptron(rand(T, n_classes, n_features),
                                                                    zeros(T, n_classes),
                                                                    n_classes,
                                                                    n_features)

function accuracy(y_true, y_hat)
    acc = 0.
    @inbounds for k = 1:length(y_true)
            acc += y_true[k] == y_hat[k]
        end
    return acc/length(y_hat)
end

function predict(h::MPerceptron, x, placeholder)
    placeholder .= A_mul_B!(placeholder, h.W,x) .+ h.b
    return indmax(placeholder)
end

function Base.show{T}(io::IO, p::MPerceptron{T})
    n_classes  = p.n_classes
    n_features = p.n_features
    print(io, "Perceptron{$T}(n_classes=$n_classes, n_features=$n_features)")
end

"""
function to fit a Perceptron
    '
    fit!(h::Perceptron, X_tr::Array, y_tr::Array, n_epochs::Int, learning_rate=0.1)
    '
"""
function fit!(h::MPerceptron, X_tr::Array, y_tr::Array, n_epochs::Int, learning_rate=0.1)

    T = eltype(X_tr)
    n_samples = size(X_tr, 2)
    y_signal_placeholder = zeros(T, h.n_classes)
    y_preds = zeros(Int64, n_samples)

    @inbounds for epoch in 1:n_epochs
        for m in 1:n_samples
            x = view(X_tr,:,m)
            y_hat = predict(h, x, y_signal_placeholder)
            if y_tr[m] != y_hat
                h.W[y_tr[m], :] .+= learning_rate .* x
                h.b[y_tr[m]]     .+= learning_rate
                h.W[y_hat, :]   .-= learning_rate .* x
                h.b[y_hat]       .-= learning_rate
            end
        end

        @inbounds for m in 1:n_samples
             y_preds[m] = predict(h, view(X_tr,:,m), y_signal_placeholder)
        end

        println("Accuracy epoch ", epoch, " is :", accuracy(y_tr, y_preds))
    end
end

end
