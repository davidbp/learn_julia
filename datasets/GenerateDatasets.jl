
module GenerateDatasets

"""
Generates bars and stripes data.
Given n_row, n_col the number of rows and number of columns respectively
this function returns all possible datapoints containing rows and columns.
"""
function generate_bars_and_stripes(n_row, n_col, T::DataType=Float32)
    
    function generate_rows(n_row,n_col, T::DataType=Float32)
    
        data = Array{T}[]
        perms =[ [ bit == '1' ? 1 : 0 for bit in bin(n,n_row) ] for n in 0:2^(n_row) ][2:end-2]
        for p in perms
            x = zeros(T, n_row, n_col)
            for (pos,val) in enumerate(p)
               if val == 1
                   x[pos,:] = 1
               end
            end
            push!(data, x)
        end
        return data
    end

    function generate_cols(n_row, n_col, T=Float32)
        data = Array{T}[]
        perms =[ [ bit == '1' ? 1 : 0 for bit in bin(n,n_col) ] for n in 0:2^(n_col) ][2:end-2]
        for p in perms
            x = zeros(T, n_row, n_col)
            for (pos,val) in enumerate(p)
               if val == 1
                   x[:,pos] = 1
               end
            end
            push!(data, x)
        end
        return data
    end

    X = Array{T}[]
    X_cols = generate_cols(n_row, n_col, T)
    X_rows = generate_rows(n_row, n_col, T)

    for x in X_cols
        push!(X, x)
    end
    
    for x in X_rows
        push!(X, x)
    end

    push!(X, ones(T, n_row, n_col))
    return X
end

end



