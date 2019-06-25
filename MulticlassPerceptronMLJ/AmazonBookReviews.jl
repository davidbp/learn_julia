
module AmazonBookReviews

export load_data, decode_words

# push!(LOAD_PATH, pwd())

function decode_words(x, pos_to_word)
    return [pos_to_word[position] for (position,counts) in enumerate(x) if counts>0]
end

function load_data(;path="",min_support=5, element_type=Float32)
    xneg = readlines(path * "negative.review")
    xpos = readlines(path * "positive.review")
    T = element_type
    word_counts_pos = Dict{String,T}([])
    word_counts_neg = Dict{String,T}([])
    
    for x in xneg
        x = split(x)[1:end-1]
        for w_c in x
            (w, c) = split(w_c, ':')
            c = parse(T, c)
            if haskey(word_counts_pos, w)
                word_counts_pos[w] = word_counts_pos[w] + c
            else
                word_counts_pos[w] = 1
            end
        end
    end

    for x in xpos
        x = split(x)[1:end-1]
        for w_c in x
            (w, c) = split(w_c, ':')
            c = parse(T, c)
            if haskey(word_counts_neg, w)
                word_counts_neg[w] = word_counts_neg[w] + c
            else
                word_counts_neg[w] = 1
            end
        end
    end

    print("min support:", min_support)
    total_counts = merge(+, word_counts_pos, word_counts_neg)
    supported_word_counts = Dict(w => c for (w,c) in total_counts if c >=min_support)
    word_to_pos = Dict(x => i for (i,x) in enumerate(keys(supported_word_counts)))
    pos_to_word = Dict(i => x for (i,x) in enumerate(keys(supported_word_counts)))

    n_samples = length(xneg)
    n_features = length(word_to_pos)
    data_encoded_neg = zeros(T, n_features , n_samples)

    for (m,x) in enumerate(xneg)
        x = split(x)[1:end-1]
        for w_c in x
            (w, c) = split(w_c, ':')
            c = parse(T, c)
            if haskey(word_to_pos,w)
                data_encoded_neg[word_to_pos[w],m] = c
            end
        end
    end

    n_samples = length(xpos)
    n_features = length(word_to_pos)
    data_encoded_pos = zeros(T, n_features , n_samples)

    for (m,x) in enumerate(xpos)
        x = split(x)[1:end-1]
        for w_c in x
            (w, c) = split(w_c, ':')
            c = parse(T, c)
            if haskey(word_to_pos,w)
                data_encoded_pos[word_to_pos[w],m] = c
            end
        end
    end

    X = hcat(data_encoded_neg , data_encoded_pos)
    y = vcat(-ones(length(xneg)), ones(length(xpos)) )

    return (word_to_pos, pos_to_word, supported_word_counts, (X,y))
end



end
