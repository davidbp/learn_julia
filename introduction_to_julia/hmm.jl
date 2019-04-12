type Sequence
    words::Array{String}
    labels::Array{String}
    
    function Sequence(words)
        states=["*" for x in words]
        return new(words, states)
    end
    
    function Sequence(words, states)
        return new(words, states)
    end
    
end

function Base.length(sequence::Sequence)
    return length(sequence.words)
end

function assign_elements_to_integers(elements)
    element_to_pos = Dict{String, Int64}()
    for (k,e) in enumerate(elements)
        element_to_pos[e] = k
    end
    return element_to_pos
end

##### Train Hmm #####

function update_initial_counts!(initial_counts, seq, state_to_pos)
    initial_counts[state_to_pos[seq.labels[1]]] = initial_counts[state_to_pos[seq.labels[1]]] + 1
end

function update_transition_counts!(transition_counts, seq, state_to_pos)
    for (t1,t2) in zip(seq.labels[1:end-1], seq.labels[2:end])
        transition_counts[state_to_pos[t1], state_to_pos[t2]] += 1 
    end    
end

function update_emission_counts!(emission_counts, seq, state_to_pos, word_to_pos)
    for (t,w) in zip(seq.labels, seq.words)
        emission_counts[state_to_pos[t], word_to_pos[w]] += 1 
    end 
end

function update_final_counts!(final_counts, seq, state_to_pos)
    final_counts[state_to_pos[seq.labels[end]]] +=1
end

function sufficient_statistics_hmm(sequences, state_to_pos, word_to_pos)
    
    n_states = length(state_to_pos)
    n_words = length(word_to_pos)
    
    initial_counts      = zeros(n_states)
    transition_counts   = zeros(n_states, n_states)
    final_counts        = zeros(n_states) 
    emission_counts     = zeros(n_states, n_words)
    
    for seq in sequences
        update_initial_counts!(initial_counts, seq, state_to_pos)
        update_transition_counts!(transition_counts, seq,  state_to_pos)
        update_emission_counts!(emission_counts, seq,  state_to_pos, word_to_pos) 
        update_final_counts!(final_counts, seq,  state_to_pos) 
    end
    
    return initial_counts, transition_counts, final_counts, emission_counts
end

type Hmm
    possible_words::Set{String}
    possible_states::Set{String}
    
    word_to_pos::Dict{String, Int64}
    state_to_pos::Dict{String, Int64}   
    pos_to_word::Dict{Int64, String}
    pos_to_state::Dict{Int64, String}

    initial_counts::Vector{Int64}
    transition_counts::Matrix{Int64} 
    emission_counts::Matrix{Int64}
    final_counts::Vector{Int64}

    initial_probs::Vector{Float64}
    transition_probs::Matrix{Float64}
    emission_probs::Matrix{Float64}
    final_probs::Vector{Float64}
    
    trained::Bool
    
    Hmm() = new(Set{String}(), 
                Set{String}(),
                Dict{String, Int64}(),
                Dict{String, Int64}(),
                Dict{Int64, String}(),
                Dict{Int64, String}(),
                Vector{Int64}(0),
                Matrix{Int64}(0, 0),
                Matrix{Int64}(0, 0),
                Vector{Int64}(0),
                Vector{Int64}(0),
                Matrix{Int64}(0, 0),
                Matrix{Int64}(0, 0),
                Vector{Int64}(0),
                false)
    
   
end


function logsum_pair(logx, logy):
    """
    Return log(x+y), avoiding arithmetic underflow/overflow.
    logx: log(x)
    logy: log(y)

    Rationale:
        x + y    = e^logx + e^logy = e^logx (1 + e^(logy-logx))
    therefore:
        log(x+y) = logx + log(1 + e^(logy-logx)) (1)

    Likewise,
    log(x+y) = logy + log(1 + e^(logx-logy)) (2)

    The computation of the exponential overflows earlier and is less precise
    for big values than for small values. Due to the presence of logy-logx
    (resp. logx-logy), (1) is preferred when logx > logy and (2) is preferred
    otherwise.
    """
    if logx == -Inf
        return logy
    elseif logx > logy
        return logx + log1p( exp(logy-logx))
    else
        return logy + log1p( exp(logx-logy))
    end
end

function logsum(logv::Array):
    """
    Return log(v[0] + v[1] + ...), avoiding arithmetic underflow/overflow.
    """
    res = -Inf
    for val in logv
        res = logsum_pair(res, val)
    end
    return res
end


#### INFERENCE #####

function compute_scores(hmm, sequence)
    length_sequence = length(sequence.words)
    n_states = length(hmm.possible_states)
    
    initial_scores = log.(hmm.initial_probs)
    transition_scores = log.(hmm.transition_probs)

    sequence_words_integers = [hmm.word_to_pos[x] for x in sequence.words]
    emission_scores = log.(hmm.emission_probs[:, sequence_words_integers])
    final_scores = log.(hmm.final_probs)
    
    return initial_scores, transition_scores, final_scores, emission_scores
end

function run_log_forward(initial_scores,
                         transition_scores,
                         final_scores,
                         emission_scores)
    """
    Compute the log_forward computations
    
    Assume there are K possible states and a sequence of length N.
    This method will compute iteritavely the log_forward quantities.
    
    * log_f is a K x N Array.
    * log_f[:,i] will contain the forward quantities at position i.
    * log_f[:,i] is a vector of size K
    
    Returns
    - log_f: Array of size K x N
    """
    length_sequence = size(emission_scores)[2]  
    n_states = length(hmm.state_to_pos)         # number of states
    
    # Forward variables initialized to Infinity because log(0) = Inf
    log_f = zeros(n_states, length_sequence) .+ Inf

    # Initialization
    log_f[:,1] = emission_scores[:,1] + initial_scores
    
    for n in 2:length_sequence
        for s in 1:n_states
            log_f[s,n] = logsum(log_f[:,n-1] + transition_scores[:,s]) + emission_scores[s,n]
        end
    end
    
    log_likelihood = logsum(log_f[:,length_sequence] + final_scores)    
    return log_likelihood, log_f
end


function run_log_backward(initial_scores,
                          transition_scores,
                          final_scores,
                          emission_scores)
    """
    Compute the log_backward computations
    
    Assume there are K possible states and a sequence of length N.
    This method will compute iteritavely the log_forward quantities.
    
    * log_b is a K x N Array.
    * log_b[:,i] will contain the forward quantities at position i.
    * log_b[:,i] is a vector of size K
    
    Returns
    - log_b::Array{Float64,2}, size(log_b)=(K,N)
    - log_likelihood::Float64
    """
    length_sequence = size(emission_scores)[2]
    n_states = length(initial_scores)
    log_b = zeros(n_states, length_sequence) - Inf

    # Initialization
    log_b[:,length_sequence] = final_scores

    for n in length_sequence-1:-1:1
        for s in 1:n_states
            log_b[s,n] = logsum(log_b[:,n+1] + transition_scores[s,:] + emission_scores[:,n+1])
        end
    end
    
    log_likelihood = logsum(log_b[:,1] + initial_scores + emission_scores[:,1])
    
    return log_likelihood, log_b
end


function compute_state_posteriors(initial_scores, transition_scores, final_scores, emission_scores)
    num_states = size(emission_scores)[1]  # Number of states.
    length = size(emission_scores)[2]      # Length of the sequence.
    
    log_likelihood, forward =  run_log_forward(initial_scores,
                                          transition_scores,
                                          final_scores,
                                          emission_scores)
    
    log_likelihood, backward = run_log_backward(initial_scores,
                                            transition_scores,
                                            final_scores,
                                            emission_scores)
    
    state_posteriors = zeros(num_states, length)      
    for pos in 1:length
        state_posteriors[:, pos] = forward[:, pos] + backward[:, pos] - log_likelihood
    end
    return state_posteriors
end

function posterior_decode(hmm::Hmm, sequence::Sequence; return_integers=false)  
    initial_scores, transition_scores, final_scores, emission_scores = compute_scores(hmm, sequence)
    state_posteriors = compute_state_posteriors(initial_scores, transition_scores, final_scores, emission_scores)
    predicted_tags = mapslices(indmax, state_posteriors, 1)
    
    if return_integers == false
        return vec([hmm.pos_to_state[tag] for tag in predicted_tags])
    else
        return predicted_tags
    end
end

function get_possible_words_and_states(sequences)
    state_counter = 1
    word_counter = 1
    
    possible_words = Set{String}()
    possible_states = Set{String}()
    
    for seq in sequences
        for (t,w) in zip(seq.labels, seq.words)
            push!(possible_states, t)
            push!(possible_words, w)
        end
    end
    
    return possible_words, possible_states
end

function fit!(hmm::Hmm, sequences::Array{Sequence})
    
    possible_words, possible_states =  get_possible_words_and_states(sequences)
    word_to_pos = assign_elements_to_integers(possible_words);
    state_to_pos = assign_elements_to_integers(possible_states);
    
    hmm.word_to_pos = word_to_pos
    hmm.state_to_pos = state_to_pos
    hmm.pos_to_word = map(reverse, hmm.state_to_pos)
    hmm.pos_to_state = map(reverse, hmm.state_to_pos)

    counts = sufficient_statistics_hmm(sequences, state_to_pos, word_to_pos)
    initial_counts, transition_counts, final_counts, emission_counts = counts
    
    hmm.possible_words = possible_words
    hmm.possible_states = possible_states
    
    hmm.initial_counts = initial_counts
    hmm.transition_counts = transition_counts
    hmm.final_counts = final_counts
    hmm.emission_counts = emission_counts
    
    ### This could be rewritten using for loops and it could be much cleaarer
    hmm.initial_probs = (initial_counts / sum(initial_counts))
    hmm.transition_probs = transition_counts./(sum(transition_counts, 2) + final_counts)
    # vec is added here because hmm.final_probs is defined as a Vector and 
    # sum(transition_counts, 2) is a  matrix of size (K,1) instead of a vector of size (K,)
    hmm.final_probs =  final_counts ./ (vec(sum(transition_counts, 2)) + final_counts )
    hmm.emission_probs = (emission_counts ./ sum(emission_counts, 2));
end

