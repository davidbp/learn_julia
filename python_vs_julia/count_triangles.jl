### Count triangles

function generate_triangles(nodes)
    result = []
    visited_ids = Set{Int}()
    for (node_a_id, nodes_node_a_id) in nodes
        temp_visited = Set()
        for node_b_id in nodes_node_a_id
            if node_b_id == node_a_id
                println("ERROR")
                break
            end
            if node_b_id in visited_ids
                continue
            end
            for node_c_id in nodes[node_b_id]
                if node_c_id in visited_ids
                    continue
                end
                if node_c_id in temp_visited
                    continue
                end
                if node_a_id in nodes[node_c_id]
                    push!(result, [node_a_id, node_b_id, node_c_id])
                else
                    continue
                end
            end        
            push!(temp_visited, node_b_id)
        end
        push!(visited_ids, node_a_id)
    end
    return result
end


function keys_to_ints(dict)
    aux = Dict{Int,Array{Int}}()
    for (k,v) in dict
       aux[parse(k)] = v
    end
    return aux
end

using JSON

aux = readlines("graph.json")
adj_dict = JSON.parse(aux[1])
adj_dict = keys_to_ints(adj_dict)

t0 = time()
println("Start Computing")
triangles = generate_triangles(adj_dict)
println("Number of triangles:", length(triangles))
println("Total time", abs(time()-t0)) 


