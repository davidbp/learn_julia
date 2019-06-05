### Count triangles
using Base.Threads
function count_triangles(g, isDirected)
    nodes = length(g) 
    count_Triangle = Atomic{Int64}(0) 
    # Consider every possible triplet of edges in graph 
    @threads for i in 1:nodes
        for j in 1:nodes
            for k in 1:nodes
                # check the triplet if it satisfies the condition 
                if i!=j && i!=k && j!=k && g[i][j]==1 && g[j][k]==1 && g[k][i]==1
                    atomic_add!(count_Triangle, 1)
                end
            end
        end
    end
    # if graph is directed , division is done by 3  else division by 6 is done 
    if isDirected==true
        return count_Triangle.value/3 
    else
        return count_Triangle.value/6
    end
end


# Load graph
#using CSV
#graph_table = CSV.read("graph.csv")
#n_nodes = size(graph_table,1)
#graph_list_of_lists = [Array(graph_table[i]) for i in 1:n_nodes ]

graph = readlines("graph.csv")
graph_list_of_lists = map(x-> parse.(Int,split(x,",")[2:end]), graph[2:end])
#graph = map(x-> split(x,",")[2:end], graph[2:end])
#graph = map(x-> parse.(Int,x), graph)

# count trianges
t0 = time()
n_triangles = count_triangles(graph_list_of_lists[1:4], false)
println("Start Computing")
n_triangles = count_triangles(graph_list_of_lists, false)
println("Number of triangles: ", n_triangles)
println("Total time: ", abs(time()-t0), " seconds")

