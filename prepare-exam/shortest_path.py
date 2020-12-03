from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import shortest_path

def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


graph = [
    [0, 15, 20, 0, 0, 0, 0],
    [0, 0, 0, 10, 25, 0, 0],
    [0, 0, 0, 15, 0, 20, 0],
    [0, 0, 0, 0, 20, 15, 30],
    [0, 0, 0, 0, 0, 0, 10],
    [0, 0, 0, 0, 0, 0, 20],
    [0, 0, 0, 0, 0, 0, 0],
]

# https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
graph = csr_matrix(graph)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
dist_matrix, predecessors = shortest_path(graph, directed=True, method='D', return_predecessors=True)

print(graph)
print(dist_matrix)
print(predecessors)

path = get_path(predecessors, 0, 6)

# 1 -> 2 -> 5 -> 7
print(path)

# 4 -> 5 = 20
# 4 -> 6 = 15
# 4 -> 7 = 30

# 5 -> 7 = 10