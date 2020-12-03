import numpy as np
from scipy.optimize import linprog

"""
Source: https://en.wikipedia.org/wiki/Shortest_path_problem#Linear_programming_formulation
Warning: it is not for real-world usage: inefficient data-structures and probably not the good solver.
Still it is good to understand linear programming and some application, like here to find the shortest path.
"""

edges = [('1', '2', 15),
         ('1', '3', 20),
         ('2', '5', 25),
         ('2', '4', 10),
         ('3', '4', 15),
         ('3', '6', 20),
         ('4', '5', 20),
         ('4', '6', 15),
         ('4', '7', 30),
         ('5', '7', 10),
         ('6', '7', 20),
         ('7', '7', 0)]

s, t = '1', '7'

""" Preprocessing """
nodes = sorted(set([i[0] for i in edges]))  # assumption: each node has an outedge
n_nodes = len(nodes)
n_edges = len(edges)

print(f'nodes = {nodes} ({n_nodes}) n edges: {n_edges}')


edge_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
for edge in edges:
    i, j, value = edge
    i_ind = nodes.index(i)
    j_ind = nodes.index(j)
    edge_matrix[i_ind, j_ind] = value


print(f'edge_matrix = {edge_matrix}')

"""
Used to know edges.
Return the indices of the elements that are non-zero.
https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
"""
nnz_edges = np.nonzero(edge_matrix)
print(f'nnz_edges = {nnz_edges}')

edge_dict = {}
counter = 0
# -1 because each node doesn't have an out edge. It is the case of the output.
for e in range(n_edges -1):
    a, b = nnz_edges[0][e], nnz_edges[1][e]
    edge_dict[(a,b)] = counter
    counter += 1

s_ind = nodes.index(s)
t_ind = nodes.index(t)

print(f'edge_dict = {edge_dict}')

""" LP """
bounds = [(0, 1) for i in range(n_edges)]
c = [i[2] for i in edges]

A_rows = []
b_rows = []

for source in range(n_nodes):
    out_inds = np.flatnonzero(edge_matrix[source, :])
    in_inds = np.flatnonzero(edge_matrix[:, source])

    """
    The equation for the source of the system = 1
    The equation for the output of the system = -1
    """
    rhs = 0
    if source == s_ind:
        rhs = 1
    elif source == t_ind:
        rhs = -1

    n_out = len(out_inds)
    n_in = len(in_inds)

    out_edges = [edge_dict[a, b] for a, b in np.vstack((np.full(n_out, source), out_inds)).T]
    in_edges = [edge_dict[a, b] for a, b in np.vstack((in_inds, np.full(n_in, source))).T]

    A_row = np.zeros(n_edges)
    A_row[out_edges] = 1
    A_row[in_edges] = -1

    A_rows.append(A_row)
    b_rows.append(rhs)

A = np.vstack(A_rows)
b = np.array(b_rows)
res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='revised simplex')

print(f'A = \n {A}, \n B = \n {b}')

print(res.x)

inv_edge_dict = {v: k for k, v in edge_dict.items()}
edges_path = [inv_edge_dict[x] for x in np.nonzero(res.x)[0]]

print(edges_path)

edges_path_name = []
for edge_path in edges_path:
    i, j = edge_path
    i_name = nodes[i]
    j_name = nodes[j]

    edges_path_name.append((i_name, j_name))

print(edges_path_name)
