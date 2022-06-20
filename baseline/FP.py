import pickle
import networkx as nx
import sys

pkl_file = open('graph.pkl', 'rb')
G = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('label.pkl', 'rb')
labelDB = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('coauthor.pkl', 'rb')
coauthor = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('target_nodes.pkl', 'rb')
target_nodes = pickle.load(pkl_file)
pkl_file.close()

result = {}
for target_node in target_nodes:
    for key in labelDB[target_node][2].keys():
        labelDB[target_node][2][key] = 0
    for key in labelDB[target_node][3].keys():
        labelDB[target_node][3][key] = 0
    for key in labelDB[target_node][4].keys():
        labelDB[target_node][4][key] = 0
for target_node in target_nodes:
    result[target_node] = labelDB[target_node]

Adj_Matrix = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), dtype=float)
nodelist = list(G.nodes())
M = Adj_Matrix * 0
M = M.tolil()
for i in G.nodes():
    for j in coauthor[i]:
        d_i = G.degree[i]
        d_j = G.degree[j]
        tmp = d_i * d_j
        M[nodelist.index(i), nodelist.index(j)] = 1 / pow(tmp, 0.5)

iter_ = 0
while iter_ < sys.argv[1]:
    print("iter %d :" % iter_)
    for person in G.nodes():
        if person not in target_nodes:
            for layer in range(2, 5):
                if max(labelDB[person][layer].values()) != 1:
                    for label in labelDB[person][layer].keys():
                        labelDB[person][layer][label] = 0
                    for j in coauthor[person]:
                        pro = M[nodelist.index(person), nodelist.index(j)]
                        for label in labelDB[person][layer].keys():
                            labelDB[person][layer][label] += pro * labelDB[j][layer][label]
                if sum(labelDB[person][layer].values()) != 0:
                    tmp = sum(labelDB[person][layer].values())
                    for label in labelDB[person][layer].keys():
                        labelDB[person][layer][label] = labelDB[person][layer][label] / tmp

    for person in target_nodes:
        for layer in range(2, 5):
            for label in result[person][layer].keys():
                result[person][layer][label] = 0
            for j in coauthor[person]:
                pro = M[nodelist.index(person), nodelist.index(j)]
                for label in result[person][layer].keys():
                    result[person][layer][label] += pro * labelDB[j][layer][label]
            if sum(result[person][layer].values()) != 0:
                tmp = sum(result[person][layer].values())
                for label in result[person][layer].keys():
                    result[person][layer][label] = result[person][layer][label] / tmp
    for target_node in target_nodes:
        labelDB[target_node] = result[target_node]
    iter_ += 1

output = open('result_FP.pkl', 'wb')
pickle.dump(result, output, 2)
output.close()
