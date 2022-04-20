# -*- coding: utf-8 -*-
import networkx as nx
import pickle
import copy
from treelib import Tree


def mysplit1(s):
    tmp = s.split("#")
    return [tmp[x] for x in range(len(tmp))]


def mysplit2(s):
    tmp = s.split(",")
    return [tmp[x] for x in range(len(tmp)) if tmp[x] != '']


def get_layer(l):
    if l in allLabels[0]:
        return 1
    elif l in allLabels[1]:
        return 2
    elif l in allLabels[2]:
        return 3
    elif l in allLabels[3]:
        return 4
    else:
        print(l)
        return 5


# graphDB
G = nx.Graph()
with open('./data/label_data', 'r') as file:
    lines = file.readlines()
    for line in lines:
        node = mysplit1(line)[3]
        G.add_node(node)
with open('./data/dblp-public-graph', 'r') as file:
    lines = file.readlines()
    for line in lines:
        edge = mysplit1(line)   # edge -> ['', '513855', ' ', '485856', '\n']
        if (edge[1] in G.nodes()) and (edge[3] in G.nodes()):
            G.add_edge(edge[1], edge[3])
for node in list(G.nodes()):
    if G.degree(node) == 0:
        G.remove_node(node)
output = open('graph.pkl', 'wb')
pickle.dump(G, output, 2)
output.close()
print(G.number_of_nodes())
print("graphDB has generated")

# labelDB
allLabels = ['cs',
             ['network', 'data', 'hardware', 'security', 'learning', 'computing', 'software'],
             ['mobile', 'wireless', 'protocol', 'qos', 'database', 'mining', 'chip', 'circuit', 'authentication',
              'attack', 'encryption', 'image', 'speech', 'theory', 'distributed', 'development', 'test'],
             ['bluetooth', 'cellular', 'wlan', 'localization', 'rfid', 'transmission', 'ip', 'zigbee', 'gateway',
              'delay', 'bandwidth', 'index', 'rdf', 'sql', 'query', 'clusters', 'classification', 'patterns', 'label',
              'community', 'recommendation', 'cmos', 'soc', 'microfluidic', 'fpga', 'vhdl', 'vlsi', 'electrode',
              'validation', 'signature', 'certification', 'infusion', 'virus', 'ddos', 'cryptography', 'aes',
              'password', 'segmentation', 'tracking', 'filtering', 'retrieval', 'recognition', 'understanding',
              'optimization', 'random', 'game', 'set', 'function', 'cloud', 'grid', 'parallel', 'programming',
              'android', 'website', 'debugging', 'maintenance']]

labelDict = {1: {'cs': 1}, 2: {'network': 0, 'data': 0, 'hardware': 0, 'security': 0, 'learning': 0, 'computing': 0, 'software': 0},
             3: {'mobile': 0, 'wireless': 0, 'protocol': 0, 'qos': 0, 'database': 0, 'mining': 0, 'chip': 0,
                 'circuit': 0, 'authentication': 0, 'attack': 0, 'encryption': 0, 'image': 0, 'speech': 0, 'theory': 0,
                 'distributed': 0, 'development': 0, 'test': 0},
             4: {'bluetooth': 0, 'cellular': 0, 'wlan': 0, 'localization': 0, 'rfid': 0, 'transmission': 0, 'ip': 0,
                 'zigbee': 0, 'gateway': 0, 'delay': 0, 'bandwidth': 0, 'index': 0, 'rdf': 0, 'sql': 0,
                 'query': 0, 'clusters': 0, 'classification': 0, 'patterns': 0, 'label': 0, 'community': 0,
                 'recommendation': 0, 'cmos': 0, 'soc': 0, 'microfluidic': 0, 'fpga': 0, 'vhdl': 0, 'vlsi': 0,
                 'electrode': 0, 'validation': 0, 'signature': 0, 'certification': 0, 'infusion': 0, 'virus': 0,
                 'ddos': 0, 'cryptography': 0, 'aes': 0, 'password': 0, 'segmentation': 0, 'tracking': 0, 'filtering': 0,
                 'retrieval': 0, 'recognition': 0, 'understanding': 0, 'optimization': 0, 'random': 0,
                 'game': 0, 'set': 0, 'function': 0, 'cloud': 0, 'grid': 0, 'parallel': 0, 'programming': 0,
                 'android': 0, 'website': 0, 'debugging': 0, 'maintenance': 0}}

label_dict = {}
with open('./data/label_data', 'r') as file:
    lines = file.readlines()
    for line in lines:
        labelDict_tmp = copy.deepcopy(labelDict)
        num = mysplit1(line)[3]  # mysplit(line) -> ['', 'Yu Chen', ' ', '4307', ' ', 'platform,huawei,cs,cloud,', '\n']
        labels = mysplit2(mysplit1(line)[5])
        for label in labels:
            labelDict_tmp[get_layer(label)][label] = 1
        label_dict[num] = labelDict_tmp
output = open('label.pkl', 'wb')
pickle.dump(label_dict, output, 2)
output.close()
print("labelDB has generated")

# unknown_nodes
unknown_nodes = []
with open('./data/unknown_nodes', 'r') as file:
    lines = file.readlines()
    for line in lines:
        rs = line.rstrip('\n')
        unknown_nodes.append(rs)
output = open('unknown_nodes.pkl', 'wb')
pickle.dump(unknown_nodes, output, 2)
output.close()
print("unknown_nodes has generated")

# labelTreeDB
tree = Tree()
tree.create_node('cs', 'cs')
with open('./data/labelTree_data', 'r') as file:
    lines = file.readlines()
    flag = 0
    for line in lines:
        line = line.strip('\n')
        label_list = mysplit2(line)
        if flag % 2 == 0:
            parent_node = label_list[0]
        else:
            for label in label_list:
                tree.create_node(label, label, parent=parent_node)
        flag += 1
output = open('labelTree.pkl', 'wb')
pickle.dump(tree, output, 2)
output.close()
print("labelTreeDB has generated")

# coauthorDB
node_list = list(G.nodes())
node_num = len(node_list)
coauthor = {}
for i in node_list:
    neighbors = G.neighbors(i)
    neighbor = []
    for node in neighbors:
        neighbor.append(node)
    coauthor[i] = neighbor
output = open('coauthor.pkl', 'wb')
pickle.dump(coauthor, output, 2)
output.close()
print("coauthorDB has generated")
