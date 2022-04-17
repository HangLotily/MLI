# -*- coding: utf-8 -*-
import pickle
import networkx as nx
from random import sample
import copy
import numpy as np
import scipy.sparse
from timeit import default_timer as timer
from treelib import Tree, Node
import sys


def get_entropy(label_dict):
    entropy = 0
    for pro in label_dict.values():
        if pro > 0:
            entropy += pro * np.log(pro)
    entropy = -entropy
    return entropy


def get_conf(author):
    num = 0
    conf = 0
    for i in coauthor[author]:
        num += 1
        conf += G.nodes[i]['confidence']
    conf = conf / num
    return conf


def update_conf():
    for person in uptoMinConf_dict.keys():
        G.nodes[person]['confidence'] = uptoMinConf_dict[person]


def run_inference(max_conf=0.9):
    # Calculate the entropy matrix M
    if iter_ == 0:
        for layer in range(2, Max_layer+1):
            M = Adj_Matrix * 0
            M = M.tolil()
            for node in target_nodes:
                entropy = get_entropy(labelDB[node][layer])
                for i in coauthor[node]:
                    M[nodelist.index(i), nodelist.index(node)] = entropy
            M_list.append(M)

    # Calculate the transition matrix pro_m
    prom_list = []
    for layer in range(2, Max_layer + 1):
        m = M_list[layer-2]
        pro_m = copy.deepcopy(m.tocsr())
        S = np.array(pro_m.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = scipy.sparse.spdiags(S.T, 0, *pro_m.shape, format='csr')
        pro_m = Q * pro_m
        for i in range(node_num):
            pro_m[i] *= G.nodes[nodelist[i]]['confidence']
        prom_list.append(pro_m)

    # Calculate attribute probability
    for person in target_nodes:
        conf = get_conf(person)
        if conf >= min_conf:
            uptoMinConf_dict[person] = conf
            if person not in already_infer:
                already_infer.append(person)
                new_infer.append(person)
            for layer in range(2, Max_layer+1):
                pro_m = prom_list[layer-2]
                for label in result[person][layer].keys():
                    if result[person][layer][label] != 1:
                        result[person][layer][label] = 0
                for i in coauthor[person]:
                    pro = pro_m[nodelist.index(i), nodelist.index(person)]
                    for label in result[person][layer].keys():
                        result[person][layer][label] += pro * labelDB[i][layer][label]
            if conf >= max_conf:
                uptoMaxConf.append(person)


# Update the entropy matrix
def update_entropy():
    for layer in range(2, Max_layer + 1):
        M = M_list[layer-2]
        for node in target_nodes:
            if node in uptoMaxConf:
                for i in coauthor[node]:
                    M[nodelist.index(i), nodelist.index(node)] = 0
            else:
                entropy = get_entropy(result[node][layer])
                for i in coauthor[node]:
                    M[nodelist.index(i), nodelist.index(node)] = entropy


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


def get_descent(l):
    tmp_list = []
    for label in children_dict[l]:
        tmp_list.append(label)
    tmpp_list = copy.deepcopy(tmp_list)
    for label in tmpp_list:
        if get_layer(label) == 4:
            break
        for l1 in children_dict[label]:
            if l1 not in tmp_list:
                tmp_list.append(l1)
    return tmp_list


def get_sum1(author, layer, label, parameter, tmp_result):
    pro_sum = 0
    descent = get_descent(label)
    for l in descent:
        pro_sum += tmp_result[author][get_layer(l)][l]
    tmp_results = (1-parameter) * tmp_result[author][layer][label] + parameter * pro_sum
    return tmp_results


def get_sum2(author, layer, parent, parameter, tmp_result):
    pro_sum = 0
    for brother in children_dict[parent]:
        descent = get_descent(brother)
        tmp = 0
        for l1 in descent:
            tmp += tmp_result[author][get_layer(l1)][l1]
        tmp1 = (1-parameter) * tmp_result[author][layer][brother] + parameter * tmp
        pro_sum += tmp1
    return pro_sum


# Calculate the probability sum of sibling nodes in the semantic tree
def get_sum3(author, layer, parent, tmp_result):
    pro_sum = 0
    for l in children_dict[parent]:
        pro_sum += tmp_result[author][layer][l]
    return pro_sum


# Number of nodes of the same parent node in the semantic tree
def get_num(parent_label):
    pro_sum = children_dict[parent_label]
    return len(pro_sum)


def run_modify(parameter=0.5):
    tmp_result = copy.deepcopy(result)
    for author in uptoMinConf_dict.keys():
        for layer in range(2, Max_layer+1):
            if layer == Max_layer:
                for label in result[author][layer].keys():
                    result[author][layer][label] = 0
                    for parent in parent_dict[label]:
                        if get_sum3(author, layer, parent, tmp_result) == 0:
                            result[author][layer][label] += result[author][layer - 1][parent] / get_num(parent)
                        else:
                            result[author][layer][label] += result[author][layer - 1][parent] * tmp_result[author][layer][label] / get_sum3(author, layer, parent, tmp_result)
            else:
                for label in result[author][layer].keys():
                    result[author][layer][label] = 0
                    for parent in parent_dict[label]:
                        if get_sum2(author, layer, parent, parameter, tmp_result) == 0:
                            result[author][layer][label] += result[author][layer - 1][parent] / get_num(label)
                        else:
                            result[author][layer][label] += result[author][layer - 1][parent] * get_sum1(author, layer, label, parameter, tmp_result) / get_sum2(author, layer, parent, parameter, tmp_result)


if __name__ == "__main__":

    start_all = timer()

    allLabels = ['cs',
                 ['network', 'data', 'hardware', 'security', 'learning', 'computing', 'software'],
                 ['mobile', 'wireless', 'protocol', 'qos', 'database', 'mining', 'chip', 'circuit', 'authentication',
                  'attack', 'encryption', 'image', 'speech', 'theory', 'distributed', 'development', 'test'],
                 ['bluetooth', 'cellular', 'wlan', 'localization', 'rfid', 'transmission', 'ip', 'zigbee', 'gateway',
                  'delay', 'bandwidth', 'index', 'rdf', 'sql', 'query', 'clusters', 'classification', 'patterns',
                  'label',
                  'community', 'recommendation', 'cmos', 'soc', 'microfluidic', 'fpga', 'vhdl', 'vlsi', 'electrode',
                  'validation', 'signature', 'certification', 'infusion', 'virus', 'ddos', 'cryptography', 'aes',
                  'password', 'segmentation', 'tracking', 'filtering', 'retrieval', 'recognition', 'understanding',
                  'optimization', 'random', 'game', 'set', 'function', 'cloud', 'grid', 'parallel', 'programming',
                  'android', 'website', 'debugging', 'maintenance']]

    # start = timer()

    pkl_file = open('graph.pkl', 'rb')
    G = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('label.pkl', 'rb')
    labelDB = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('labelTree.pkl', 'rb')
    labelTree = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('coauthor.pkl', 'rb')
    coauthor = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('unknown_nodes.pkl', 'rb')
    unknown_nodes = pickle.load(pkl_file)
    pkl_file.close()

    # task_time = timer() - start
    # print("loading took %f seconds" % task_time)

    # Randomly sample unknown users and reset the probability
    layer2_num = 7
    layer3_num = 17
    layer4_num = 56
    target_num = 5000
    target_nodes = sample(unknown_nodes, target_num)
    output = open('target_nodes.pkl', 'wb')
    pickle.dump(target_nodes, output, 2)
    output.close()
    target_num_before = len(target_nodes)

    result = {}
    for target_node in target_nodes:
        for key in labelDB[target_node][2].keys():
            labelDB[target_node][2][key] = 1 / layer2_num
        for key in labelDB[target_node][3].keys():
            labelDB[target_node][3][key] = 1 / layer3_num
        for key in labelDB[target_node][4].keys():
            labelDB[target_node][4][key] = 1 / layer4_num
    for target_node in target_nodes:
        result[target_node] = labelDB[target_node]

    # add confidence for users
    for node in G.nodes():
        if node in target_nodes:
            G.nodes[node]['confidence'] = 0
        else:
            G.nodes[node]['confidence'] = 1

    nodelist = list(G.nodes())
    node_num = len(nodelist)
    Adj_Matrix = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), dtype=float)
    Max_layer = 4
    label_num = 81
    min_conf = 0.5
    M_list = []
    already_infer = []
    iter_ = 0

    children_dict = {'cs': ['network', 'data', 'hardware', 'security', 'learning', 'computing', 'software'],
                     'network': ['mobile', 'wireless', 'protocol', 'qos'],
                     'data': ['database', 'mining'], 'hardware': ['chip', 'circuit'],
                     'security': ['authentication', 'attack', 'encryption'],
                     'learning': ['image', 'speech', 'theory', 'mining'], 'computing': ['distributed'],
                     'software': ['development', 'test'],
                     'mobile': ['bluetooth', 'cellular', 'wlan', 'ip', 'cloud'],
                     'wireless': ['localization', 'rfid', 'transmission', 'cellular', 'clusters', 'ip', 'cloud'],
                     'protocol': ['ip', 'zigbee', 'gateway'], 'qos': ['delay', 'bandwidth', 'ip'],
                     'database': ['index', 'rdf', 'sql', 'query', 'retrieval'],
                     'mining': ['label', 'patterns', 'community', 'clusters', 'classification', 'recommendation'],
                     'chip': ['cmos', 'soc', 'microfluidic'], 'circuit': ['fpga', 'vhdl', 'vlsi', 'electrode'],
                     'authentication': ['validation', 'signature', 'certification'],
                     'attack': ['infusion', 'virus', 'ddos'], 'encryption': ['cryptography', 'aes', 'password'],
                     'image': ['segmentation', 'tracking', 'filtering', 'retrieval', 'classification', 'recognition'],
                     'speech': ['recognition', 'understanding'],
                     'theory': ['optimization', 'random', 'game', 'set', 'function'],
                     'distributed': ['cloud', 'grid', 'parallel'], 'development': ['programming', 'android', 'website'],
                     'test': ['debugging', 'maintenance']}

    parent_dict = {'network': ['cs'], 'data': ['cs'], 'hardware': ['cs'], 'security': ['cs'], 'learning': ['cs'],
                   'computing': ['cs'], 'software': ['cs'],
                   'mobile': ['network'], 'wireless': ['network'], 'protocol': ['network'], 'qos': ['network'],
                   'database': ['data'], 'mining': ['data', 'learning'], 'image': ['learning'], 'speech': ['learning'],
                   'theory': ['learning'],
                   'chip': ['hardware'], 'circuit': ['hardware'], 'authentication': ['security'],
                   'attack': ['security'], 'encryption': ['security'], 'distributed': ['computing'],
                   'development': ['software'],
                   'test': ['software'], 'bluetooth': ['mobile'],
                   'cellular': ['mobile', 'wireless'], 'wlan': ['mobile'],
                   'ip': ['mobile', 'protocol', 'wireless', 'qos'],
                   'localization': ['wireless'], 'rfid': ['wireless'], 'transmission': ['wireless'],
                   'zigbee': ['protocol'], 'gateway': ['protocol'],
                   'delay': ['qos'], 'bandwidth': ['qos'], 'index': ['database'], 'rdf': ['database'],
                   'sql': ['database'],
                   'query': ['database'], 'label': ['mining'], 'patterns': ['mining'], 'community': ['mining'],
                   'classification': ['mining', 'image'], 'recommendation': ['mining'],
                   'clusters': ['wireless', 'mining'],
                   'cmos': ['chip'], 'soc': ['chip'], 'microfluidic': ['chip'], 'fpga': ['circuit'],
                   'vhdl': ['circuit'],
                   'vlsi': ['circuit'], 'electrode': ['circuit'],
                   'validation': ['authentication'], 'signature': ['authentication'],
                   'certification': ['authentication'],
                   'infusion': ['attack'], 'virus': ['attack'], 'ddos': ['attack'],
                   'cryptography': ['encryption'], 'aes': ['encryption'], 'password': ['encryption'],
                   'segmentation': ['image'], 'tracking': ['image'], 'filtering': ['image'],
                   'retrieval': ['image', 'database'], 'recognition': ['image', 'speech'], 'understanding': ['speech'],
                   'optimization': ['theory'], 'random': ['theory'], 'game': ['theory'], 'set': ['theory'],
                   'function': ['theory'], 'cloud': ['distributed', 'wireless', 'mobile'], 'grid': ['distributed'],
                   'parallel': ['distributed'], 'programming': ['development'], 'android': ['development'],
                   'website': ['development'], 'debugging': ['test'], 'maintenance': ['test']}

    while iter_ < 10:
        print("iter %d :" % iter_)
        new_infer = []
        uptoMaxConf = []
        uptoMinConf_dict = {}

        # The probability before inference
        resultBeforeInfer = []
        for node in target_nodes:
            for layer in range(2, Max_layer+1):
                for label in result[node][layer].keys():
                    resultBeforeInfer.append(result[node][layer][label])

        # start = timer()
        run_inference()
        # task_time = timer() - start
        # print("inference took %f seconds" % task_time)

        # start = timer()
        run_modify()
        # task_time = timer() - start
        # print("modify took %f seconds" % task_time)

        # start = timer()
        update_entropy()
        # task_time = timer() - start
        # print("update entropy took %f seconds" % task_time)

        # The probability after inference
        resultAfterInfer = []
        for node in target_nodes:
            for layer in range(2, Max_layer+1):
                for label in result[node][layer].keys():
                    resultAfterInfer.append(result[node][layer][label])

        # Determine convergence or not
        v = list(map(lambda x: x[0] - x[1], zip(resultAfterInfer, resultBeforeInfer)))
        sub = 0
        for i in v:
            sub += abs(i)
        print(sub)
        if sub < target_num_before * label_num * 1e-4:
            break

        # Update confidence and attribute probabilities
        # start = timer()
        for target_node in target_nodes:
            labelDB[target_node] = result[target_node]
        update_conf()
        # task_time = timer() - start
        # print("update confidence and probability took %f seconds" % task_time)

        # Users reaching confidence level are no longer inferred
        for person in uptoMaxConf:
            del target_nodes[target_nodes.index(person)]

        # If no new user joins the inference at the current minimum confidence level
        if len(new_infer):
            min_conf = 0.5
        else:
            min_conf /= 2

        if len(target_nodes) == 0:
            break
        iter_ = iter_ + 1

    output = open('result.pkl', 'wb')
    pickle.dump(result, output, 2)
    output.close()

    task_time = timer() - start_all
    print("inference is done took %f seconds" % task_time)
