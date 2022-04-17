# -*- coding: utf-8 -*-
import pickle
import random


def find(node, layer):
    tmp_list = []
    max_pro = max(result[node][layer].values())
    for key, value in result[node][layer].items():
        if value == max_pro:
            tmp_list.append(key)
    return tmp_list


def get_ancestor(label, layer, is_tree):
    ancestor_list = [label]
    if is_tree:
        if layer == 2:
            return ancestor_list
        elif layer == 3:
            ancestor_list.append(labelTree.parent(label).tag)
            return ancestor_list
        else:
            layer3_parent = labelTree.parent(label).tag
            layer2_parent = labelTree.parent(layer3_parent).tag
            ancestor_list.append(layer2_parent)
            ancestor_list.append(layer3_parent)
            return ancestor_list
    else:
        ancestor_list.extend(ancestor_dict[label])
        return ancestor_list


if __name__ == "__main__":
    pkl_file = open('target_nodes.pkl', 'rb')
    target_nodes = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('result.pkl', 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('label.pkl', 'rb')
    labelDB = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open('labelTree.pkl', 'rb')
    labelTree = pickle.load(pkl_file)
    pkl_file.close()

    Max_layer = 4
    target_num = len(target_nodes)

    ancestor_dict = {'network': [], 'data': [], 'hardware': [], 'security': [], 'learning': [], 'computing': [],
                     'software': [], 'mobile': ['network'], 'wireless': ['network'], 'protocol': ['network'], 'qos': ['network'],
                     'database': ['data'], 'mining': ['data', 'learning'], 'image': ['learning'], 'speech': ['learning'],
                     'theory': ['learning'], 'chip': ['hardware'], 'circuit': ['hardware'], 'authentication': ['security'],
                     'attack': ['security'], 'encryption': ['security'], 'distributed': ['computing'], 'development': ['software'],
                     'test': ['software'], 'bluetooth': ['mobile'], 'cellular': ['mobile', 'wireless', 'network'],
                     'wlan': ['mobile', 'network'], 'ip': ['mobile', 'protocol', 'wireless', 'qos', 'network'],
                     'localization': ['wireless', 'network'], 'rfid': ['wireless', 'network'], 'transmission': ['wireless', 'network'],
                     'zigbee': ['protocol', 'network'], 'gateway': ['protocol', 'network'],
                     'delay': ['qos', 'network'], 'bandwidth': ['qos', 'network'], 'index': ['database', 'data'], 'rdf': ['database', 'data'],
                     'sql': ['database', 'data'], 'query': ['database', 'data'], 'label': ['mining', 'data', 'learning'], 'patterns': ['mining', 'data', 'learning'],
                     'community': ['mining', 'data', 'learning'], 'classification': ['mining', 'data', 'learning', 'image'],
                     'recommendation': ['mining', 'data', 'learning'], 'clusters': ['wireless', 'network', 'mining', 'data', 'learning'],
                     'cmos': ['chip', 'hardware'], 'soc': ['chip', 'hardware'], 'microfluidic': ['chip', 'hardware'],
                     'fpga': ['circuit', 'hardware'], 'vhdl': ['circuit', 'hardware'], 'vlsi': ['circuit', 'hardware'],
                     'electrode': ['circuit', 'hardware'], 'validation': ['authentication', 'security'], 'signature': ['authentication', 'security'],
                     'certification': ['authentication', 'security'], 'infusion': ['attack', 'security'], 'virus': ['attack', 'security'],
                     'ddos': ['attack', 'security'], 'cryptography': ['encryption', 'security'], 'aes': ['encryption', 'security'],
                     'password': ['encryption', 'security'], 'segmentation': ['image', 'learning'], 'tracking': ['image', 'learning'],
                     'filtering': ['image', 'learning'], 'retrieval': ['image', 'database', 'learning', 'data'], 'recognition': ['image', 'speech', 'learning'],
                     'understanding': ['speech', 'learning'], 'optimization': ['theory', 'learning'], 'random': ['theory', 'learning'],
                     'game': ['theory', 'learning'], 'set': ['theory', 'learning'], 'function': ['theory', 'learning'],
                     'cloud': ['distributed', 'wireless', 'mobile', 'computing', 'network'], 'grid': ['distributed', 'computing'],
                     'parallel': ['distributed', 'computing'], 'programming': ['development', 'software'], 'android': ['development', 'software'],
                     'website': ['development', 'software'], 'debugging': ['test', 'software'], 'maintenance': ['test', 'software']}

    # H-precision、H-Recall、H-F1
    H_precision = 0
    H_recall = 0
    for node in target_nodes:
        H_precision_1node = 0
        H_recall_1node = 0
        for layer in range(2, Max_layer+1):
            H_precision_1layer = 0
            H_recall_1layer = 0
            predict_set = set()
            if len(set(result[node][layer].values())) == 1:
                H_precision_1node += 0
                H_recall_1node += 0
                continue
            for label in find(node, layer):
                for ancestor in get_ancestor(label, layer, is_tree=False):
                    predict_set.add(ancestor)
            tmp_list = []
            for key, value in labelDB[node][layer].items():
                if value == 1:
                    tmp_list.append(key)
            for label in tmp_list:
                real_set = set()
                for ancestor in get_ancestor(label, layer, is_tree=False):
                    real_set.add(ancestor)
                tmp_pre = len(list(predict_set & real_set)) / len(list(real_set))
                tmp_rec = len(list(predict_set & real_set)) / len(list(predict_set))
                if tmp_pre > H_precision_1layer:
                    H_precision_1layer = tmp_pre
                if tmp_rec > H_recall_1layer:
                    H_recall_1layer = tmp_rec
            H_precision_1node += H_precision_1layer
            H_recall_1node += H_recall_1layer
        H_precision += (H_precision_1node / 3)
        H_recall += (H_recall_1node / 3)
    H_precision = H_precision / len(target_nodes)
    H_recall = H_recall / len(target_nodes)
    H_F1 = 2 * H_precision * H_recall / (H_precision + H_recall)
    print("H-Precision:", H_precision)
    print("H-Recall:", H_recall)
    print("H-F1:", H_F1)

    # Jaccard dist
    tmp = 0
    num = 0
    for node in target_nodes:
        if len(set(result[node][4].values())) == 1:
            continue
        num += 1
        real_label_set = []
        pre_label_set = []
        for layer in range(2, Max_layer+1):
            for label in labelDB[node][layer].keys():
                if labelDB[node][layer][label] == 1:
                    real_label_set.append(label)
        for layer in range(2, Max_layer+1):
            for label in find(node, layer):
                pre_label_set.append(label)
        intersection_list = list(set(real_label_set) & set(pre_label_set))
        union_list = list(set(real_label_set) | set(pre_label_set))
        tmp += (len(intersection_list)) / (len(union_list))
    Jaccard = tmp / num
    print('\n')
    print("Jaccard dist:", Jaccard)

    # Accuracy
    right_node = 0
    empty_node = 0
    right_list = []

    for node in target_nodes:
        flag1 = False
        flag2 = False
        flag3 = False
        for layer in range(2, Max_layer+1):
            right_label = []
            for label in labelDB[node][layer].keys():
                if labelDB[node][layer][label] == 1:
                    right_label.append(label)
            if len(set(result[node][layer].values())) == 1:
                empty_node += 1
                break
            result_list = find(node, layer)
            result_random = random.sample(result_list, 1)
            for label in result_random:
                if label in right_label:
                    if layer == 2:
                        flag1 = True
                    elif layer == 3:
                        flag2 = True
                    else:
                        flag3 = True
                    break
        if flag1 and flag2 and flag3:
            right_list.append(node)
            right_node += 1
    Accuracy = right_node / (len(target_nodes) - empty_node)
    print("People Accuracy:", Accuracy)

    # Hamming Loss
    Loss = 0
    for node in target_nodes:
        for layer in range(2, Max_layer+1):
            right_label = []
            for label in labelDB[node][layer].keys():
                if labelDB[node][layer][label] == 1:
                    right_label.append(label)
            result_list = find(node, layer)
            result_random = random.sample(result_list, 1)
            for label in result_random:
                if label not in right_label:
                    Loss += 1
    Hamming_Loss = Loss / (3 * len(target_nodes))
    print("Hamming Loss:", Hamming_Loss)
