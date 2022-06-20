import pickle
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

pkl_file = open('graph.pkl', 'rb')
G = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('target_nodes.pkl', 'rb')
target_nodes = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('label.pkl', 'rb')
labelDB = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('coauthor.pkl', 'rb')
coauthor = pickle.load(pkl_file)
pkl_file.close()

node_list = G.nodes()
layer2_num = 7
layer3_num = 17
layer4_num = 56
Max_layer = 4

result = {}
for target_node in target_nodes:
    result[target_node] = {}
    for layer in range(1, Max_layer + 1):
        result[target_node][layer] = {}
for target_node in target_nodes:
    for key in labelDB[target_node][1].keys():
        result[target_node][1][key] = 1
    for key in labelDB[target_node][2].keys():
        result[target_node][2][key] = 1 / layer2_num
    for key in labelDB[target_node][3].keys():
        result[target_node][3][key] = 1 / layer3_num
    for key in labelDB[target_node][4].keys():
        result[target_node][4][key] = 1 / layer4_num

# Layer2 SVM
train_attr_2 = []
train_class_2 = []
for people in node_list:
    if people in target_nodes:
        continue
    neighbor_label =  [0] * layer2_num
    for neighbor in coauthor[people]:
        if neighbor in target_nodes:
            continue
        i = 0
        for l in labelDB[neighbor][2].keys():
            neighbor_label[i] += labelDB[neighbor][2][l]
            i += 1
    if sum(neighbor_label) != 0:
        neighbor_label = list(map(lambda x: x / sum(neighbor_label), neighbor_label))
    train_attr_2.append(neighbor_label)
    j = 1
    flag = True
    for label in labelDB[people][2].keys():
        if labelDB[people][2][label] == 1:
            train_class_2.append(j)
            flag = False
            break
        else:
            j += 1
    if flag:
        train_attr_2.pop()

test_attr_2 = []
test_class_2 = []
for people in target_nodes:
    neighbor_label =  [0] * layer2_num
    for neighbor in coauthor[people]:
        if neighbor in target_nodes:
            continue
        i = 0
        for l in labelDB[neighbor][2].keys():
            neighbor_label[i] += labelDB[neighbor][2][l]
            i += 1
    if sum(neighbor_label) != 0:
        neighbor_label = list(map(lambda x: x / sum(neighbor_label), neighbor_label))
    test_attr_2.append(neighbor_label)
    j = 1
    for label in labelDB[people][2].keys():
        if labelDB[people][2][label] == 1:
            test_class_2.append(j)
            break
        else:
            j += 1

model = OneVsRestClassifier(svm.LinearSVC())
model.fit(train_attr_2, train_class_2)
result2 = model.predict(test_attr_2)
result2 = result2.tolist()
i = 0
for node in target_nodes:
    label_index = 1
    for label in labelDB[node][2].keys():
        if result2[i] == label_index:
            result[node][2][label] = 1
        else:
            result[node][2][label] = 0
        label_index += 1
    i += 1
print('Level 2 is done')

# Layer3 SVM
train_attr_3 = []
train_class_3 = []
for people in G.nodes():
    if people in target_nodes:
        continue
    neighbor_label = [0] * layer3_num
    for neighbor in coauthor[people]:
        if neighbor in target_nodes:
            continue
        i = 0
        for l in labelDB[neighbor][3].keys():
            neighbor_label[i] += labelDB[neighbor][3][l]
            i += 1
    if sum(neighbor_label) != 0:
        neighbor_label = list(map(lambda x: x / sum(neighbor_label), neighbor_label))
    train_attr_3.append(neighbor_label)
    j = layer2_num + 1
    flag = True
    for label in labelDB[people][3].keys():
        if labelDB[people][3][label] == 1:
            train_class_3.append(j)
            flag = False
            break
        else:
            j += 1
    if flag:
        train_attr_3.pop()

test_attr_3 = []
test_class_3 = []
for people in target_nodes:
    neighbor_label = [0] * layer3_num
    for neighbor in coauthor[people]:
        if neighbor in target_nodes:
            continue
        i = 0
        for l in labelDB[neighbor][3].keys():
            neighbor_label[i] += labelDB[neighbor][3][l]
            i += 1
    if sum(neighbor_label) != 0:
        neighbor_label = list(map(lambda x: x / sum(neighbor_label), neighbor_label))
    test_attr_3.append(neighbor_label)
    j = layer2_num + 1
    for label in labelDB[people][3].keys():
        if labelDB[people][3][label] == 1:
            test_class_3.append(j)
            break
        else:
            j += 1

model = OneVsRestClassifier(svm.LinearSVC())
clt = model.fit(train_attr_3, train_class_3)
result3 = clt.predict(test_attr_3)
result3 = result3.tolist()
i = 0
for node in target_nodes:
    label_index = layer2_num + 1
    for label in labelDB[node][3].keys():
        if result3[i] == label_index:
            result[node][3][label] = 1
        else:
            result[node][3][label] = 0
        label_index += 1
    i += 1
print('Level 3 is done')

# Layer4 SVM
train_attr_4 = []
train_class_4 = []
for people in G.nodes():
    if people in target_nodes:
        continue
    neighbor_label = [0] * layer4_num
    for neighbor in coauthor[people]:
        if neighbor in target_nodes:
            continue
        i = 0
        for l in labelDB[neighbor][4].keys():
            neighbor_label[i] += labelDB[neighbor][4][l]
            i += 1
    if sum(neighbor_label) != 0:
        neighbor_label = list(map(lambda x: x / sum(neighbor_label), neighbor_label))
    train_attr_4.append(neighbor_label)
    j = layer2_num + layer3_num + 1
    flag = True
    for label in labelDB[people][4].keys():
        if labelDB[people][4][label] == 1:
            train_class_4.append(j)
            flag = False
            break
        else:
            j += 1
    if flag:
        train_attr_4.pop()

test_attr_4 = []
test_class_4 = []
for people in target_nodes:
    neighbor_label = [0] * layer4_num
    for neighbor in coauthor[people]:
        if neighbor in target_nodes:
            continue
        i = 0
        for l in labelDB[neighbor][4].keys():
            neighbor_label[i] += labelDB[neighbor][4][l]
            i += 1
    if sum(neighbor_label) != 0:
        neighbor_label = list(map(lambda x: x / sum(neighbor_label), neighbor_label))
    test_attr_4.append(neighbor_label)
    j = layer2_num + layer3_num +1
    for label in labelDB[people][4].keys():
        if labelDB[people][4][label] == 1:
            test_class_4.append(j)
            break
        else:
            j += 1

model = OneVsRestClassifier(svm.LinearSVC())
clt = model.fit(train_attr_4, train_class_4)
result4 = clt.predict(test_attr_4)
result4 = result4.tolist()
i = 0
for node in target_nodes:
    label_index = layer2_num + layer3_num +1
    for label in labelDB[node][4].keys():
        if result4[i] == label_index:
            result[node][4][label] = 1
        else:
            result[node][4][label] = 0
        label_index += 1
    i += 1
print('Level 4 is done')

output = open('result_svm.pkl', 'wb')
pickle.dump(result, output, 2)
output.close()