import pickle
import numpy
import copy
from keras.models import Sequential
from keras.layers import Dense
from timeit import default_timer as timer


def deep_model(feature_dim, label_dim):
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=feature_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


pkl_file = open('graph.pkl', 'rb')
G = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('label.pkl', 'rb')
labelDB = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('target_nodes.pkl', 'rb')
target_nodes = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('node_embeddings_dict.pkl', 'rb')
node_embeddings = pickle.load(pkl_file)
pkl_file.close()

test_nodes = []
for node in G.nodes():
    if node not in target_nodes:
        test_nodes.append(node)

X_train_row_num = len(G.nodes()) - len(target_nodes)
X_test_row_num = len(target_nodes)
Y_train_row_num = len(G.nodes()) - len(target_nodes)
Y_test_row_num = len(target_nodes)
X_train = numpy.zeros(shape=(X_train_row_num, 64))
X_test = numpy.zeros(shape=(X_test_row_num, 64))
Y_train = numpy.zeros(shape=(Y_train_row_num, 80))
Y_test = numpy.zeros(shape=(Y_test_row_num, 80))


start = timer()
i = 0
for person in test_nodes:
    X_train[i] = numpy.array(node_embeddings[person])
    label = []
    for layer in range(2, 5):
        for key in labelDB[person][layer].keys():
            label.append(float(labelDB[person][layer][key]))
    Y_train[i] = label
    i += 1
task_time = timer() - start
print("X_train and Y_train is ready, took %f seconds" % task_time)

start = timer()
i = 0
for person in target_nodes:
    X_test[i] = numpy.array(node_embeddings[person])
    label = []
    for layer in range(2, 5):
        for key in labelDB[person][layer].keys():
            label.append(float(labelDB[person][layer][key]))
    Y_test[i] = label
    i += 1
task_time = timer() - start
print("X_test and Y_test is ready, took %f seconds" % task_time)


feature_dim = X_train.shape[1]
label_dim = Y_train.shape[1]
model = deep_model(feature_dim, label_dim)
model.summary()
model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_data=(X_test, Y_test))

result = model.predict(X_test)

result_dict = copy.deepcopy(labelDB)
result_index = 0
for person in target_nodes:
    list_index = 0
    tmp_list = result[result_index].tolist()
    for layer in range(2, 5):
        for label in labelDB[person][layer].keys():
            result_dict[person][layer][label] = tmp_list[list_index]
            list_index += 1
    result_index += 1

output = open('result_node2vec.pkl', 'wb')
pickle.dump(result_dict, output, 2)
output.close()

