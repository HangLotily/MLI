# MLI

This is a Python 3.6 implementation of "MLI: A multi-level inference mechanism for user attributes over social network". MLI is a method for inferring user attributes with hierarchical structure in social networks.

## Requirements

- networkx
- numpy
- scipy
- pickle

## Dataset

1. dblp-public-graph

   The relationship between users. Each line represents two researchers who are co-authors and have an undirected edge on the graph.

   Sample:

   ```
   #184938# #481197#
   #45948# #57606#
   #58010# #859500#
   ```

   

2. label_data

   Users and their attributes. Each line represents a user's username, user id and his research interests which wrapped in a '#' pair.

   Sample:

   ```
   #Ankit Singh# #184938# #cs,image,learning,classification#
   ```

   

3. labelTree_data

   The tree structure represents the hierarchy of user attributes. The lower level attributes are refinements of the upper level attributes. The first line is the parent attribute (cs) and the next line is its child attributes (network, data, hardware, security, learning, computing and software). Each two lines represent a hierarchical edge.

   Sample:

   ```
   cs
   network,data,hardware,security,learning,computing,software
   network
   mobile,wireless,protocol,qos
   ```

   

4. unknown_nodes

   Unknown users to be inferred. By random sampling a different number of unknown users can be obtained.

## Run codes

Firstly, run getDB.py to transform the data into undirected graphs, dictionaries, etc. and store them as pkl files for subsequent use in inference.

```
$ python getDB.py
```

Run inference.py to infer unknown user's hierarchical attributes. 

There are three parameters that specify the maximum iteration number, the minimum confidence level allowed to enter the iteration (0-1) and the maximum confidence level to jump out of the iteration (0-1). 

```
$ python inference.py 10 0.5 0.9
```

When the inference is completed, result.pkl is generated to store the inference results.

```
$ python evaluate.py
```

The inference results can be evaluated by multiple metrics through evaluate.py.

```python
>>> H-Precision
0.8608388888888828
>>> H-Reccall
0.8346622756111017
>>> H-F1
0.8475485136530957
>>> User Accuracy
0.5045427013930951
>>> Jaccard Dist
0.6135742501884266
>>> Hamming Loss
0.22106666666666666
```
