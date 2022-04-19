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

   The tree structure represents the hierarchy of user attributes. The lower level attributes are refinements of the upper level attributes. The first line is the parent attribute (cs) and the second line is its child attributes (network, data, hardware, security, learning, computing and software). Each two lines represent a hierarchical edge.

   ```
   cs
   network,data,hardware,security,learning,computing,software
   network
   mobile,wireless,protocol,qos
   ......
   ```

   

4. unknown_nodes

   Unknown users to be inferred. By random sampling a different number of unknown users can be obtained.

## Run codes

xxxxx

```
$ python getDB.py
```

xxxx

```
$ python inference.py 10 0.5 0.9
```

xxxxx

```
$ python evaluate.py
```
