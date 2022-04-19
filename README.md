# MLI

This is a Python 3.6 implementation of "MLI: A multi-level inference mechanism for user attributes over social network". MLI is a method for inferring user attributes with hierarchical structure in social networks.

## Requirements

- networkx
- numpy
- scipy
- pickle

## Dataset

1. dblp-public-graph

   xxxxxxx

   Sample:

   ```
   #184938# #481197#
   #45948# #57606#
   #58010# #859500#
   ```

2. label_data

   xxxxx

   Sample:

   ```
   #Ankit Singh# #184938# #cs, image, learning, classification#
   ```

   

3. labelTree_data

   xxxx

   ```
   cs
   network,data,hardware,security,learning,computing,software
   network
   mobile,wireless,protocol,qos
   ......
   ```

   

4. unknown_nodes

   xxxxx

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

