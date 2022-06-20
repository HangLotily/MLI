# MLI

This is a Python 3.6 implementation of "MLI: A multi-level inference mechanism for user attributes over social network". MLI is a method for inferring user attributes with hierarchical structure in social networks.

## Requirements

- networkx
- numpy
- scipy
- pickle

## Dataset

1. how the user attribute hierarchy prepared?

   We first clean the user attributes, which are the research fields in the dataset. For the problems that there are too much research fields and most of them rarely appear in the dataset, we merge attributes with the same or similar semantics, such as “Cluster”, “Clusters” and “Clustering”, “Tracking” and “Trace”, remove the attributes with unclear semantics, such as “z, un, old”, and only keep frequently occurring attributes. 

   After that, we focus on how to construct a reasonable hierarchy from a large number of attributes, which should not only semantically conform to the professional domain background, but also make most users in the dataset satisfy the hierarchy constraint.

   We construct the hierarchy in top-down manner, and the root node is “Computer Science”. Taking the classifications of computer research fields by ACM Computing Classification System (https://dl.acm.org/ccs) and China Computer Federation (https://www.ccf.org.cn) as background knowledge, we find that “Network”, “Big Data”, “Hardware”, “Security”, “Learning”, “Computing” and “Software” appear most frequently in the dataset and can basically cover the research fields of computer science, so we take these eight research fields as the second-level attributes of the hierarchy . When constructing the next level, we first select some attributes, which according to background knowledge are sub-research fields of second-level attributes and also appear frequently in the dataset. For Saeed Salem in the example, “Big Data” is his second-level attribute, and then we select “Data Mining” as the third-level attribute. Finally, we use the same method to construct the fourth-level, achieving level-by-level refinement of user attributes.
   
   
   
2. dblp-public-graph

   The relationship between users. Each line represents two researchers who are co-authors and have an undirected edge on the graph.

   Sample:

   ```
   #184938# #481197#
   #45948# #57606#
   #58010# #859500#
   ```

   

3. label_data

   Users and their attributes. Each line represents a user's username, user id and his research interests which wrapped in a '#' pair.

   Sample:

   ```
   #Ankit Singh# #184938# #cs,image,learning,classification#
   ```

   

4. labelTree_data

   The tree structure represents the hierarchy of user attributes. The lower level attributes are refinements of the upper level attributes. The first line is the parent attribute (cs) and the next line is its child attributes (network, data, hardware, security, learning, computing and software). Each two lines represent a hierarchical edge.

   Sample:

   ```
   cs
   network,data,hardware,security,learning,computing,software
   network
   mobile,wireless,protocol,qos
   ```

   

5. unknown_nodes

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



## Baselines

fp.py is one of our baselines from Rossi, Emanuele, et al. "[On the Unreasonable Effectiveness of Feature propagation in Learning on Graphs with Missing Node Features](https://markdown.com.cn)". arXiv preprint arXiv:2111.12128 (2021). According to the article, 40 iterations are enough to provide convergence, so we set the parameter of the number of iterations to 40.

```
$ python fp.py 40
```

In addition we also chose node2vec, one-all svm and other methods as baselines, they are implemented as follows.

```
$ python node2vec.py
$ python node2vec_dnn.py
$ python one-all svm.py
```
