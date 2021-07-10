
# Self-Made Survey

This folder contains the data of the self-made survey and the code used for the 
analysis. This folder contains following files:

1. zachary.ipynb: This is an example implementation of a GCN using the well-known
    Zachary's Karate club dataset. This file was used to gain a general understanding
    as to how a GNNs can be implemented in Python using dgl and pytorch. 

2. The csv files "Kenntnisse und Einstellungen im Finanzbereich" and 
    "Attitudes and Knowledge Regarding Finance" contain the survey responses. The 
    files are separate as the survey was conducted in German and in English in order
    to reach as many respondent as possible. The responses were collected using 
    Google forms and the row headers contain the questions and the answers below.

3. The survey.ipynb file contain the initial analysis for the self-made survey
    and focuses on the implementation of a Graph Convolutional Network.

4. The Graph_Sage_impl.ipynb file contains the implementation of GraphSage as well 
    Node2Vec for the self-made survey. This file contains the most relevant analyses 
    for the self-made survey.

5. The GAT.ipynb file contains an implementation for graph attention networks for the 
    self-made survey. The GAT model is not considered for the master's thesis, which 
    is why the file is not further commented. The GAT file contains a model implementation 
    and is kept for completeness. 

Due to the small sample size of only n=113 responses, this dataset cannot be considered
for any serious analysis and machine learning task. It however sparked the question of 
whether stochastic or deterministic MAG graphs should be created. Given the limited use 
of the dataset due to its small size, the dataset is only analyzed to a minimum extent.
