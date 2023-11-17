# Graph Sparsifications using Neural Network Assisted Monte Carlo Tree Search

Graph neural networks have been successful for machine learning, as well as for combinatorial and graph problems such as the Subgraph Isomorphism Problem and the Traveling Salesman Problem. We describe an approach for computing graph sparsifiers by combining a graph neural network and Monte Carlo Tree Search. We first train a graph neural network that takes as input a partial solution and  proposes a new node to be added as output. This neural network is then used in a Monte Carlo search to compute a sparsifier. The proposed method consistently outperforms several standard approximation algorithms on many different types of graphs and often finds the optimal solution.


The following packages are needed to run the code: torch, torch-scatter, torch-sparse, pytorch_geometric, networkx. Standard tool like pip can be used to install these pachages:
```
import torch
import os

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

The datasets are available in the "dataset/" folder. These datasets need to be converted to appropriate format using the "gnn/convert_data_steiner.py" file. To train the neural network, "gnn/train_steiner.py" needs to be used. To run the MCTS, "mcts/main_steiner.py" needs to be used.
