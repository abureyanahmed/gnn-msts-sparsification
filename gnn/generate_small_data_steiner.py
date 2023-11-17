import math
import networkx as nx
import argparse
from parse_config import ConfigParser
import env.generator_steiner as gen
import collections
from networkx.algorithms import approximation as approx

def distance(points, i, j):
    dx = points[i][0] - points[j][0]
    dy = points[i][1] - points[j][1]
    return math.sqrt(dx*dx + dy*dy)

def createGeometric(n):
    #G = nx.random_geometric_graph(n, 0.6)
    #G = nx.random_geometric_graph(n, 2.0)
    G = nx.random_geometric_graph(n, 0.5)
    while not nx.is_connected(G):
      G = nx.random_geometric_graph(n, 0.5)
    V = []
    V=range(n)
    points = []
    for i, d in G.nodes.data():
        points.append(d["pos"])
    for u,v in G.edges():
        G[u][v]['weight'] = distance(points, u, v)
    return G, points

# Data generation
#'''
nGraphs = 100
#n = 5
n = 15
#train_test = "train"
train_test = "test"
#fileName = "data/random/tsp"+str(n)+"/tsp"+str(n)+"_train.txt"
#fileNamePath = "data/random/tsp"+str(n)+"/tsp"+str(n)+"_train_path.txt"
fileName = "data/random/st"+str(n)+"/st"+str(n)+"_" + train_test + ".txt"
fileNamePath = "data/random/st"+str(n)+"/st"+str(n)+"_" + train_test + "_steiner_nodes.txt"
fileNameTerminals = "data/random/st"+str(n)+"/st"+str(n)+"_" + train_test + "_terminals.txt"
fileNameEdges = "data/random/st"+str(n)+"/st"+str(n)+"_" + train_test + "_edges.txt"
f = open(fileName, "w")
fPath = open(fileNamePath, "w")
fTerminals = open(fileNameTerminals, "w")
fEdges = open(fileNameEdges, "w")
for i in range(nGraphs):
    G, points = createGeometric(n)
    #nx.algorithms.tsp.traveling_salesperson_qubo(G)
    #cycle = approx.greedy_tsp(G)
    terminal_nodes = [0, 1, 2, 4, 5]
    st = approx.steinertree.steiner_tree(G, terminal_nodes)
    print(st.nodes())
    steiner_nodes = []
    for node in st.nodes():
      if node not in terminal_nodes:
        steiner_nodes.append(node)
    l = ""
    for p in points:
        l = l + str(p[0]) + " " + str(p[1]) + " "
    #l = l + "output "
    for v in steiner_nodes:
      l = l + str(v+1) + " "
    f.write(l + "\n")
    l = ""
    for v in steiner_nodes:
      l = l + str(v) + " "
    fPath.write(l + "\n")
    l = ""
    for v in terminal_nodes:
      l = l + str(v+1) + " "
    fTerminals.write(l + "\n")
    l = ""
    for e in G.edges(data="weight"):
      u, v, w = e
      l = l + str(u) + " " + str(v) + " " + str(w) + " "
    fEdges.write(l + "\n")
f.close()
fPath.close()
fTerminals.close()
fEdges.close()
#'''
quit()

args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-c', '--config', default=None, type=str,
                  help='config file path (default: None)')
args.add_argument('-r', '--resume', default=None, type=str,
                  help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,
                  help='indices of GPUs to enable (default: all)')
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
    CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
]
config = ConfigParser.from_args(args, options)
dataGenerator = gen.DataGenerator(config)
dataGenerator.run("dataset/random/st_15/processed/")




