import math
import networkx as nx
import argparse
from parse_config import ConfigParser
import env.generator_steiner as gen
import collections
from networkx.algorithms import approximation as approx
import os

def is_comment(x):
 if x[0]=='#':
  return True
 return False

def take_input(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 while True:
  l = file.readline()
  #print(l)
  if not is_comment(l):
   break
 m = int(l)
 edge_list = list()
 for i in range(m):
    while True:
     l = file.readline()
     if len(l) == 0:
      break
     if not is_comment(l):
      break
    t_arr1 = []
    t_arr2 = l.split()
    if(len(t_arr2)<3):break
    t_arr1.append(int(t_arr2[0])-1)
    t_arr1.append(int(t_arr2[1])-1)
    #t_arr1.append(int(t_arr2[0]))
    #t_arr1.append(int(t_arr2[1]))
    #t_arr1.append(int(t_arr2[2]))
    t_arr1.append(float(t_arr2[2]))
    edge_list.append(t_arr1)

 levels = int(file.readline())
 tree_ver=[]
 #tree_ver = [(int(x)-1) for x in raw_input().split()]
 for l in range(levels):
  #print "Steiner tree vertices of level "+str(l+1)+":"
  tree_ver.append([(int(x)-1) for x in file.readline().split()])
  #tree_ver.append([(int(x)) for x in file.readline().split()])

 file.close()
 return edge_list, tree_ver

def build_networkx_graph(filename):
 edge_list, tree_vers  = take_input(filename)
 G=nx.Graph()
 for e in edge_list:
  G.add_weighted_edges_from([(e[0], e[1], e[2])])
 return G, tree_vers

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

def get_points(file_name):
  points = []
  f = open(file_name, 'r')
  s = f.read()
  s = s.split()
  for i in range(len(s)):
   if i%2==0:
    points.append([float(s[i]), float(s[i+1])])
  f.close()
  return points

# Data generation
#'''
graph_ids = []
graph_names = []
#from_folder = "dataset/exp_GE_20/"
#from_folder = "dataset/exp_GE_20_no_rand/"
#from_folder = "dataset/exp_GE_20_400/"
#from_folder = "dataset/exp_ER_20/"
#from_folder = "dataset/exp_WS_20/"
#from_folder = "dataset/exp_BA_20/"
#from_folder = "dataset/exp_BA_20_p10/"
#from_folder = "dataset/exp_WS_20_t25/"
#from_folder = "dataset/exp_ER_20_wgt/"
#from_folder = "dataset/exp_WS_20_wgt/"
from_folder = "dataset/I080/"
ID_file = "id_to_file.csv"
f_id = open(from_folder+ID_file, 'r')
s = f_id.read()
lines = s.split('\n')
for l in lines:
  if len(l)>1:
    records = l.split(';')
    graph_ids.append(records[0])
    graph_names.append(records[3])
f_id.close()
#print(graph_ids)
#print(graph_names)
valid_graph_ids = []
valid_graph_names = []
output_folder = "log_folder_exact/"
for i, id in enumerate(graph_ids):
  output_file = from_folder+output_folder+graph_names[i]+"_output.txt"
  if os.path.exists(output_file):
    #print(output_file)
    valid_graph_ids.append(graph_ids[i])
    valid_graph_names.append(graph_names[i])
nGraphs = len(valid_graph_ids)
#n = 5
#n = 15
#n = 20
n = 80
#train_size = 90
#train_size = 160
#train_size = 360
train_size = 80

nGraphs = train_size
train_test = "train"
valid_graph_ids = valid_graph_ids[:nGraphs]
valid_graph_names = valid_graph_names[:nGraphs]

'''
nGraphs = len(valid_graph_ids) - train_size
train_test = "test"
valid_graph_ids = valid_graph_ids[train_size:]
valid_graph_names = valid_graph_names[train_size:]
print(valid_graph_ids)
'''

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
    #G, points = createGeometric(n)
    G, Ts = build_networkx_graph(from_folder+valid_graph_names[i]+".txt")
    #nx.algorithms.tsp.traveling_salesperson_qubo(G)
    #cycle = approx.greedy_tsp(G)
    #terminal_nodes = [0, 1, 2, 4, 5]
    terminal_nodes = Ts[0]
    print(terminal_nodes)
    #st = approx.steinertree.steiner_tree(G, terminal_nodes)
    st, _ = build_networkx_graph(from_folder + output_folder + valid_graph_names[i]+"_output.txt")
    print(st.nodes())
    steiner_nodes = []
    for node in st.nodes():
      if node not in terminal_nodes:
        steiner_nodes.append(node)
    l = ""
    if 'GE' in from_folder:
      points = get_points(from_folder+valid_graph_names[i]+"_pos.txt")
      for p in points:
        l = l + str(p[0]) + " " + str(p[1]) + " "
    else:
      for p in range(n):
        l = l + str(0.0) + " " + str(0.0) + " "
    #l = l + "output "
    for v in steiner_nodes:
      l = l + str(v+1) + " "
    f.write(l + "\n")
    l = ""
    for v in steiner_nodes:
      l = l + str(v+1) + " "
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
#quit()
#'''

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
dataGenerator.run("dataset/random/st_"+str(n)+"/processed/")




