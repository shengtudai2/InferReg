pos_count = dict()
edge_count = dict()
poseEgdes = open("data/raw/pos_edges.tsv", "r")
allEdges = open("data/raw/edges.tsv", "r")
for line in poseEgdes:
    tf = line.split('\t')[0]
    target = line.split('\t')[1].strip()
    pos_count[tf] = len(target.split(','))
for line in allEdges:
    tf = line.split('\t')[0]
    target = line.split('\t')[1].strip()
    edge_count[tf] = len(target.split(','))

for tf in pos_count:
    print(tf, pos_count[tf], edge_count[tf])