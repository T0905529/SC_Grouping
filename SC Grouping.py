import numpy as np
import pandas as pd
import torch as th
import math
import h5py
from sklearn.cluster import AgglomerativeClustering as sk_cluster
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import euclidean_distances as eu_dist
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def preprocess():
    df = pd.read_csv('e95_seurat_normalized_data.csv')
    df.set_index(['Unnamed: 0'], inplace=True)
    df.index.name = 'Gene'

    lr = pd.read_excel('ligand_receptor_E9.5_E10.5.xlsx', 'E9.5')[['Ligand', 'Receptor']]
    ligands = list(lr['Ligand'].apply(lambda x: str(x).lower()))
    receptors = list(lr['Receptor'].apply(lambda x: str(x).lower()))

    for gene in df.index:
        if str(gene).lower() not in ligands and str(gene).lower() not in receptors:
            df.drop(labels=gene, inplace=True)

    print(len(df.index))
    df.to_excel('Single Cell - Ligands + Receptors 9.5.xlsx')


def gen_conn():
    df = pd.read_excel('Single Cell - Ligands + Receptors 10.5.xlsx')
    lr = pd.read_excel('ligand_receptor_E9.5_E10.5.xlsx', 'E10.5')[['Ligand', 'Receptor']]
    df.index = df['Gene'].str.lower()
    df.drop(columns=['Gene'], inplace=True)
    df.columns = df.columns.str.lower()
    df.drop(index=['ncam1', 'hhip'], inplace=True)  # both show up as ligands and receptors. This was causing non-symmetry in the resulting connection matrix. You can imagine this wasn't easy to notice
    lr['Ligand'] = lr['Ligand'].str.lower()
    lr['Receptor'] = lr['Receptor'].str.lower()

    ltor = {ligand : [receptor for receptor in lr.loc[lr['Ligand'] == ligand]['Receptor'] if receptor not in ['ncam1', 'hhip']] for ligand in lr['Ligand'] if ligand not in ['ncam1', 'hhip']}
    rtol = {receptor : [ligand for ligand in lr.loc[lr['Receptor'] == receptor]['Ligand'] if ligand not in ['ncam1', 'hhip']] for receptor in lr['Receptor'] if receptor not in ['ncam1', 'hhip']}

    # for row in df.index:
    #     for col in df.columns:
    #         if float(df.loc[row, col]) > 0:
    #             df.loc[row, col] = 1
    #         else:
    #             df.loc[row, col] = 0

    num_cells = len(df.columns)
    conn = np.zeros((num_cells, num_cells))

    ls = list(lr['Ligand'])
    rs = list(lr['Receptor'])
    for row in df.index:
        ligand = row in ls
        receptor = row in rs
        assert ligand or receptor
        col_mod = np.expand_dims(np.array(df.loc[row]), axis=1)
        if ligand:
            row_mod = np.array(df.loc[ltor[row]]).sum(axis=0)
        if receptor:
            row_mod = np.array(df.loc[rtol[row]]).sum(axis=0)

        conn_mod = np.stack([row_mod] * num_cells, axis=0) * col_mod
        conn += conn_mod

    np.fill_diagonal(conn, 0)
    np.save('Single Cell Connections - 10.5', conn, fix_imports=False)


def grab_rows(graph, clusters, c1, c2):
    access = ((clusters[graph['Src'].astype(int)] == c1) & (clusters[graph['Tgt'].astype(int)] == c2)) | ((clusters[graph['Src'].astype(int)] == c2) & (clusters[graph['Tgt'].astype(int)] == c1))
    return graph[access]['Weight'].sum() / 100


def weight(conn, clusters, row, col):
    if clusters[row] == clusters[col]:
        return conn[row, col]
    else:
        return 1e-5


def write_graph():
    columns = ['Src', 'Tgt', 'Weight']
    conn = np.load('Single Cell Connections - 9.5.npy', fix_imports=False)
    thresh = conn.mean() + 3.5 * conn.std()
    clusters = np.load('True Labels 9.5.npy', fix_imports=False)
    full_graph = pd.DataFrame(columns=columns)
    lim_graph = full_graph.__deepcopy__()

    for row in range(conn.shape[0]):
        print(row)
        edges = pd.DataFrame([[row, col, weight(conn, clusters, row, col)] for col in range(conn.shape[1]) if col > row and conn[row, col] >= thresh], columns=columns)
        lim_graph = lim_graph.append(edges, ignore_index=True)
        edges = pd.DataFrame([[row, col, conn[row, col]] for col in range(conn.shape[1]) if col > row and conn[row, col] >= thresh], columns=columns)
        full_graph = full_graph.append(edges, ignore_index=True)

    added = set()
    for i in range(clusters.max() + 1):
        edges = pd.DataFrame([['C' + str(i), 'C' + str(j), grab_rows(full_graph, clusters, i, j) + 1e-5]
                              for j in range(clusters.max() + 1) if (i, j) not in added and (j, i) not in added and i != j], columns=columns)
        lim_graph = lim_graph.append(edges, ignore_index=True)
        added.update([(i, j) for j in range(clusters.max() + 1) if i != j])

    edges = pd.DataFrame([[row, 'C' + str(clusters[row]), 100] for row in range(conn.shape[0])], columns=columns)
    lim_graph = lim_graph.append(edges, ignore_index=True)


    print(len(lim_graph))
    lim_graph.to_csv('Single Cell Visualization/9.5.2.csv')
    # pd.DataFrame(clusters).to_excel('Single Cell Visualization/True Labels 10.5.xlsx')


def gen_pos():
    th.set_printoptions(linewidth=150)
    conn = np.load('Single Cell Connections - 9.5.npy', fix_imports=False)
    conn = th.from_numpy(conn).cuda().type(dtype=th.float)
    tgt = conn / conn.max()  # normalize
    tgt = -th.log(tgt)  # values now between 0 and 1, cast as distances
    tgt = tgt.fill_diagonal_(0)  # negative log should cast diagonal (0) to high value, replace with 0
    tgt = tgt.clamp(max=100)  # remove any inf values

    pos = th.randn((1, conn.size(0), 2), dtype=th.float, device='cuda')
    weight = 1 / (tgt ** 2)
    weight[weight.isnan()] = 0
    weight[weight.isinf()] = 0
    lr_max = 1 / weight.max()
    gamma = .1
    losses = []
    for t in range(100):
        lr = weight * (lr_max * math.exp(-gamma * t))
        square = pos.transpose(0, 1).repeat(1, conn.size(0), 1)
        diff = square - pos
        mag = th.sqrt((diff ** 2).sum(dim=2))
        gradient = ((mag - tgt) / (2 * mag)).unsqueeze(2) * diff
        gradient[gradient.isnan()] = 0
        updates = th.triu(gradient.permute(2, 0, 1) * lr.unsqueeze(0), diagonal=1)
        updates = updates.permute(1, 2, 0)
        update = updates.sum(dim=0) - updates.sum(dim=1)
        pos += update
        losses.append(float((th.abs(mag - tgt)).sum()))

    pos = pos.squeeze()
    np.save('Cell Positions', pos.cpu().numpy(), fix_imports=False)
    domain = np.arange(0, len(losses))
    plt.plot(domain, losses)
    plt.show()


def count_inter():
    a = np.where(load_region(num=3) == 1, np.ones_like(load_region(num=3)), np.zeros_like(load_region(num=3))).sum()
    m = np.where(load_region(num=3) == 2, np.ones_like(load_region(num=3)), np.zeros_like(load_region(num=3))).sum()
    p = np.where(load_region(num=3) == 3, np.ones_like(load_region(num=3)), np.zeros_like(load_region(num=3))).sum()
    ap_m = a * m + p * m
    a_p = a * p
    print(ap_m)  # = 542,956
    print(a_p)  # = 805,272
    print(a_p - ap_m)  # = 262,316


def display_gauss(std):
    inp = np.linspace(0, 100, num=100)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-1/2 * (inp / std) ** 2)
    plt.plot(inp, y)
    plt.show()


def sim_to_dist(x, max):
    x = (x - x.min()) / (x.max() - x.min())
    x = -np.tan(x + np.pi / 2)
    x = np.clip(x, 0, max)
    np.fill_diagonal(x, 0)
    return x


def load_region(num):
    y = pd.read_csv('E9.5_seurat_normalized_data_with_region_and_cluster.csv', low_memory=False)
    y.set_index(['Unnamed: 0'], inplace=True)
    region = y.loc['Region']
    if num == 3:
        region[region == 'A'] = 1
        region[region == 'M'] = 2
        region[region == 'P'] = 3
        return np.expand_dims(region.to_numpy(dtype=np.int32), axis=1)
    if num == 6:
        region[:377] = 1
        region[377:754] = 2
        region[754:903] = 3
        region[903:1052] = 4
        region[1052:1586] = 5
        region[1586:2120] = 6
        return np.expand_dims(region.to_numpy(dtype=np.str_), axis=1)


# Why MDS?
def mds():
    x = np.load('Single Cell Connections - 9.5.npy')
    similarity = x.copy()

    # x = pd.read_excel('Single Cell - Ligands + Receptors 9.5.xlsx')
    # x.drop(columns=['Gene'], inplace=True)
    # x = x.to_numpy().T
    # x = eu_dist(x, x)

    x = sim_to_dist(x, 100)
    delta = 1 + .01 * np.random.randn(x.shape[0], x.shape[0])
    delta[delta < 0] = 0

    ## GROUPING
    # RANDOM - OFFSET
    # clusters = load_region(num=3)  # np.random.randint(1, high=4, size=(x.shape[0], 1))
    # overlap = np.equal(clusters.T, clusters)
    # weight = np.where(overlap, np.full_like(overlap, 1, dtype=np.float32), np.full_like(overlap, 1e-8, dtype=np.float32))
    # CLUSTERING
    # clusters = sk_cluster(n_clusters=3, affinity='precomputed', linkage='complete').fit_predict(x)
    # clusters = np.expand_dims(clusters, 1)
    # overlap = np.equal(clusters.T, clusters)
    # grouping = np.where(overlap, np.full_like(overlap, .5, dtype=np.float32), np.full_like(overlap, 2, dtype=np.float32))

    ## OFFSET
    factor = 4.0
    weight = load_region(num=3)
    weight = weight.astype(np.float32)
    weight = weight.T * weight
    weight[(weight == 1) | (weight == 4) | (weight == 9)] = 1
    weight[(weight == 2) | (weight == 6)] = 1 / factor
    weight[(weight == 3)] = 1 / (2 * factor)
    # weight = np.char.add(weight.T, weight)
    # weight = weight.astype(np.float32)
    # weight[(weight == 11) | (weight == 22) | (weight == 33) | (weight == 44) | (weight == 55) | (weight == 66)] = 1
    # weight[(weight == 12) | (weight == 21) | (weight == 34) | (weight == 43) | (weight == 56) | (weight == 65)] = .8
    # weight[(weight == 13) | (weight == 31) | (weight == 23) | (weight == 32) | (weight == 14) | (weight == 41) | (weight == 24) | (weight == 42)] = .5
    # weight[(weight == 53) | (weight == 35) | (weight == 63) | (weight == 36) | (weight == 54) | (weight == 45) | (weight == 64) | (weight == 46)] = .5
    # weight[(weight == 15) | (weight == 51) | (weight == 25) | (weight == 52) | (weight == 16) | (weight == 61) | (weight == 26) | (weight == 62)] = .25
    weight *= delta

    ## GAUSSIAN
    # std = 2.0
    # x = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-1 / 2 * (x / std) ** 2)
    # x = x * similarity
    # x = sim_to_dist(x, 500)

    ## GROUPING
    # x = x * grouping

    ## THRESHOLDING
    # thresh = similarity.mean() + -.6 * similarity.std()
    # x[similarity < thresh] = 0
    # print((np.where(x == 0, np.ones_like(x), np.zeros_like(x)).sum() - 2120) / (2120 ** 2))

    mds = MDS(2, metric=True, weight=weight, dissimilarity='precomputed', max_iter=100, n_init=1)
    mds.fit(x)
    x = mds.embedding_

    ## OFFSET
    # offset = np.random.randint(0, high=3, size=(x.shape[0], x.shape[1]))
    # offset[offset == 0] = -100
    # offset[offset == 1] = 0
    # offset[offset == 2] = 100
    # offset[:, 1] = np.zeros(x.shape[0])
    # x += offset

    stress = mds.stress_
    print(stress)

    clusters = np.load('True Labels 9.5.npy', fix_imports=False)
    plt.scatter(x[:, 0], x[:, 1], s=3, c=load_region(num=3), cmap='rainbow')
    plt.show()


def mds2():
    # x = np.load('Single Cell Connections - 9.5.npy')
    # x = sim_to_dist(x, 100)

    w = pd.read_csv('e95_seurat_normalized_data.csv')
    # w = pd.read_excel('Single Cell - Ligands + Receptors 9.5.xlsx')
    w.drop(columns=['Unnamed: 0'], inplace=True)
    w = w.to_numpy().T
    w = cos_sim(w)
    x = sim_to_dist(w, 100)

    weight = load_region(num=3)
    weight = weight.astype(np.float32)
    weight = weight.T * weight
    grouping = weight.copy()
    weight[(weight == 1) | (weight == 4) | (weight == 9)] = 1
    weight[(weight == 2) | (weight == 6)] = 1e-7
    weight[(weight == 3)] = 1e-7

    grouping[(grouping == 1) | (grouping == 4) | (grouping == 9)] = 0
    grouping[(grouping == 2) | (grouping == 6)] = 1
    grouping[(grouping == 3)] = 2
    x += grouping

    mds = MDS(2, metric=True, weight=weight, dissimilarity='precomputed', max_iter=100, n_init=1)
    mds.fit(x)
    x = mds.embedding_

    stress = mds.stress_
    print(stress)

    clusters = np.load('True Labels 9.5.npy', fix_imports=False)
    plt.scatter(x[:, 0], x[:, 1], s=3, c=load_region(num=3), cmap='rainbow')
    # figure, axis = plt.subplots(2, figsize=(10, 10))
    # axis[0].set_yscale('linear')
    # axis[0].scatter(x[:, 0], x[:, 1], s=3, c=load_region(num=3), cmap='rainbow')
    # axis[1].set_yscale('linear')
    # axis[1].scatter(x[:, 0], x[:, 1], s=3, c=clusters, cmap='rainbow')
    plt.show()


def multi_mds():
    x = np.load('Single Cell Connections - 9.5.npy')
    x = sim_to_dist(x, 100)

    w = pd.read_csv('e95_seurat_normalized_data.csv')
    # w = pd.read_excel('Single Cell - Ligands + Receptors 9.5.xlsx')
    w.drop(columns=['Unnamed: 0'], inplace=True)
    w = w.to_numpy().T
    w = cos_sim(w)
    # x = sim_to_dist(w, 100)
    x1 = x[:754, :754]
    x2 = x[754: 1052, 754: 1052]
    x3 = x[1052: 2120, 1052: 2120]

    mds = MDS(2, metric=True, weight=None, dissimilarity='precomputed', max_iter=100, n_init=1)
    mds.fit(x1)
    x1 = mds.embedding_
    print(mds.stress_)
    mds.fit(x2)
    x2 = mds.embedding_
    print(mds.stress_)
    mds.fit(x3)
    x3 = mds.embedding_
    print(mds.stress_)

    clusters = np.load('True Labels 9.5.npy', fix_imports=False)
    fig, ax = plt.subplots(3, figsize=(5, 9))
    ax[0].set_yscale('linear')
    ax[1].set_yscale('linear')
    ax[2].set_yscale('linear')
    ax[0].scatter(x1[:, 0], x1[:, 1], s=3, c=clusters[:754], cmap='rainbow')
    ax[1].scatter(x2[:, 0], x2[:, 1], s=3, c=clusters[754: 1052], cmap='rainbow')
    ax[2].scatter(x3[:, 0], x3[:, 1], s=3, c=clusters[1052: 2120], cmap='rainbow')
    plt.show()


def cluster_similarity():
    df = pd.read_excel('Single Cell - Ligands + Receptors 10.5.xlsx')
    df.index = df['Gene'].str.lower()
    df.drop(columns=['Gene'], inplace=True)
    df.columns = df.columns.str.lower()
    expression = np.transpose(np.array(df))
    cells = df.columns

    conn = np.load('Single Cell Connections - 10.5.npy', fix_imports=False)
    conn = th.Tensor(conn)
    tgt = (conn - conn.min()) / (conn.max() - conn.min())  # normalize
    tgt = -th.log(tgt)  # values now between 0 and 1, cast as distances
    tgt = tgt.fill_diagonal_(0)  # negative log should cast diagonal (0) to high value, replace with 0
    tgt = tgt.clamp(max=100)  # remove any inf values

    clustering_con = sk_cluster(n_clusters=10, affinity='precomputed', linkage='complete')
    clustering_con.fit(tgt)
    clustering = th.LongTensor(clustering_con.labels_)

    # gmm = GMM(n_components=10, n_init=3, init_params='kmeans')
    # gmm.fit(expression)
    # clustering = th.LongTensor(gmm.predict(expression))

    # clustering = th.LongTensor(np.load('Predicted Labels.npy', fix_imports=False))

    identity = np.array([int(cell.split('.')[0].split('_')[-1]) for cell in cells])
    np.save('True Labels 10.5', identity, fix_imports=False)
    identity = th.LongTensor(identity)
    # clustering = th.LongTensor(np.load('Predicted Labels.npy', fix_imports=False))

    similarity = np.zeros((10, 10))
    differences = np.zeros((10, 10))
    for i in range(10):
        labels = th.where(clustering == i, th.ones(clustering.size()), th.zeros(clustering.size()))
        for j in range(10):
            true = th.where(identity == j, th.ones(identity.size()), th.zeros(identity.size()))
            assert labels.size() == true.size()
            overlap = th.where(labels + true == 2, th.ones(labels.size()), th.zeros(labels.size()))
            similarity[i, j] = (overlap.sum() / th.max(labels.sum(), true.sum())).item()
            differences[i, j] = (true.sum() - labels.sum()).item()

    # print(similarity)
    print(similarity.max(axis=1))
    print(np.arange(0, 10))
    print(similarity.argmax(axis=1))
    print('-------------------------------')
    print(np.squeeze(np.take_along_axis(differences, np.expand_dims(similarity.argmax(axis=1), axis=1), axis=1)))


def dendrogram():
    identity = np.load('True Labels 10.5.npy', fix_imports=False)
    dist = np.load('Log Distance 10.5.npy', fix_imports=False)

    dist = squareform(dist)
    z = hierarchy.linkage(dist, 'ward')
    hierarchy.dendrogram(z) #, p=200, truncate_mode='level')
    plt.title('Ward Dendrogram (All Levels)')
    plt.show()


def plot_identity():  # plot cells given the positions (was for force directed algorithm prediciting locations) (identity here is how the true labels are created)
    cells = list(pd.read_excel('Single Cell - Ligands + Receptors 9.5.xlsx').columns)
    cells.remove('Gene')
    pos = np.load('Cell Positions.npy', fix_imports=False)

    identity = np.array([int(cell.split('.')[0].split('_')[-1]) for cell in cells], dtype=np.int32)
    plt.scatter(pos[:, 0], pos[:, 1], s=3, c=identity, cmap='rainbow')
    plt.show()


def main():
    # dendrogram()
    # write_graph()
    mds2()
    # multi_mds()


main()
