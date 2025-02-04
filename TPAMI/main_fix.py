import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import linear_model
from finch import FINCH
import numpy as np
from tslearn.metrics import dtw
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx
import dgl
import torch as th
from torch import nn
from torch.nn import functional as F
from dgl.nn import SAGEConv, GATConv, ChebConv, SGConv, PNAConv, GMMConv, TAGConv
from datetime import datetime
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler


# from torch_geometric.data import Data


def load_data(path_to_data=None, hasDate=True):
    if hasDate:
        data = pd.read_csv(path_to_data, index_col=0, low_memory=False)
        data.index = [pd.to_datetime(t) for t in data.index]
    else:
        data = pd.read_csv(path_to_data, low_memory=False)
    data.fillna(0, inplace=True)

    index = data.index
    cols = data.columns

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    data = pd.DataFrame(data=data, index=index, columns=cols)

    return data


def reshape_sequence(subsequence, lag):
    mat = sliding_window_view(subsequence, lag)[::1, :]
    X, y = mat[:, 0:-1], mat[:, -1]
    return X, y


def get_subsequences(data, time_interval):
    return data.loc[time_interval, :]


def get_representative(subsequences, models, lag):
    score = {}
    rep = None
    for i, model1 in enumerate(models):
        tmp = np.zeros(len(list(subsequences)))
        for j, name2 in enumerate(list(subsequences)):
            seq2 = subsequences[name2]
            X2, y2 = reshape_sequence(seq2, lag)
            y2_hat = model1.predict(X2)
            error = dtw(y2, y2_hat)
            tmp[j] = error
        score[i] = tmp.mean()
    for k in score:
        if score[k] == min(score.values()):
            rep = (models[k], list(subsequences)[k], score[k])
            break
    return rep


def get_next_time_interval(whole_times, current_time_interval, moving_step):
    return whole_times[whole_times.index(current_time_interval[-1]) + 1:whole_times.index(
        current_time_interval[-1]) + moving_step + 1]


def pattern_crosscheck(current_models, subsequences, lag, step, model_name='linear'):
    residual = {}
    if model_name == 'linear':
        voters = {}
        delta = []
        for name in list(subsequences):
            best = ''
            for mod in current_models:
                # if step-1 == current_models[mod]['steps'][-1]:
                # if len(current_models[mod]['voters']) == step:
                model = current_models[mod]['model']
                thresh = current_models[mod]['threshold']
                if mod not in voters:
                    voters[mod] = []
                seq = list(current_models[mod]['representative_sequence_value'])
                seq.extend(list(subsequences[name].values))
                X, y = reshape_sequence(np.array(seq), lag)
                pred = model.predict(X)
                error = dtw(y, pred)
                if error <= thresh:
                    if best == '':
                        best = (mod, error)
                    else:
                        if error < best[1]:
                            best = (mod, error)
            if best == '':
                delta.append(name)
            else:
                voters[best[0]].append(name)

        for mod in current_models:
            if mod in voters and len(voters[mod]) > 0:
                current_models[mod]['voters'].append(voters[mod])
                current_models[mod]['steps'].append(step)

        if len(delta) > 0:
            residual = subsequences.loc[:, delta]
    return residual


def get_patterns(subsequences, lag, step, model_name='linear', current_models={}):
    patterns_hat = {}
    if model_name == 'linear':
        for name in list(subsequences):
            seq = subsequences[name].values
            X, y = reshape_sequence(seq, lag)
            model = linear_model.LinearRegression()
            model.fit(X, y)
            patterns_hat[name] = model
    dico = {}
    for name, model in patterns_hat.items():
        dico[name] = model.coef_
    data = pd.DataFrame(dico).T
    clusters = FINCH(data.values)[0]
    classes = {}
    classes_model = {}
    for i, index in enumerate(clusters[:, -1]):
        if index in classes:
            classes[index].append(data.index[i])
            classes_model[index].append(patterns_hat[data.index[i]])
        else:
            classes[index] = [data.index[i]]
            classes_model[index] = [patterns_hat[data.index[i]]]

    if len(current_models) == 0:
        for id, models in classes_model.items():
            subdata = subsequences.loc[:, classes[id]]
            result = get_representative(subdata, models, lag)
            mod = 'model' + str(id + 1)
            current_models[mod] = {}
            current_models[mod]['model'] = result[0]
            current_models[mod]['representative_sequence_name'] = result[1]
            current_models[mod]['representative_sequence_value'] = subsequences[result[1]].values
            current_models[mod]['voters'] = [classes[id]]
            current_models[mod]['threshold'] = result[2]
            current_models[mod]['steps'] = [step]
    else:
        for id, models in classes_model.items():
            subdata = subsequences.loc[:, classes[id]]
            result = get_representative(subdata, models, lag)
            mod = 'model' + str(len(current_models) + 1)
            current_models[mod] = {}
            current_models[mod]['model'] = result[0]
            current_models[mod]['representative_sequence_name'] = result[1]
            current_models[mod]['representative_sequence_value'] = subsequences[result[1]].values
            current_models[mod]['voters'] = [classes[id]]
            current_models[mod]['threshold'] = result[2]
            current_models[mod]['steps'] = [step]

    return current_models


def pattern_identification(co_evolving_data=None, model_name='linear', first_time_interval=None, moving_step=None,
                           lag=None):
    subsequences = get_subsequences(co_evolving_data, first_time_interval)
    time_intervals = [first_time_interval]
    # Initialization
    step = 1
    patterns = get_patterns(subsequences, lag, step, model_name=model_name)

    # Updating over time
    while first_time_interval[-1] < co_evolving_data.index[-1]:
        first_time_interval = get_next_time_interval(list(co_evolving_data.index),
                                                     current_time_interval=first_time_interval, moving_step=moving_step)
        time_intervals.append(first_time_interval)
        subsequences = get_subsequences(co_evolving_data, first_time_interval)
        step += 1
        residual = pattern_crosscheck(patterns, subsequences, lag, step, model_name='linear')
        if len(residual) > 0:
            # print(residual)
            patterns = get_patterns(residual, lag=lag, step=step, model_name='linear', current_models=patterns)
    nbre_steps = 0
    for mod in patterns:
        patterns[mod]['time_intervals'] = []
        if max(patterns[mod]['steps']) > nbre_steps:
            nbre_steps = max(patterns[mod]['steps'])
        for step in patterns[mod]['steps']:
            patterns[mod]['time_intervals'].append(time_intervals[step - 1])
    return patterns, nbre_steps


def get_edge_weight(model, mutual_seq, init_seq, lag):
    errors = 0
    for seq1 in mutual_seq:
        seq = init_seq
        seq.extend(seq1)
        X, y = reshape_sequence(seq, lag)
        pred = model.predict(X)
        dist = dtw(pred, y)
        errors += np.exp(-dist)
    return errors


def get_node_features(step, patterns_, co_evolving_data, node):
    header = list(co_evolving_data)
    vec = np.zeros(len(header))
    if step in patterns_[node]['steps']:
        for voter in patterns_[node]['voters'][patterns_[node]['steps'].index(step)]:
            vec[header.index(voter)] = 1
    return vec


def causal_graph(patterns_, number_steps, lag, co_evolving_data=None):
    list_graphs = {}
    nodes = [i for i in range(2 * len(patterns_))]

    for step1, step2 in zip(range(1, number_steps), range(2, number_steps + 1)):
        list_graphs[str(step1) + '-' + str(step2)] = {}
        weights = []
        models = list(patterns_.keys())
        from_ = []
        to_ = []
        for mod1 in models:
            for mod2 in models:
                voters1, voters2 = [], []
                t = []
                if step1 in patterns_[mod1]['steps']:
                    voters1.extend(patterns_[mod1]['voters'][patterns_[mod1]['steps'].index(step1)])
                if step2 in patterns_[mod2]['steps']:
                    voters2.extend(patterns_[mod2]['voters'][patterns_[mod2]['steps'].index(step2)])
                    t.extend(patterns_[mod2]['time_intervals'][patterns_[mod2]['steps'].index(step2)])
                mutual_voters = list(set(voters1).intersection(set(voters2)))
                mutual_sequences = []
                if len(mutual_voters) > 0:
                    from_.append(int(mod1.replace('model', '')) - 1)
                    to_.append(int(mod2.replace('model', '')) - 1 + len(patterns_))
                    for name in mutual_voters:
                        mutual_sequences.append(list(co_evolving_data[name][t].values))
                    model = patterns_[mod2]['model']
                    init_seq = list(patterns_[mod2]['representative_sequence_value'])
                    weights.append(get_edge_weight(model, mutual_sequences, init_seq, lag))

        graph_frame = {}
        graph_frame['from'] = from_
        graph_frame['to'] = to_
        graph_frame['weight'] = weights
        list_graphs[str(step1) + '-' + str(step2)]['frame'] = pd.DataFrame(graph_frame)

        nxgr = nx.DiGraph()
        nxgr.add_nodes_from(nodes)
        nxgr.add_edges_from(
            [(u, v, {'weight': w}) for u, v, w in zip(graph_frame['from'], graph_frame['to'], graph_frame['weight'])])
        list_graphs[str(step1) + '-' + str(step2)]['nx_graph'] = nxgr
        dglgr = dgl.from_networkx(nxgr, edge_attrs=['weight'])
        features = np.zeros((len(patterns_) * 2, len(list(co_evolving_data))))
        for node in graph_frame['from']:
            features[node] = get_node_features(step1, patterns_, co_evolving_data, 'model' + str(node + 1))
        for node in graph_frame['to']:
            features[node] = get_node_features(step1, patterns_, co_evolving_data,
                                               'model' + str(node % len(patterns_) + 1))

        dglgr.ndata['features'] = th.from_numpy(features)
        list_graphs[str(step1) + '-' + str(step2)]['dgl_graph'] = dglgr
        # pygr = from_networkx(nxgr, group_node_attrs=[list(features[i,:]) for i in range(features.shape[0])], group_edge_attrs=weights)
        # list_graphs[str(step1) + '-' + str(step2)]['pyg_graph'] = pygr
        # list_graphs.append(gr)
        # list_graphs.append(pd.DataFrame(graph_frame))
    return list_graphs


def view_pattern(patterns_, number_steps, title1, title2, save_pattern_as):
    fr = {}
    fr['Timestamps'] = ['T' + str(i + 1) for i in range(number_steps)]
    print(number_steps)
    for mod in patterns_:
        fr[mod] = np.zeros(number_steps)
        # for step in patterns_[mod]['steps']:
        #     fr[mod][]
        for i, step in enumerate(patterns_[mod]['steps']):
            fr[mod][step - 1] = len(patterns_[mod]['voters'][i])
    fr = pd.DataFrame(fr)
    k = 3
    models = []
    for mod in list(fr):
        models.append(mod + '_')
        vec = []
        vec2 = []
        for el in fr[mod]:
            if el == 0:
                vec.append(np.nan)
                vec2.append(np.nan)
            else:
                vec.append(el)
                vec2.append(k)
        k += 1
        fr[mod] = vec
        fr[mod + '_'] = vec2
    fr.set_index('Timestamps', inplace=True)
    fig = plt.figure(dpi=150, figsize=(32, 4))
    axe1 = fig.add_subplot(121)
    sns.lineplot(data=fr.loc[:, [col for col in list(fr) if col not in models]], ax=axe1, marker='o', legend=False,
                 markersize=3)
    axe1.grid(True)
    axe1.set_title(title1, fontsize=18)
    axe1.set_ylabel('# exhibited sequences', fontsize=16)
    axe1.set_xticklabels(['T' + str(i + 1) for i in range(number_steps)], fontsize=8, rotation=90)

    axe2 = fig.add_subplot(122)
    sns.lineplot(data=fr.loc[:, models], ax=axe2, legend=False, marker='o', lw=2, alpha=1, markersize=3)
    axe2.grid(True)
    axe2.set_title(title2, fontsize=18)
    axe2.set_ylabel('Model index', fontsize=16)
    axe2.set_xticklabels(['T' + str(i + 1) for i in range(number_steps)], fontsize=8, rotation=90)
    plt.savefig(f'{save_pattern_as}.pdf', bbox_inches='tight')
    plt.show()


def view_causal_relations(patterns_, list_graphs, save_file_as):
    total_edges = []
    fig = plt.figure(figsize=(52, 8), dpi=90)
    cpt = 1
    for step in list_graphs:
        axe = fig.add_subplot(1, len(list_graphs), cpt)
        print(step, list_graphs[step]['nx_graph'], list_graphs[step]['dgl_graph'])
        G = list_graphs[step]['nx_graph']
        nx.draw_networkx(
            G,
            pos=nx.drawing.layout.bipartite_layout(G, [i for i in range(len(patterns_))]),
            width=1,
            ax=axe,
            node_size=350
        )
        total_edges.append(G.number_of_edges())
        t1 = step.split('-')[0]
        t2 = str(int(t1) + 1)
        axe.set_title(r'$G_{'+t1+','+t2+'}$', fontsize=16)
        # pos = nx.bipartite_layout(G, 'vertical')
        # nx.draw(G, pos=pos)
        # plt.show()
        cpt += 1
    print(sum(total_edges))
    plt.savefig(f'{save_file_as}.pdf', bbox_inches='tight')
    plt.show()


# def view_causal_relations(patterns_, list_graphs, save_file_as):
#     fig = plt.figure(figsize=(52, 8), dpi=90)
#     cpt = 1
#     for step in list_graphs:
#         axe = fig.add_subplot(1, len(list_graphs), cpt)
#         print(step, list_graphs[step]['nx_graph'], list_graphs[step]['dgl_graph'])
#         G = list_graphs[step]['nx_graph']
#         nx.draw_networkx(
#             G,
#             pos=nx.drawing.layout.bipartite_layout(G, [i for i in range(len(patterns_))]),
#             width=3,
#             ax=axe
#         )
#         axe.set_title(step)
#         # pos = nx.bipartite_layout(G, 'vertical')
#         # nx.draw(G, pos=pos)
#         # plt.show()
#         cpt += 1
#     plt.savefig(f'{save_file_as}.pdf', bbox_inches='tight')
#     plt.show()


class encoder(nn.Module):
    def __init__(self, in_feats, h_feats, k, dropout=0, allow_zero_in_degree=True):
        super(encoder, self).__init__()
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(TAGConv(in_feats, h_feats[0], k))

        for i in range(1, len(h_feats)):
            self.conv_layers.append(TAGConv(h_feats[i - 1], h_feats[i], k))
        # self.conv_layers.append(TAGConv(in_feats*lag, h_feats[-1], k))

    def forward(self, list_g, list_features):
        embeddings = []
        for g, X in zip(list_g, list_features):
            h = X
            for i, layer in enumerate(self.conv_layers):
                h = layer(g, h)
                # if i < len(self.conv_layers) - 1:
                h = F.dropout(F.relu(h), p=self.dropout)
            embeddings.append(h)
        # res = F.dropout(F.relu(th.cat(embeddings, 1)), p=self.dropout)
        return embeddings


class decoder(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=.87):
        super(decoder, self).__init__()
        self.decod_layers = nn.ModuleList()
        self.dropout = dropout
        self.decod_layers.append(nn.Linear(in_feats, h_feats[0], bias=False))
        for i in range(1, len(h_feats)):
            self.decod_layers.append(nn.Linear(h_feats[i - 1], h_feats[i]))

    def forward(self, list_embeddings):
        h = th.cat(list_embeddings, 1)
        for i, layer in enumerate(self.decod_layers):
            h = layer(h)
            h = F.dropout(F.relu(h), p=self.dropout)
        return F.softmax(h, dim=1)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()


class silhouette_loss:

    def __init__(self, X, labels, device='cpu', dtype=th.float):

        self.X = X
        self.labels = labels
        self.unique_labels = th.unique(labels)
        self.device = device
        self.dtype = dtype

    def score(self):

        """Compute the mean Silhouette Coefficient of all samples.
        The Silhouette Coefficient is calculated using the mean intra-cluster
        distance (a) and the mean nearest-cluster distance (b) for each sample.
        The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
        To clarrify, b is the distance between a sample and the nearest cluster
        that b is not a part of.
        This function returns the mean Silhoeutte Coefficient over all samples.
        The best value is 1 and the worst value is -1. Values near 0 indicate
        overlapping clusters. Negative values generally indicate that a sample has
        been assigned to the wrong cluster, as a different cluster is more similar.

	Code developed in NumPy by Alexandre Abraham:
	https://gist.github.com/AlexandreAbraham/5544803  Avatar
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
                 label values for each sample
        loss : Boolean
                If True, will return negative silhouette score as
                torch tensor without moving it to the CPU. Can therefore
                be used to calculate the gradient using autograd.
                If False positive silhouette score as float
                on CPU will be returned.
        Returns
        -------
        silhouette : float
            Mean Silhouette Coefficient for all samples.
        References
        ----------
        Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
            Interpretation and Validation of Cluster Analysis". Computational
            and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
        http://en.wikipedia.org/wiki/Silhouette_(clustering)
        """

        A = self._intra_cluster_distances_block()
        B = self._nearest_cluster_distance_block()
        sil_samples = (B - A) / th.maximum(A, B)

        mean_sil_score = th.mean(th.nan_to_num(sil_samples))
        return (1 - mean_sil_score) / 2.

    # @staticmethod
    def _intra_cluster_distances_block_(self, subX):
        distances = th.cdist(subX, subX)
        return distances.sum(axis=1) / (distances.shape[0] - 1)

    # @staticmethod
    def _nearest_cluster_distance_block_(self, subX_a, subX_b):
        dist = th.cdist(subX_a, subX_b)
        dist_a = dist.mean(axis=1)
        dist_b = dist.mean(axis=0)
        return dist_a, dist_b

    def _intra_cluster_distances_block(self):
        """Calculate the mean intra-cluster distance.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        Returns
        -------
        a : array [n_samples_a]
            Mean intra-cluster distance
        """
        intra_dist = th.zeros(self.labels.size(), dtype=self.dtype,
                              device=self.device)
        values = [self._intra_cluster_distances_block_(
            self.X[th.where(self.labels == label)[0]])
            for label in self.unique_labels]
        for label, values_ in zip(self.unique_labels, values):
            intra_dist[th.where(self.labels == label)[0]] = values_
        return intra_dist

    # @staticmethod
    def _nearest_cluster_distance_block(self):
        """Calculate the mean nearest-cluster distance for sample i.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        X : array [n_samples_a, n_features]
            Feature array.
        Returns
        -------
        b : float
            Mean nearest-cluster distance for sample i
        """
        inter_dist = th.full(self.labels.size(), 1000000000,
                             dtype=self.dtype,
                             device=self.device)
        # Compute cluster distance between pairs of clusters

        label_combinations = th.combinations(self.unique_labels, 2)

        values = [self._nearest_cluster_distance_block_(
            self.X[th.where(self.labels == label_a)[0]],
            self.X[th.where(self.labels == label_b)[0]])
            for label_a, label_b in label_combinations]

        for (label_a, label_b), (values_a, values_b) in \
                zip(label_combinations, values):
            indices_a = th.where(self.labels == label_a)[0]
            inter_dist[indices_a] = th.minimum(values_a, inter_dist[indices_a])
            del indices_a
            indices_b = th.where(self.labels == label_b)[0]
            inter_dist[indices_b] = th.minimum(values_b, inter_dist[indices_b])
            del indices_b
        return inter_dist


class reconstruction_loss:

    def __init__(self, true_weights, generated_weights, true_attributes, generated_attributes):
        self.true_weights = true_weights
        self.generated_weights = generated_weights
        self.true_attributes = true_attributes
        self.generated_attributes = generated_attributes

    def score(self):
        pass


def main(save_as, save_pattern_as, path_to_data=None, size=42, moving_step=30, lag=3, view_pattern_lifespan=False,
         view_causal_graph=False,
         containsDate=True):
    data = load_data(path_to_data, hasDate=containsDate)
    time = data.index
    first_time_interval = time[:size]
    patterns, nbre_steps = pattern_identification(co_evolving_data=data, model_name='linear',
                                                  first_time_interval=first_time_interval,
                                                  moving_step=moving_step, lag=lag)
    evolving_graphs = causal_graph(patterns, nbre_steps, lag, data)

    # coder = encoder(data.shape[1], [32, 15], 3, .85)
    # print(coder)
    # list_g = []
    # list_feat = []
    # for i in range(5):
    #     gr = evolving_graphs[str(i+1)+'-'+str(i+2)]['dgl_graph']
    #     feat = gr.ndata['features'].to(device='cpu', dtype=th.float)
    #     list_g.append(gr)
    #     list_feat.append(feat)
    #
    # decod = decoder(15*5, [200, 130, 76])
    # print(decod)
    # list_embeddings = coder(list_g, list_feat)
    #
    # print(list_embeddings)
    #
    # reconstructed_feat = decod(list_embeddings)
    #
    # print(reconstructed_feat)

    if view_pattern_lifespan:
        for mod in patterns:
            print(mod)

            for i, voters in enumerate(patterns[mod]['voters']):
                print(patterns[mod]['steps'][i], len(voters), end=',')
            print('\n -----------------')

        equation = r'$'
        for i in range(lag):
            if i == 0:
                equation += 'a_' + str(i)
            else:
                equation += ' + a_' + str(i) + 'x_' + str(i)
        equation += '$'
        title1 = 'Patterns lifespan with respect to\n the number of sequences exhibited\n' + 'linear model: ' + equation
        title2 = 'Patterns lifespan with respect to\n the time duration\n' + 'linear model: ' + equation
        view_pattern(patterns, nbre_steps, title1, title2, save_pattern_as=save_pattern_as)

    if view_causal_graph:
        view_causal_relations(patterns, evolving_graphs, save_file_as=save_as)

    # print('total number of models: ', len(patterns))
    # for step in evolving_graphs:
    #     print(step, evolving_graphs[step]['nx_graph'], evolving_graphs[step]['dgl_graph'])
    #     G = evolving_graphs[step]['nx_graph']
    #     nx.draw_networkx(
    #         G,
    #         pos=nx.drawing.layout.bipartite_layout(G, [i for i in range(len(patterns))]),
    #         width=3)
    #     # pos = nx.bipartite_layout(G, 'vertical')
    #     # nx.draw(G, pos=pos)
    #     plt.show()


path_to_data = dict(
    it_sp="/Users/owup2301/Documents/onCourse/taxi/co-evolve/IT-SP_latest.csv",
    sp500="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/SP500stocks-2023-03-30.csv",
    etf2="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/Columbus_ETFs-2023-03-30_(no Na).csv",
    eeg="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/eeg.csv",
    energy="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/enernoc.csv",
    earthquake="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/eqd.csv",
    apartment="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/apartment.csv",
    gps="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/gps_data.csv",
    sds="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/SDS1.csv",
    fx="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/FXs_interpolated.csv",
    index="/Users/owup2301/Documents/onCourse/taxi/etienne/datasets/world_indices_reduced.csv",

    ##############################################################################################

    weather = '/home/local/USHERBROOKE/owup2301/Téléchargements/all_six_datasets/all_six_datasets/'
                  'weather/weather.csv',
    traffic = '/home/local/USHERBROOKE/owup2301/Téléchargements/all_six_datasets/all_six_datasets/'
                'traffic/traffic.csv',
    ETTh1 = '/home/local/USHERBROOKE/owup2301/Téléchargements/all_six_datasets/all_six_datasets/ETT-small/ETTh1.csv',
    ETTh2 = '/home/local/USHERBROOKE/owup2301/Téléchargements/all_six_datasets/all_six_datasets/ETT-small/ETTh2.csv',
    ETTm1 = '/home/local/USHERBROOKE/owup2301/Téléchargements/all_six_datasets/all_six_datasets/ETT-small/ETTm1.csv',
    ETTm2 = '/home/local/USHERBROOKE/owup2301/Téléchargements/all_six_datasets/all_six_datasets/ETT-small/ETTm2.csv',
)

size = 500  # 42 672
moving_step = 50 # 30 168
lag = 3
view_pattern_lifespan = True
view_causal_graph = True
data_name = 'weather'

if __name__ == '__main__':
    start = datetime.now()
    main(
        containsDate=True,
        path_to_data=path_to_data[data_name],
        save_as=f"{data_name}_causal_graph_fx_{size}_{moving_step}",
        save_pattern_as=f"{data_name}_pattern_graph_fx_{size}_{moving_step}",
        size=size,
        moving_step=moving_step,
        lag=lag,
        view_pattern_lifespan=view_pattern_lifespan,
        view_causal_graph=view_causal_graph
    )
    print(datetime.now()-start)
