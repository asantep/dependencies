import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import linear_model
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
from finch import FINCH
import numpy as np
from tslearn.metrics import dtw
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx
import dgl
import torch as th
from sklearn.preprocessing import StandardScaler
import pickle
import json


def load_data(path_to_data=None, normalize=True):
    data = pd.read_csv(path_to_data, index_col=0)

    data.index = [pd.to_datetime(t) for t in data.index]
    data.fillna(0, inplace=True)
    if normalize == True:
        scaler = StandardScaler()
        data2 = scaler.fit_transform(data)
        data2 = pd.DataFrame(data2)
        data2.index = data.index
        data2.columns = data.columns
        data = data2

    print(data)
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


def modeling(model_name='linear', hyperparameters=None):
    the_model = None
    if model_name == 'linear':
        the_model = linear_model.LinearRegression(**hyperparameters)
    if model_name == 'svr':
        the_model = NuSVR(**hyperparameters)
    if model_name == 'mlp':
        the_model = MLPRegressor(**hyperparameters)
    return the_model


def get_coefficients(model, size=None, model_name='linear'):
    coefficients = None
    if model_name == 'linear':
        coefficients = model.coef_
    if model_name == 'svr':
        coefficients = np.zeros(size)
        for i, coef in zip(model.support_, model.dual_coef_.flatten()):
            coefficients[i] = coef
    if model_name == 'mlp':
        coefficients = []
        for mat in model.coefs_:
            coefficients.extend(list(mat.flatten()))
    return coefficients


def get_patterns(subsequences, lag, step, model_name='linear', hyperparameters=None, current_models={}):
    patterns_hat = {}

    for name in list(subsequences):
        seq = subsequences[name].values
        X, y = reshape_sequence(seq, lag)
        model = modeling(model_name=model_name, hyperparameters=hyperparameters)
        model.fit(X, y)
        patterns_hat[name] = model

    # if model_name == 'linear':
    #     for name in list(subsequences):
    #         seq = subsequences[name].values
    #         X, y = reshape_sequence(seq, lag)
    #         model = linear_model.LinearRegression()
    #         model.fit(X, y)
    #         patterns_hat[name] = model

    dico = {}
    for name, model in patterns_hat.items():
        # dico[name] = model.coef_
        dico[name] = get_coefficients(model, size=subsequences.shape[0], model_name=model_name)
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


def pattern_crosscheck(current_models, subsequences, lag, step, model_name='linear'):
    residual = {}
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
    # if model_name == 'linear':
    #     voters = {}
    #     delta = []
    #     for name in list(subsequences):
    #         best = ''
    #         for mod in current_models:
    #             # if step-1 == current_models[mod]['steps'][-1]:
    #                 # if len(current_models[mod]['voters']) == step:
    #             model = current_models[mod]['model']
    #             thresh = current_models[mod]['threshold']
    #             if mod not in voters:
    #                 voters[mod] = []
    #             seq = list(current_models[mod]['representative_sequence_value'])
    #             seq.extend(list(subsequences[name].values))
    #             X, y = reshape_sequence(np.array(seq), lag)
    #             pred = model.predict(X)
    #             error = dtw(y, pred)
    #             if error <= thresh:
    #                 if best == '':
    #                     best = (mod, error)
    #                 else:
    #                     if error < best[1]:
    #                         best = (mod, error)
    #         if best == '':
    #             delta.append(name)
    #         else:
    #             voters[best[0]].append(name)
    #
    #     for mod in current_models:
    #         if mod in voters and len(voters[mod])>0:
    #             current_models[mod]['voters'].append(voters[mod])
    #             current_models[mod]['steps'].append(step)
    #
    #     if len(delta) > 0:
    #         residual = subsequences.loc[:, delta]
    return residual


def track_sequence(pattern, seq_name, nbre_steps):
    vec = [''] * nbre_steps
    for model in pattern:
        for i, voters in enumerate(pattern[model]['voters']):
            if seq_name in voters:
                vec[pattern[model]['steps'][i] - 1] = model
    return vec


def view_tracked_sequence(pattern, seq_name, nbre_steps, intervals, co_evolving_data, lag):
    tracked = track_sequence(pattern, seq_name, nbre_steps)
    fr = {}
    fr['true'] = co_evolving_data[seq_name]
    fr['generated'] = []
    fr['model'] = tracked

    for i, model in enumerate(tracked):
        function = pattern[model]['model']
        rep = list(pattern[model]['representative_sequence_value'])
        rep.extend(list(co_evolving_data[intervals[0]:intervals[1]][seq_name].values))
        X, y = reshape_sequence(np.array(rep), lag)
        pred = function.predict(X)
        fr['generated'].extend(pred)
    return fr


def pattern_identification(
        co_evolving_data=None,
        model_name='linear',
        hyperparameters=None,
        first_time_interval=None,
        moving_step=None,
        lag=None
):
    subsequences = get_subsequences(co_evolving_data, first_time_interval)
    time_intervals = [first_time_interval]

    ####################################################################################################################
    # Initialization
    ####################################################################################################################
    step = 1
    patterns = get_patterns(subsequences,
                            lag=lag,
                            step=step,
                            model_name=model_name,
                            hyperparameters=hyperparameters)

    ####################################################################################################################
    # Updating over time
    ####################################################################################################################
    while first_time_interval[-1] < co_evolving_data.index[-1]:
        first_time_interval = get_next_time_interval(list(co_evolving_data.index),
                                                     current_time_interval=first_time_interval,
                                                     moving_step=moving_step)
        time_intervals.append(first_time_interval)
        subsequences = get_subsequences(co_evolving_data, first_time_interval)
        step += 1
        residual = pattern_crosscheck(patterns,
                                      subsequences,
                                      lag=lag,
                                      step=step,
                                      model_name=model_name)
        if len(residual) > 0:
            # print(residual)
            # patterns = get_patterns(residual, lag=lag, step=step, model_name='linear', current_models=patterns)
            patterns = get_patterns(residual,
                                    lag=lag,
                                    step=step,
                                    model_name=model_name,
                                    hyperparameters=hyperparameters,
                                    current_models=patterns)
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


def view_pattern(patterns_, number_steps, title1, title2):
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
    sns.lineplot(data=fr.loc[:, [col for col in list(fr) if col not in models]], ax=axe1, legend=False, lw=2,
                 markers='o', markersize=7, alpha=1)
    axe1.grid(True, alpha=.2)
    axe1.set_title(title1, fontsize=18)
    axe1.set_ylabel('Nbre exhibited sequences', fontsize=16)
    axe1.set_xticklabels(['T' + str(i + 1) for i in range(number_steps)], fontsize=8, rotation=90)

    axe2 = fig.add_subplot(122)
    sns.lineplot(data=fr.loc[:, models], ax=axe2, legend=False, lw=4, alpha=1, markers='o', markersize=7)
    # sns.scatterplot(data=fr.loc[:,models], ax=axe2, legend=False, markers='o')
    axe2.grid(True, alpha=.2)
    axe2.set_title(title2, fontsize=18)
    axe2.set_ylabel('Model index', fontsize=16)
    axe2.set_xticklabels(['T' + str(i + 1) for i in range(number_steps)], fontsize=8, rotation=90)
    plt.show()


def view_causal_relations(patterns_, list_graphs):
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
        t1 = step.split('-')[0]
        t2 = str(int(t1) + 1)
        axe.set_title(r'$G_{' + t1 + ',' + t2 + '}$', fontsize=16)
        # pos = nx.bipartite_layout(G, 'vertical')
        # nx.draw(G, pos=pos)
        # plt.show()
        cpt += 1
    plt.show()


def main(
        path_to_data=None,
        size=42,
        moving_step=30,
        lag=3,
        model_name='linear',
        hyperparameters=None,
        view_pattern_lifespan=False,
        view_causal_graph=False
):
    data = load_data(path_to_data)
    time = data.index
    first_time_interval = time[:size]
    patterns, nbre_steps = pattern_identification(
        co_evolving_data=data,
        model_name=model_name,
        hyperparameters=hyperparameters,
        first_time_interval=first_time_interval,
        moving_step=moving_step,
        lag=lag
    )
    evolving_graphs = causal_graph(
        patterns,
        nbre_steps,
        lag,
        data
    )
    # print(track_sequence(patterns, 'IEF', nbre_steps))
    if view_pattern_lifespan:
        for mod in patterns:
            print(mod)
            for i, voters in enumerate(patterns[mod]['voters']):
                print(patterns[mod]['steps'][i], len(voters), end=',')
            print('\n -----------------')

        equation = None
        if model_name == 'linear':
            equation = r'$a_0 + \sum_{i=1}^{' + str(lag - 1) + '} a_i x_i $'
        if model_name == 'svr':
            equation = r'$\sum_{i=0}^{' + str(size - 1) + '} a_i\phi(x_i,x) $'
        if model_name == 'mlp':
            equation = r'$hidden\; layers\;' + str(hyperparameters['hidden_layer_sizes']) + '$'
        title1 = 'Patterns\' persistence.\n Model: ' + equation
        title2 = 'Patterns\' lifespan. \n Model: ' + equation
        view_pattern(patterns, nbre_steps, title1, title2)

    if view_causal_graph:
        view_causal_relations(patterns, evolving_graphs)

    return patterns, evolving_graphs, nbre_steps, data


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

size = 200
moving_step = 50
lag = 3
model_name = 'linear' # svr and mlp
linear_hyperparameters={
    'fit_intercept':True,
    'n_jobs':None,
    'positive':False
}
# svr_poly_hyperparameters={'nu':0.2,
#                  'C':1.0,
#                  'kernel':'poly',
#                  'degree':2,
#                  'gamma':'scale',
#                  'coef0':0.0,
#                  'shrinking':True,
#                  'tol':0.001,
#                  'cache_size':200,
#                  'verbose':False,
#                  'max_iter':-1
#                  }
# mlp_hyperparameters = {
#     'hidden_layer_sizes': (50, 30),
#     'activation': 'relu',
#     'solver': 'adam',
#     'alpha': 0.0001,
#     'batch_size': 'auto',
#     'learning_rate': 'constant',
#     'learning_rate_init': 0.001,
#     'power_t': 0.5,
#     'max_iter': 200,
#     'shuffle': True,
#     'random_state': 1024,
#     'tol': 0.0001,
#     'verbose': False,
#     'warm_start': False,
#     'momentum': 0.9,
#     'nesterovs_momentum': True,
#     'early_stopping': True,
#     'validation_fraction': 0.1,
#     'beta_1': 0.9,
#     'beta_2': 0.999,
#     'epsilon': 1e-08,
#     'n_iter_no_change': 10,
#     'max_fun': 15000
# }
view_pattern_lifespan = True
view_causal_graph = True

data_name = 'weather'

if __name__ == '__main__':
    patterns, evolving_graphs, nbre_steps, data = main(
        path_to_data=path_to_data[data_name],
        size=size,
        moving_step=moving_step,  # stride
        lag=lag,
        model_name=model_name,
        hyperparameters=linear_hyperparameters,
        view_pattern_lifespan=view_pattern_lifespan,
        view_causal_graph=view_causal_graph
    )
    dico = {
        'settings': (size, moving_step, lag),
        'evolving graphs': evolving_graphs,
        'nbre_steps': nbre_steps,
        'data': data,
        'patterns': patterns
    }

    name = path_to_data.split('/')[-1].split('.')[0]

    f = open("/home/local/USHERBROOKE/owup2301/PycharmProjects/dependence/TPAMI/storage/"+name+".pkl", "wb")

    pickle.dump(dico, f)
    # close file
    f.close()
