
from multiprocessing import Pool

from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx


def trunc_by_thresh(core_states_index, drive_term, thresh=1e-2, total_trunc=None):
    """
    Returns a list of state indices that are connected to the specified core states
    via large entries in the given drive term

    Args:
        core_states_index (list[int]): The indices of the states to begin with 
                                   (should be the logical states).
        drive_term (np.array complex): The operator used to drive a gate.
        thresh (float, optional): The threshold above which states count as connected.
                                  Defaults to 1e-2.
        total_trunc (int, optional): Highest index to consider. If none is given
                                     then considers all entries in drive_term

    Returns:
        list[int]: list of state indices
    """
    
    if total_trunc is None:
        total_trunc = drive_term.shape[1]
    
    hspace_index = [s for s in core_states_index]
    # Add every state that is connected by entries above thresh
    # to the core states in the drive term
    # By adding to hspace_index as you're looping through it,
    # we consider as many degrees of connection as we need 
    for s in hspace_index:
        for i, s2 in enumerate(np.arange(total_trunc)):
            if np.abs(drive_term[s, i]) > thresh and i not in hspace_index:
                hspace_index.append(i)

    return sorted(hspace_index)


def pop_rate(A, n_ij, delta):
    """Calculates the maximum population that could transfer
    between two states if it were considered as a two state system

    Args:
        A (float): drive amplitude
        n_ij (complex): entry i,j of the operator used to drive
        delta (float): detuning from drive transition
    Returns:
        float: max population that could transfer (between 0-1)
    """
    return (np.abs(A*n_ij)**2) / (np.abs(A*n_ij)**2 + delta**2)


def make_rate_graph(drive_term, evals, wd, A, labels = None, normalization=True):
    """
    Makes a graph that represents the population transfer rate
    of a system under the presence of the specified monotone drive

    Args:
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float): drive frequency
        A (float): drive amplitude
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given

    Returns:
        nx.Graph: Graph representing the system
    """

    if labels is None:
        labels = np.arange(drive_term.shape[0])

    G = nx.DiGraph()
    max_n_ij = np.max(np.abs(drive_term.data))
    for i, s_i in enumerate(labels):
        for j, s_j in enumerate(labels):
            if i < j:
                n_ij = np.abs(drive_term[i, j])
                if normalization:
                    n_ij *= n_ij/max_n_ij
                delta = abs(wd - (evals[j] - evals[i]))
                population_rate = pop_rate(A, n_ij, delta)
                if population_rate > 0:
                    G.add_edge(s_i, s_j, weight=-np.log(population_rate))
    
    return G

def shortest_path_to_core(G, core_states, target):
    """Finds the sortest path to the specified core states

    Args:
        G (nx.Graph): graph where nodes are states and edges
                      represent a population transfer rate under
                      the specified drive
        core_states (list[int, str]): list of core states, as they are
                                      labeled in the graph
        target (int or str): target state

    Returns:
       target, (shortest path length, shortest path)
    """
    shortest_path = ""
    shortest_path_len = np.inf
    for source in core_states[:-1]:
        if nx.has_path(G, source, target):
            path = nx.shortest_path(G, source=source, target=target,
                                    weight="weight")
            path_len = nx.path_weight(G, path,'weight')
        if path_len < shortest_path_len:
            shortest_path_len = path_len
            shortest_path = ",".join([str(x) for x in path])
    return target, (shortest_path_len, shortest_path)


def all_path_to_core(G, core_states, target, cutoff=2):
    """Finds all paths to the specified core states
    under a specified length

    Args:
        G (nx.Graph): graph where nodes are states and edges
                      represent a population transfer rate under
                      the specified drive
        core_states (list[int, str]): list of core states, as they are
                                      labeled in the graph
        target (int or str): target state
        cutoff (float): maximum length to consider for paths

    Returns:
       target, (total length of paths, all paths)
    """
    path_tot = []
    if target in core_states:
        weight_tot = 1
    else:
        weight_tot = 0
        for source in core_states:
            for path in nx.all_simple_paths(G, source, target, cutoff=cutoff):
                weight_tot += np.exp( - nx.path_weight(G, path,'weight') )
                path_tot.append(path)
    return target, (-np.log(weight_tot), path_tot)


def make_leakage_df(core_states, drive_term, evals, wd, A, labels=None, 
                    path_func=shortest_path_to_core, G=None):
    """
    Makes a dataframe where each row is a state rated by how much
    leakage is expected

    Args:
        core_states (list[int]): The indices of the states to begin with 
                                   (should be the logical states).
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float or list of float): drive frequency(s)
        A (float or list of float): drive amplitude(s)
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given
        path_func (func): function that takes in (G, core_states, s) and returns
                          a distance from s to core_states. Either shortest_path_to_core
                          or all_path_to_core
        G (nx.Graph): pre-computed rate-graph. Makes one if none is given.

    Returns:
        dataframe
    """
    if labels is None:
        labels = np.arange(drive_term.shape[0])

    df = []
    if G is None:
        G = make_rate_graph(drive_term, evals, wd, A, labels = labels)

    for s in tqdm(labels, total=len(labels)):
        target, (shortest_path_len, shortest_path) = path_func(G, core_states, s)
        entry = {}
        entry["i"] = target
        entry["path"] = shortest_path
        entry["path_len"] = np.exp(-shortest_path_len)
        df.append(entry)
    
    return pd.DataFrame(df).sort_values(by="path_len", ascending=False)


def trunc_by_graph_estimate(n, core_states, drive_term, evals, wd, A, labels=None,
                            path_func=shortest_path_to_core):
    """
    Returns indices/state labels for a truncated model, keeping the n most important states
    according to the graph search estimate.

    You can give multiple drive pulses by making wd and A lists. In this case it
    will combine the dataframes, keeping the maximum entry for each state.

    Args:
        n (int): number of states to include in the reduced model
        core_states (list[int]): The indices of the states to begin with 
                                   (should be the logical states).
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float or list of float): drive frequency(s)
        A (float or list of float): drive amplitude(s)
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given
        path_func (func): function that takes in (G, core_states, s) and returns
                          a distance from s to core_states. Either shortest_path_to_core
                          or all_path_to_core
        G (nx.graph or list of nx.graph): precomputed rate graphs, must match len of A,wd

    Returns:
        list of state indices/labels
    """

    if isinstance(wd, float):
        wd = [wd]
    if isinstance(A, float):
        A = [A]
    
    df_list = []
    for i in range(len(wd)):
        df = make_leakage_df(core_states, drive_term, evals, wd[i], A[i], G=None, labels=labels,
                             path_func=path_func)
        df_list.append(df)
    df = pd.concat(df_list).sort_values("path_len", ascending=False).drop_duplicates("i", keep="first")
    return list(df["i"].values[:n])

if __name__ == "__main__":

    truc1, truc_tot, charge_pick  = 400, 500, True

    folder = f'two_qubit_data_truc1={truc1}_truc2={truc_tot}_pick={charge_pick}/'
    eval_tot = pd.read_csv(folder+ 'eval_tot.txt').to_numpy().flatten()
    n_theta0_dress = pd.read_csv(folder+ 'n_theta0_dress.txt').to_numpy()
    n_theta1_dress = pd.read_csv(folder+ 'n_theta1_dress.txt').to_numpy()
    hspace_full = pd.read_csv(folder+ 'hspace_full.txt').to_numpy().flatten().tolist()


    import utils_2Q_gate_zp as ut

    truc_tot_2 = 500
    truc_list = np.arange(truc_tot_2)
    hspace_full = hspace_full[:truc_tot_2]
    eval_tot = eval_tot[:truc_tot_2]

    n_theta0_dress = ut.truncate_2(n_theta0_dress, truc_list)
    n_theta1_dress = ut.truncate_2(n_theta1_dress, truc_list)


    A = 0.02
    W_20_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-0')]
    W_22_50 = eval_tot[hspace_full.index('5-0')] - eval_tot[hspace_full.index('2-2')]
    detune = 0. # Assume very close to resonance driving -- not actually true
    wd = W_20_50 + detune
    hspace_logi = ['0-0', '0-2', '2-0', '2-2']
    hspace_index = [hspace_full.index(i) for i in hspace_logi]
    core_states = hspace_logi + ['5-0']

    drive_term = n_theta1_dress

    # hspace_index_2 = trunc_by_thresh(hspace_index, drive_term, thresh=1e-2)

    # G = make_rate_graph(drive_term, eval_tot, wd, A, labels = hspace_full)
    # df = make_leakage_df(core_states, drive_term, eval_tot, wd, A, labels = hspace_full, n_cpu=100)

    A = [0.02, 0.01]
    wd = [W_20_50, W_22_50]
    states_short = trunc_by_graph_estimate(100, core_states, drive_term, eval_tot, wd, A, labels=hspace_full,
                                         path_func=shortest_path_to_core)
    states_all = trunc_by_graph_estimate(100, core_states, drive_term, eval_tot, wd, A, labels=hspace_full,
                                         path_func=all_path_to_core)