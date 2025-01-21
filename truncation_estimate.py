
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx


def trunc_by_thresh(core_states, drive_term, labels = None, thresh=1e-2, total_trunc=None):
    """
    Returns a list of state indices that are connected to the specified core states
    via large entries in the given drive term

    Args:
        core_states (list[int]): The indices of the states to begin with 
                                   (should be the logical states).
        drive_term (np.array complex): The operator used to drive a gate.
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given
        thresh (float, optional): The threshold above which states count as connected.
                                  Defaults to 1e-2.
        total_trunc (int, optional): Highest index to consider. If none is given
                                     then considers all entries in drive_term

    Returns:
        list[int]: list of state indices
    """
    
    if total_trunc is None:
        total_trunc = drive_term.shape[1]

    if labels is None:
        labels = np.arange(drive_term.shape[1])
    
    hspace_index = [s for s in core_states]
    # Add every state that is connected by entries above thresh
    # to the core states in the drive term
    # By adding to hspace_index as you're looping through it,
    # we consider as many degrees of connection as we need 
    for s in hspace_index:
        for i, s2 in enumerate(labels):
            if np.abs(drive_term[s, i]) > thresh and i not in hspace_index:
                hspace_index.append(s2)

    return hspace_index


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


def make_rate_graph(drive_term, evals, wd, A, labels = None):
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
    for i, s_i in enumerate(labels):
        for j, s_j in enumerate(labels):
            if i < j:
                n_ij = drive_term[i, j]
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


def _shortest_path_to_core_parallel(p):
    return shortest_path_to_core(*p)


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
    return target, (weight_tot, path_tot)


def make_leakage_df(core_states, drive_term, evals, wd, A, G=None, labels = None, n_cpu=4):
    """
    Makes a dataframe where each row is a state rated by how much
    leakage is expected

    Args:
        core_states (list[int]): The indices of the states to begin with 
                                   (should be the logical states).
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float): drive frequency
        A (float): drive amplitude
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given

    Returns:
        dataframe
    """
    if labels is None:
        labels = np.arange(drive_term.shape[0])

    df = []
    if G is None:
        G = make_rate_graph(drive_term, evals, wd, A, labels = labels)

    pool = Pool(processes=n_cpu)
    params = [(G, core_states, s) for s in labels]
    for target, (shortest_path_len, shortest_path) in tqdm(pool.imap_unordered(_shortest_path_to_core_parallel, params),
                total=len(labels)):
        entry = {}
        entry["i"] = target
        entry["path"] = shortest_path
        entry["path_len"] = np.exp(-shortest_path_len)
        df.append(entry)
    
    return pd.DataFrame(df).sort_values(by="path_len", ascending=False)


def trunc_by_graph_estimate(n, core_states, drive_term, evals, wd, A, G=None, labels=None, n_cpu=4):
    """
    Returns indices/state labels for a truncated model, keeping the n most important states
    according to the graph search estimate

    Args:
        n (int): number of states to include in the reduced model
        core_states (list[int]): The indices of the states to begin with 
                                   (should be the logical states).
        drive_term (np.array[complex]): The operator used to drive a gate.
        evals (list[float]): eigenvalues of the system
        wd (float): drive frequency
        A (float): drive amplitude
        labels (list[str], optional): labels to use for each state. Uses
                                      the index of the state if none is given

    Returns:
        list of state indices/labels
    """

    df = make_leakage_df(core_states, drive_term, evals, wd, A, G=G, labels=labels, n_cpu=n_cpu)
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
    detune = 0. # Assume very close to resonance driving -- not actually true
    wd = W_20_50 + detune
    hspace_logi = ['0-0', '0-2', '2-0', '2-2']
    hspace_index = [hspace_full.index(i) for i in hspace_logi]
    core_states = hspace_logi + ['5-0']

    drive_term = n_theta1_dress

    G = make_rate_graph(drive_term, eval_tot, wd, A, labels = hspace_full)
    df = make_leakage_df(core_states, drive_term, eval_tot, wd, A, labels = hspace_full, n_cpu=100)
    states = trunc_by_graph_estimate(100, core_states, drive_term, eval_tot, wd, A, labels=hspace_full)