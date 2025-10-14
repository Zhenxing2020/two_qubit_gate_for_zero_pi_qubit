import sys
sys.path.append('../')
from multiprocessing import Pool
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import utils_2Q_gate_zp as ut
import scqubits as scq


from joblib import Parallel, delayed

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

    for s in labels:
        target, (shortest_path_len, shortest_path) = path_func(G, core_states, s)
        entry = {}
        entry["i"] = target
        entry["path"] = shortest_path
        entry["path_len"] = np.exp(-shortest_path_len)
        df.append(entry)

    return pd.DataFrame(df).sort_values(by="path_len", ascending=False)


def trunc_by_graph_estimate(core_states, drive_term, evals, wd, A, labels=None,
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
    return list(df["i"].values)


def rank_by_fid_contrib(initial_order, pulse_argz, savefile, n_jobs=50, start=3, end=None):
    
    def eval_fid(n):
        argz = pulse_argz[:4] + [sorted(initial_order[:n])] + pulse_argz[5:]
        return ut.xgate_fidelity_log(argz)
    
    if end is None:
        end = len(initial_order)

    n_states = np.arange(start, end)
    log_infid = Parallel(n_jobs=min(n_jobs, len(n_states)), verbose=10)(delayed(eval_fid)(x) for x in n_states)
    fid = 1-10**np.array(log_infid)

    diff = list(range(1, start+2))[::-1] + list(np.abs(np.diff(fid)))
    re_order = np.array(initial_order)[np.argsort(diff)[::-1]]

    np.savez(savefile, log_infid=log_infid, fid=fid, initial_order=initial_order, diff=diff, re_order=re_order)

    return fid, diff, re_order


def compare_orders(orders, labels, n_states, pulse_argz, savefile, n_jobs=50, true_val=None, fids = {}):

    # Get "True" Value
    if true_val is None:
        argz = pulse_argz[:4] + [sorted(orders[0])] + pulse_argz[5:]
        true_val = 1-10**ut.xgate_fidelity_log(argz)
        print(f"-----------------------({len(orders[0])} states)")
        print("True Val", true_val)
        print("-----------------------")
    
    # Define fidelity comparison functions
    funcs = []
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(ncols=2, figsize=(12,4))
    for order, label in zip(orders, labels):
        eval_fid = lambda n: 1-10**ut.xgate_fidelity_log(pulse_argz[:4] + [sorted(order[:n])] + pulse_argz[5:])
        if label not in fids:
            fids[label] = Parallel(n_jobs=min(n_jobs, len(n_states)), verbose=10)(delayed(eval_fid)(x) for x in n_states)
        ax[0].plot(n_states, fids[label], label=label)
        ax[1].plot(n_states, true_val - np.array(fids[label]), label=label)


    plt.suptitle("Fidelity of Test Pulse vs. Num States in Model (Using Experimental Mixed Coupling, 800 ns pulse)")

    ax[0].plot(n_states, [true_val]*len(n_states), "--", label="600 States")
    ax[0].set_xlabel("States in Model")
    ax[0].set_ylim(true_val*0.99, 1)
    ax[0].legend()
    ax[0].set_ylabel("Fidelity");

    # ax[1].plot(n_states, [true_val]*len(n_states), "--", label="600 States")
    ax[1].set_xlabel("States in Model")
    # ax[1].set_ylim(true_val*0.999, true_val*1.001)
    ax[1].legend()
    ax[1].set_title("Error")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Fidelity");
    plt.tight_layout()
    plt.savefig(savefile)

    return fids




if __name__ == "__main__":

    # Phi Rank by delta of fidelity
    savename = "H_exp.npz"
    trunc1 = 1000
    import os
    import qutip as qt
    if os.path.exists(savename):
        params = np.load(savename)
        H0 = qt.Qobj(params["H0"])
        drive_term = qt.Qobj(params["drive"])
        w_trans_1 = params["w_trans_1"]
        w_trans_2 = params["w_trans_2"]
        hspace_charge = list(params["hspace_reduced"])
        trunc_model = list(params["trunc_model"])
        hspace_full=np.arange(trunc1)

    tg, amp_A, amp_B, detune_A, detune_B = [828.759495, 0.013563, 0.034964, -0.003029, -0.003182] 
    pulse_argz = [H0, drive_term, w_trans_1, w_trans_2, [0, 2, 9], 1, tg,
                amp_A, amp_B, detune_A, detune_B]
    fid, diff, reorder = rank_by_fid_contrib(trunc_model, pulse_argz, "reorder.npz", start=25, end=500)
    reorder = list(np.load("reorder.npz")["re_order"])

    # _, _, reorder2 = rank_by_fid_contrib(trunc_model, pulse_argz, "reorder2.npz", start=10, end=200, n_jobs=200)
    # _, _, reorder3 = rank_by_fid_contrib(trunc_model, pulse_argz, "reorder3.npz", start=25, end=200, n_jobs=200)

    # Find big reorder difference
    df = []
    import pandas as pd
    for i in range(len(reorder)):
        entry = {"state":reorder[i],
                 "graph":trunc_model.index(reorder[i]),
                 "reorder":i}
        entry["diff"] = entry["graph"] - entry["reorder"]
        df.append(entry)
    df = pd.DataFrame(df)
    df = df.sort_values(by="diff", ascending=False)
    df = df.sort_values(by="reorder", ascending=True)


    # truc_full = 300
    true_val = 1-10**(-2.93808283)
    for n in [0, 2, 9]:
        hspace_charge.remove(n)
    hspace_charge = [0, 2, 9] + hspace_charge
    fids = compare_orders([hspace_charge, trunc_model, reorder], ["state", "graph", "graph + reorder"],
                          np.arange(10, 180), pulse_argz,
                   savefile="order_comparison.png", true_val=true_val, n_jobs=200)