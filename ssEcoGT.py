"""
Stage-structured ecological graph theory (ssEcoGT) model implementation.

Author: Ata Kalirad
Date: 14-01-2026
"""
import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx
import random as rnd
import os
import uuid

def get_prop_neighbours(graph, node):
    """Find proportion of conspecific and allospecific neighbours

    Args:
        graph (NetworkX Graph): The graph representing the population.  
        node (Node): The node for which to calculate the proportion of conspecific and allospecific neighbours.

    Returns:
        tuple: A tuple containing the number of conspecific and allospecific neighbours.
    """
    neighbors =list(graph.neighbors(node))
    conspecific = len([v for v in neighbors if (graph.nodes[v]['Phenotype'] == graph.nodes[node]['Phenotype'] and (graph.nodes[v]['Stage'] == 'Juvenile' or graph.nodes[v]['Stage'] == 'Adult'))])
    allospecific = len([v for v in neighbors if (graph.nodes[v]['Phenotype'] != graph.nodes[node]['Phenotype'] and (graph.nodes[v]['Stage'] == 'Juvenile' or graph.nodes[v]['Stage'] == 'Adult'))])
    return conspecific, allospecific

def connectivity_based_death(graph, node, pD, alpha={'A': 0.5, 'B': 0.5}):
    """Calculate death probability based on connectivity.

    Args:
        graph (NetworkX Graph): The graph representing the population.
        node (Node): The node for which to calculate the proportion of conspecific and allospecific neighbours.
        pD (float): Dath propensity base value.
        alpha (dict, optional): Species competition parameters. Defaults to {'A': 0.5, 'B': 0.5}.

    Returns:
        float: The death probability for the node.
    """
    conspecific, allospecific = get_prop_neighbours(graph, node)
    comp_par = alpha[graph.nodes[node]['Phenotype']]
    exp = comp_par - 1.0
    if allospecific == 0:
        return pD
    else:
        return pD * np.power(allospecific, exp)
    
def random_rewire(u, G):
    """Randomly rewire an edge connected to node u in graph G.

    Args:
        u (Node): The node to rewire.
        G (NetworkX Graph): The graph to rewire.

    Returns:
        NetworkX Graph: The rewired graph.
    """
    graph = deepcopy(G)
    assert u in graph.nodes
    assert graph.degree(u) > 0
    neighbors = set(graph.neighbors(u))
    candidates = set(graph.nodes) - neighbors - {u}
    v = rnd.choice(list(neighbors))
    graph.remove_edge(u, v)
    w = rnd.choice(list(candidates))
    graph.add_edge(u, w)
    return graph

def SC_on_graph(J_to_A, A_to_D, eta, rA, rB, mf, alpha, N, n, max_time, path, rewiring_rate, ER):
    """Simulate stage-structured ecological dynamics on a graph.

    Args:
        J_to_A (float): Juvenile to Adult maturation rate.
        A_to_D (float): Adult to Dead death rate.
        eta (float): Predation rate.
        rA (float): Reproduction rate for species A.
        rB (float): Reproduction rate for species B.
        mf (dict): Probability of an adult being predator for each species.
        alpha (dictionary): Interaction parameters for each species.
        N (int): Number of nodes in the graph.
        n (int): Number of initial individuals.
        max_time (int): Maximum simulation time.
        path (str): Path to save simulation results.
        rewiring_rate (float): Rate of graph rewiring.
        ER (float): Erdos-Renyi rewiring probability.
    """
    J_to_A = J_to_A
    A_to_D = A_to_D
    pred_rate = eta
    rA = rA
    rB = rB
    mf_prob = mf
    int_dict = alpha
    rewiring_rate = rewiring_rate
    N = N
    if ER > 0.0:
        graph_type = 'ER'
        pop = nx.erdos_renyi_graph(N, ER)
    elif rewiring_rate > 0:
        graph_type = 'ER'
        pop = nx.erdos_renyi_graph(N, 0.1)
    else:
        graph_type = 'Complete'
        pop = nx.complete_graph(N)
    nodes = list(pop.nodes)
    possible_attributes = [{'Phenotype': 'A', 'Stage': "Juvenile", 'MF': None, 'R':rA} , 
                    {'Phenotype': 'B', 'Stage': "Juvenile", 'MF': None, 'R':rB}]
    attributes = {nodes[i]: {'Phenotype': 'NA', 'Stage': "Empty", 'MF': None, 'R':0.0} for i in range(N)}
    if n < N:
        chosen_nodes = rnd.sample(list(pop.nodes()), n)
        group1 = set(rnd.sample(chosen_nodes, n // 2))
        group2 = set(chosen_nodes) - group1
        for node in group1:
            attributes[node] = possible_attributes[0]
        for node in group2:
            attributes[node] = possible_attributes[1]
    else:
       
        group1 = set(rnd.sample(list(attributes), n // 2))
        attributes = {k: possible_attributes[0] if k in group1 else possible_attributes[1] for k in attributes}

    nx.set_node_attributes(pop, attributes)

    columns = [
    'Time', 'JA', 'AA', 'DA', 'KA', 'JB', 'AB', 'DB', 'KB',
    'Rewiring', 'Predation_AB', 'Predation_BA',
    'MaturationA', 'MaturationB',
    'ReproductionA', 'ReproductionB',
    'DeathA', 'DeathB'
]
    
    comp_path = path + '/' + graph_type + f'_{ER}_JtoA_{J_to_A}_AtoD_{A_to_D}_pred_{pred_rate}_rA_{rA}_rB_{rB}_mfA_{mf["A"]}_mfB_{mf["B"]}_alphaA_{alpha["A"]}_alphaB_{alpha["B"]}_N_{N}_n_{n}_t_{max_time}_r_{rewiring_rate}/'
    if not os.path.exists(comp_path):
        os.makedirs(comp_path)
    id = str(uuid.uuid4())
    buffer, flush_every = [], 1000
    output_path = comp_path + 'sim_results_'+ id + '.csv'
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)



    clock = 0
    types_of_reactions = {'Rewiring': 0, 'Predation_AB': 0, 'Predation_BA': 0, 'MaturationA': 0, 'MaturationB': 0, 'ReproductionA': 0, 'ReproductionB': 0, 'DeathA': 0, 'DeathB': 0}
    row = {'Time': clock,'JA': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Juvenile')]), 
        'AA': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Adult')]), 
        'DA':len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Dead')]), 
        'KA': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Killed')]), 
        'JB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Juvenile')]), 
        'AB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Adult')]), 
        'DB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Dead')]), 
        'KB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Killed')])}
    
    combined = {**row, **types_of_reactions}
    buffer.append(combined)
    while clock < max_time:
        edge_was_rewired = False
        types_of_reactions = {'Rewiring': 0, 'Predation_AB': 0, 'Predation_BA': 0, 'MaturationA': 0, 'MaturationB': 0, 'ReproductionA': 0, 'ReproductionB': 0, 'DeathA': 0, 'DeathB': 0}
        reaction_types = []
        comb_reactions = []
        # reproduction reactions
        edges = [
        (u, v) for u, v in pop.edges()
        if (pop.nodes[u]["Stage"] == 'Adult' and (pop.nodes[v]["Stage"] == 'Killed' or pop.nodes[v]["Stage"] == 'Dead' or pop.nodes[v]["Stage"] == 'Empty'))
        or (pop.nodes[v]["Stage"] == 'Adult' and (pop.nodes[u]["Stage"] == 'Killed' or pop.nodes[u]["Stage"] == 'Dead' or pop.nodes[u]["Stage"] == 'Empty'))
        ]
        potential_mother = [u if pop.nodes[u]["Stage"] == "Adult" else v for u, v in edges if (pop.nodes[u]["Stage"] == "Adult") ^ (pop.nodes[v]["Stage"] == "Adult")]
        props_rep = [pop.nodes[v].get("R") for v in potential_mother]
        reaction_types += ['Rep'] * len(potential_mother)
        comb_reactions += edges
        # dev reactions
        J_to_A_nodes = [u for u in pop.nodes if pop.nodes[u]['Stage'] == 'Juvenile']
        props_JA = [J_to_A] * len(J_to_A_nodes)
        comb_reactions += J_to_A_nodes
        A_to_D_nodes = [u for u in pop.nodes if pop.nodes[u]['Stage'] == 'Adult']
        #props_AD = [A_to_D] * len(A_to_D_nodes)
        props_AD = [connectivity_based_death(pop, u, A_to_D, alpha=int_dict) for u in A_to_D_nodes]
        comb_reactions += A_to_D_nodes
        reaction_types += ['Maturation'] * len(props_JA)
        reaction_types += ['Death'] * len(props_AD)
        # predation reactions
        edges_pred = [
            (u, v) for u, v in pop.edges()
            if (
                (
                    pop.nodes[v]["MF"] == 1 and
                    pop.nodes[u]["Stage"] == "Juvenile" and pop.nodes[v]["Stage"] == "Adult"
                )
                or
                (
                    pop.nodes[u]["MF"] == 1  and
                    pop.nodes[u]["Stage"] == "Adult" and pop.nodes[v]["Stage"] == "Juvenile"
                )
            )
            and pop.nodes[u]["Phenotype"] != pop.nodes[v]["Phenotype"]
        ]
        comb_reactions += edges_pred
        props_pred = [pred_rate] * len(edges_pred)
        reaction_types += ['Pred'] * len(edges_pred)
        if rewiring_rate > 0:
            # rewiring reaction
            rewiring_nodes = [u for u in pop.nodes if (pop.nodes[u]['Stage'] != 'Dead' and pop.nodes[u]['Stage'] != 'Killed' and pop.nodes[u]['Stage'] != 'Empty') and pop.degree(u) > 0]
            props_rewiring = [rewiring_rate] * len(rewiring_nodes)
            comb_reactions += rewiring_nodes
            reaction_types += ['Rewiring'] * len(props_rewiring)
        # comb propensities
        if rewiring_rate > 0:
            props = props_rep + props_JA + props_AD + props_pred + props_rewiring
        else:
            props = props_rep + props_JA + props_AD + props_pred
        if len(props) > 0:
            times = [(1/(i + 1e-10))*np.log(1/np.random.uniform()) for i in props]
            next_reaction = np.argmin(times)
            if reaction_types[next_reaction] == 'Rep':
                if pop.nodes[comb_reactions[next_reaction][0]]['Stage'] == 'Adult':
                    pop.nodes[comb_reactions[next_reaction][1]]['Stage'] = 'Juvenile'
                    pop.nodes[comb_reactions[next_reaction][1]]['Phenotype'] = pop.nodes[comb_reactions[next_reaction][0]]['Phenotype']
                    pop.nodes[comb_reactions[next_reaction][1]]['MF'] = 0
                    pop.nodes[comb_reactions[next_reaction][1]]['R'] = pop.nodes[comb_reactions[next_reaction][0]]['R']
                    if pop.nodes[comb_reactions[next_reaction][0]]['Phenotype'] == 'A':
                        types_of_reactions['ReproductionA'] += 1 
                    else:
                        types_of_reactions['ReproductionB'] += 1

                else:
                    pop.nodes[comb_reactions[next_reaction][0]]['Stage'] = 'Juvenile'
                    pop.nodes[comb_reactions[next_reaction][0]]['Phenotype'] = pop.nodes[comb_reactions[next_reaction][1]]['Phenotype']
                    pop.nodes[comb_reactions[next_reaction][0]]['MF'] = 0
                    pop.nodes[comb_reactions[next_reaction][0]]['R'] = pop.nodes[comb_reactions[next_reaction][1]]['R']
                    if pop.nodes[comb_reactions[next_reaction][1]]['Phenotype'] == 'A':
                        types_of_reactions['ReproductionA'] += 1
                    else:
                        types_of_reactions['ReproductionB'] += 1
            elif reaction_types[next_reaction] == 'Maturation':
                pop.nodes[comb_reactions[next_reaction]]['Stage'] = 'Adult'
                mf_stage = np.random.binomial(1, mf_prob[pop.nodes[comb_reactions[next_reaction]]['Phenotype']])
                pop.nodes[comb_reactions[next_reaction]]['MF'] = mf_stage
                if pop.nodes[comb_reactions[next_reaction]]['Phenotype'] == 'A':
                    types_of_reactions['MaturationA'] += 1
                else:
                    types_of_reactions['MaturationB'] += 1
            elif reaction_types[next_reaction] == 'Pred':
                if pop.nodes[comb_reactions[next_reaction][0]]['Stage'] == "Juvenile":
                    pop.nodes[comb_reactions[next_reaction][0]]['Stage'] = 'Killed'
                    if pop.nodes[comb_reactions[next_reaction][1]]['Phenotype'] == 'A':
                        types_of_reactions['Predation_AB'] += 1
                else:
                    pop.nodes[comb_reactions[next_reaction][1]]['Stage'] = 'Killed'
                    if pop.nodes[comb_reactions[next_reaction][0]]['Phenotype'] == 'A':
                        types_of_reactions['Predation_AB'] += 1
                    else:
                        types_of_reactions['Predation_BA'] += 1
            elif reaction_types[next_reaction] == 'Rewiring':
                u = comb_reactions[next_reaction]
                pop = random_rewire(u, pop)
                types_of_reactions['Rewiring'] += 1
                edge_was_rewired = True
            else:
                pop.nodes[comb_reactions[next_reaction]]['Stage'] = 'Dead'
                if pop.nodes[comb_reactions[next_reaction]]['Phenotype'] == 'A':
                    types_of_reactions['DeathA'] += 1
                else:
                    types_of_reactions['DeathB'] += 1
            clock += times[next_reaction]
            if not edge_was_rewired:
                row = {'Time': clock, 'JA': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Juvenile')]), 
            'AA': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Adult')]), 
            'DA':len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Dead')]), 
            'KA': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'A' and pop.nodes[v]['Stage'] == 'Killed')]), 
            'JB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Juvenile')]), 
            'AB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Adult')]), 
            'DB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Dead')]), 
            'KB': len([v for v in pop.nodes if (pop.nodes[v]['Phenotype'] == 'B' and pop.nodes[v]['Stage'] == 'Killed')])}
                combined = {**row, **types_of_reactions}
                buffer.append(combined)
                if len(buffer) >= flush_every:
                    pd.DataFrame.from_records(buffer).to_csv(output_path, mode='a', header=False, index=False)
                    buffer.clear()

