from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter
import random
import pickle
import re
import signal
import functools
import json
import numpy as np
import walker
import math
import time
import pandas as pd
# from tasks.base import *
from .base import *


class Diameter_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(Diameter_Task, self).__init__(data_loc, 'Diameter')
        self.examples = []
    
    def check_solution(self, problem_id, response, graph=None, problem_text=None):
        """Check the solution for Diameter problem."""
        #print("\n---Diameter CHECK SOLUTION DEBUG---")
        # #print("Problem ID:", problem_id)
        #print("Response:", response)

        # g = self.problem_set[problem_id]['graph']
        if graph:
            # #print("Using provided graph:")
            # #print(f"- Nodes: {graph.number_of_nodes()}")
            # #print(f"- Edges: {graph.number_of_edges()}")
            # #print("- Sample node names:", [graph.nodes[n]["name"] for n in list(graph.nodes())[:3]])
            g = graph
        else:
            if problem_id not in self.problem_set:
                # #print("Problem ID not found in problem_set!")
                return -1
            g = self.problem_set[problem_id]['graph']
            # #print("Using problem_set graph")       

        pattern = re.compile(r'\[(.*?)\]')
        matches = pattern.findall(response)
        #print("Found patterns:", matches)

        if matches: # 匹配上了
            # for match in reversed(matches):
            match = matches[-1]
            match = match.split(",")
            name_list = [name.strip() for name in match]
            #print("Name list:", name_list)
            node_list = []
            for name in name_list:
                node = find_node_by_name(g, name)
                # #print(f"Looking up node for name {name}: {node}")
                if node is None:
                    continue
                node_list.append(node)
            # if len(node_list) == 0:
            #     continue
            # #print("Final node list:", node_list)
            if not node_list:
                #print("No valid nodes found!")
                return -2 # 匹配上了但是没有节点
            if self.is_feasible(g, node_list):
                #print(f"Valid solution - found Diameter of {len(node_list) - 1} edges")
                return len(node_list) -1 # 边数=点数-1
            else:
                #print("Invalid solution")
                return -2 # 匹配上了但是不合法
        #print("No solution pattern found")
        return -1
    
    def generate_dataset(self, count=500):          
        G = pickle.load(open('source/DBPedia.pkl', 'rb'))
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.5)
        
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 14) if difficulty == 'easy' else (15, 30)
            while len(self.problem_set) < count:
                node_size = sample_node_size(min_nodes, max_nodes)
                selected_walk = random.choice(all_walks)
                visited = set()
                for node in selected_walk:
                    visited.add(node)
                    if len(visited) >= node_size:
                        break
                H = nx.induced_subgraph(G, visited)                
                if nx.number_connected_components(H) > 1:
                    continue
                
                exact_answer, path = self.exact_solver(H)
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(H, path))
                    continue
                
                self.problem_set.append({
                    'id' : len(self.problem_set),
                    'problem_text' : self.generate_problem(H),
                    'graph': H,
                    'path': path,
                    'exact_answer': exact_answer
                })
            self.save_dataset(difficulty)
    
    def is_feasible(self, g, route_list):
        for i in range(len(route_list)):
            if i == len(route_list) - 1:
                break
            node1 = route_list[i]
            node2 = route_list[i+1]            
            if g.has_edge(node1, node2) == False:
                ## #print('non_existance connection')
                return False
        if len(nx.shortest_path(g, route_list[0], route_list[-1])) != len(route_list):
            # # #print('not the shortest path')
            return False
        return True
    
    def generate_problem(self, graph):
        description = [
            "You are required to calculate the diameter of an undirected knowledge graph.",
            "The diameter of a graph is the maximum distance between any pair of nodes in the graph. To compute this, you need to find the shortest path between all pairs of nodes and then determine the maximum length of these shortest paths."
        ]
        description.append('\n**Problem to Solve**\n')
        description.append("- Entities in this knowledge graph: " + ", ".join(node['name'] for node in graph.nodes.values()))
        description.append("- The relationships between these entities are as follows:")
        for u, v, data in graph.edges(data=True):
            description.append(f" - {graph.nodes[u]['name']} is connected to {graph.nodes[v]['name']} via the relationship {data['relation']}.")
        description.append("Please determine the diameter of this network and output the corresponding path in the following format: [Entity1, Entity2, ..., EntityN].")
        
        return '\n'.join(description)
    
    def generate_example(self, graph, path):
        example = []
        example.append('- Entities in this knowledge graph: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        example.append(f"- The relationships between these entities are as follows:")
        for u, v, data in graph.edges(data=True):
            example.append(f" - {graph.nodes[u]['name']} is connected to {graph.nodes[v]['name']} via the relationship {data['relation']}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"One shortest path corresponding to the diameter in this graph is: [{answer}]")
        return '\n'.join(example)

    @staticmethod
    def exact_solver(graph):
        diameter = nx.diameter(graph)
        for node1 in graph.nodes:
            for node2 in graph.nodes:
                if node1 != node2:
                    if nx.shortest_path_length(graph, node1, node2) == diameter:
                        path = nx.shortest_path(graph, node1, node2)
                        return diameter, path
