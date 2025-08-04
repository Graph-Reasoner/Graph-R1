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
from tqdm import tqdm
# from tasks.base import *
from .base import *
class Connected_Task(NPTask):
    def __init__(self, data_loc='dataset', task_name='Connected'):
        super(Connected_Task, self).__init__(data_loc, task_name)
        self.examples = []
    
    def check_solution(self, problem_id=None, response=None, graph=None, problem_text=None):
        """Check the solution for Connected Components problem."""
        #print("\n---Connected CHECK SOLUTION DEBUG---")
        # #print("Problem ID:", problem_id)
        #print("Response:", response)
        
        if graph:
            # #print("Using provided graph:")
            # print(f"- Nodes: {graph.number_of_nodes()}")
            # print(f"- Edges: {graph.number_of_edges()}")
            # print("- Sample node names:", [graph.nodes[n]["name"] for n in list(graph.nodes())[:3]])
            g = graph
        else:
            if problem_id not in self.problem_set:
                # print("Problem ID not found in problem_set!")
                return -1
            g = self.problem_set[problem_id]['graph']
            # print("Using problem_set graph")

        pattern = re.compile(r'\[(.*?)\]')
        p = pattern.findall(response)
        print("Found patterns:", p)
        
        if p:
            matches = p[-1]
            matches = matches.split(",")
            name_list = [name.strip() for name in matches]
            print("Name list:", name_list)
            
            node_list = []
            for name in name_list:
                # node = self.find_node_by_name(g, name)
                node = find_node_by_name(g, name)
                # #print(f"Looking up node for name {name}: {node}")
                if node is None:
                    continue
                node_list.append(node)
            
            # #print("Final node list:", node_list)
            if not node_list:
                #print("No valid nodes found!")
                return -2
                
            # 验证每个节点是否来自不同连通分量
            components = list(nx.connected_components(g))
            node_to_component = {}
            for i, comp in enumerate(components):
                for node in comp:
                    node_to_component[node] = i
            
            found_components = set()
            for node in node_list:
                if node in node_to_component:
                    found_components.add(node_to_component[node])
            
            if len(found_components) == len(node_list):
                #print(f"Valid solution - found {len(found_components)} distinct components")
                return len(found_components)
            else:
                #print("Invalid solution - some nodes from same component")
                return -2
        #print("No solution pattern found")
        return -1
        
    # def find_node_by_name(self, graph, name):
    #     """查找具有给定名称的节点ID"""
    #     try:
    #         for node_id, node_data in graph.nodes(data=True):
    #             if node_data.get("name", "").strip() == name.strip():
    #                 return node_id
    #         return None
    #     except:
    #         # #print(f"Error finding node with name: {name}")
    #         return None

    def is_feasible(self, g, node_list): 
        """检查节点列表中的节点是否来自不同的连通分量"""
        if not node_list:
            return False
            
        # 获取连通分量
        components = list(nx.connected_components(g))
        
        # 创建从节点到组件索引的映射
        node_to_component = {}
        for i, comp in enumerate(components):
            for node in comp:
                node_to_component[node] = i
                
        # 检查每个节点所在的组件
        found_components = set()
        for node in node_list:
            if node in node_to_component:
                comp_idx = node_to_component[node]
                if comp_idx in found_components:
                    # 如果这个组件已经被表示了，则不可行
                    return False
                found_components.add(comp_idx)
        
        # 所有节点都来自不同的组件
        return True
        
    def generate_problem(self, graph):
        description = []
        description.append("You are required to identify all connected components in the given social network and output one representative node from each component.")
        description.append("Within a connected component, any node can be reached from any other node through the edges in the graph. Different connected components are isolated from each other.")
        description.append('\n**Problem to Solve**\n')
        description.append("- Names in the network: " + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        description.append('- Fiendship connections: ' + ", ".join([f"{graph.nodes[u]['name']} to {graph.nodes[v]['name']}" for u,v,data in graph.edges(data=True)]))
        description.append("Identify all connected components in this network. Note that for each connected component, you should only output one of its nodes.")
        description.append('Present your answer in the following format: [UserA, UserB, UserC, UserD, ...]')
        return '\n'.join(description)
    
    def generate_example(self, graph, path):
        example = []
        example.append('- Names in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        relations = ", ".join([f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u,v,data in graph.edges(data=True)])
        example.append(f"- Friendship connections: {relations}.")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"The answer including one representative element from each connected component in the given social network: [{answer}]")
        return '\n'.join(example)
    
    def generate_dataset(self, count=500):             
        G = pickle.load(open('./source/social_network_union.pkl', 'rb'))
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)
        for difficulty in ['easy', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (4, 14) if difficulty == 'easy' else (15, 30)
            
            while len(self.problem_set) < count:
                node_size = sample_node_size(min_nodes, max_nodes)
                c = Counter()
                for e in range(random.randint(1, 1+node_size//5)):
                    c.update(all_walks[random.randint(0, G.number_of_nodes()-1)])
  
                node_list = [k for k, v in c.most_common(node_size)]
                if len(node_list) < node_size:
                    continue       
                H = nx.induced_subgraph(G, node_list).copy()
                
                exact_answer, path = self.exact_solver(H)  
                if len(self.examples) < 100:
                    self.examples.append(self.generate_example(H, path))
                    continue
                    
                self.problem_set.append({
                    'id' : len(self.problem_set),
                    'problem_text' : self.generate_problem(H),
                    'graph': H,
                    'path': path,
                    'exact_answer': exact_answer,
                })
            self.save_dataset(difficulty)

    @staticmethod
    def exact_solver(graph): 
        connected_num = nx.number_connected_components(graph)
        components = nx.connected_components(graph)
        representative_nodes = [list(comp)[0] for comp in components]
        return connected_num, representative_nodes