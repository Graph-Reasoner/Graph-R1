import networkx as nx
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
import itertools
import pandas as pd
import fast_tsp
from .base import *


class TSP_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(TSP_Task, self).__init__(data_loc, 'TSP')
        self.examples = []
        self.example_num = 0
        self.problem_set = {}  # Initialize empty problem set
    
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
        
    def build_graph_from_text(self, problem_text):
        """Build graph from problem text"""
        # #print("Building graph from problem text...")
        g = nx.Graph()
        
        # Parse airport names
        airport_line = re.search(r'- Airports to visit: (.+)', problem_text)
        if airport_line:
            airports = [airport.strip() for airport in airport_line.group(1).split(',')]
            # #print(f"Found airports: {airports}")
            
            # Add nodes
            for i, airport in enumerate(airports):
                g.add_node(i, name=airport)
                
            # Parse distance information
            distance_pattern = re.compile(r'(\w+) to (\w+): (\d+)')
            for match in distance_pattern.finditer(problem_text):
                source, destination, weight = match.groups()
                # source_id = self.find_node_by_name(g, source)
                source_id = find_node_by_name(g, source)
                # dest_id = self.find_node_by_name(g, destination)
                dest_id = find_node_by_name(g, destination)
                
                if source_id is not None and dest_id is not None:
                    g.add_edge(source_id, dest_id, weight=int(weight))
            
            # #print(f"Built graph: Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
            
            # Ensure the graph is fully connected
            for i in range(g.number_of_nodes()):
                for j in range(i+1, g.number_of_nodes()):
                    if not g.has_edge(i, j):
                        # #print(f"Warning: No edge between nodes {i} and {j}, trying shortest path")
                        try:
                            path_length = nx.shortest_path_length(g, i, j, weight='weight')
                            g.add_edge(i, j, weight=path_length)
                        except nx.NetworkXNoPath:
                            # #print(f"Cannot find path between nodes {i} and {j}, adding an edge with large weight")
                            g.add_edge(i, j, weight=99999)  # Add an edge with large weight
            
            return g
        return None
    
    def check_solution(self, problem_id, response, graph=None, problem_text=None):
        #print("\n---TSP CHECK SOLUTION DEBUG---")
        # #print("Problem ID:", problem_id)
        #print("Response:", response)
        
        # Determine which graph to use from provided parameters
        if problem_text:
            # #print("Building graph from problem_text")
            g = self.build_graph_from_text(problem_text)
            if not g:
                # #print("Failed to build graph from problem_text!")
                return -1
        elif graph:
            # #print("Using provided graph parameter")
            # #print(f"- Nodes: {graph.number_of_nodes()}")
            # #print(f"- Edges: {graph.number_of_edges()}")
            g = graph
        else:
            if problem_id not in self.problem_set:
                # #print("Problem ID not found in problem_set!")
                return -1
            g = self.problem_set[problem_id]['graph']
            # #print("Using graph from problem_set")
        
        pattern = re.compile(r'\[\s*([A-Z\s,]*)\s*\]')
        p = pattern.findall(response)
        #print("Patterns found:", p)
        
        if p:
            matches = p[-1]
            matches = matches.split(",")
            # route_list = [self.find_node_by_name(g, node.strip()) for node in matches]
            route_list = [find_node_by_name(g, node.strip()) for node in matches]
            # #print("Route list:", route_list)
            
            # Check if all nodes are valid
            if None in route_list:
                #print("Solution contains invalid airport names")
                return -2
                
            # Check if all nodes are visited exactly once
            if set(route_list[:-1]) == set(g.nodes()) and len(route_list[:-1]) == len(g.nodes()) and route_list[0] == route_list[-1]:
                tour_length = self.compute_tour_length(g, route_list)
                #print(f"Valid route found, length: {tour_length}")
                return tour_length
            #print("Not a valid route - not all nodes visited exactly once or route doesn't return to start")
            return -2
        #print("No solution pattern found")
        return -1
    
    @staticmethod
    def compute_tour_length(graph, route):
        tour_length = 0
        for i in range(len(route) - 1):
            tour_length += graph.get_edge_data(route[i], route[i + 1])['weight']
        return tour_length

    def generate_example(self, graph, path):
        example = []
        example.append('- Airports to visit: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()])+'.')
        example.append("- Travel distances (in kilometers) between each pair of airports:")
        for edge in graph.edges(data=True):
            example.append(f"{graph.nodes[edge[0]]['name']} to {graph.nodes[edge[1]]['name']}: {edge[2]['weight']}")
        answer = ", ".join([graph.nodes[node]['name'] for node in path])
        example.append(f"One shortest route: [{answer}].")
        return '\n'.join(example)

    def exact_solver(self, graph):
        dis_mat = self.build_distance_matrix(graph)
        route = fast_tsp.solve_tsp_exact(dis_mat)
        route.append(route[0])
        return self.compute_tour_length(graph, route), route
    
    def approx_solver(self, graph, method='greedy'):
        if method == 'random':
            route = list(np.random.permutation(graph.nodes()))
            route = route + [route[0]]
        elif method == 'greedy':
            route = nx.approximation.traveling_salesman_problem(graph, cycle=True, weight='weight', method=nx.approximation.greedy_tsp)
        elif method == 'approximated':
            route = nx.approximation.traveling_salesman_problem(graph, cycle=True, weight='weight', method=nx.approximation.christofides)
        return self.compute_tour_length(graph, route), route

    @staticmethod
    def build_distance_matrix(graph):
        n = len(graph)
        dist = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dist[i][j] = graph.get_edge_data(i, j)['weight']
                dist[j][i] = graph.get_edge_data(i, j)['weight']
        return dist