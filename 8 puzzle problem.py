import random
import time
from itertools import permutations
from queue import PriorityQueue
from collections import deque


# Creates a graph class, which is a Graph of Nodes for the 8-puzzle problem, each Node representing a state
class Graph:
    # Initialize a dict
    def __init__(self):
        self.nodes = dict()

    # Adds a node to the dictionary. The key is the state, the node being a Node
    def add_node(self, node):
        self.nodes[node.state] = node

    # Returns a Node from the graph
    def get_node(self, node):
        return self.nodes.get(node)

    # Returns a Node from the graph by key, with the key being the state
    def get_node_by_state(self, state):
        return self.nodes.get(state)

    # Gets the first node (starting puzzle state)
    def get_first_node(self):
        return self.nodes[next(iter(self.nodes))]

    # Gets the key of a given index
    def get_key_index(self, key):
        for index, k in enumerate(self.nodes):
            if k == key:
                return index


class Node:
    # Node class representing a state in the puzzle
    def __init__(self, state):
        self.state = state
        self.edges = []

    def add_edge(self, node):
        self.edges.append(node)


# Generates a solvable random 8-puzzle problem
def create_8_puzzle_problem():
    puzzle = list(range(9))
    solvable = False
    while not solvable:
        random.shuffle(puzzle)
        solvable = is_solvable(puzzle[:])
    return tuple(puzzle)


# Checks if the given puzzle configuration is solvable
def is_solvable(puzzle):
    inversions = 0
    for i in range(len(puzzle)):
        if puzzle[i] == 0:
            continue
        for j in range(i + 1, len(puzzle)):
            if puzzle[j] == 0:
                continue
            if puzzle[i] > puzzle[j]:
                inversions += 1
    return inversions % 2 == 1


# Generates all possible permutations of a given puzzle state
def create_permutations(state):
    result = []
    for permutation in permutations(state):
        # This saves time later when creating the edges for the graph; the unsolvable states are unreachable anyway,
        # so removing them here will improve our graph generation time
        if is_solvable(permutation):
            result.append(permutation)
    return result


# Creates edges between nodes in the graph
def create_edges(graph):
    for node in graph.nodes.values():
        node.edges = find_edges(node, graph)


# Finds adjacent nodes and create edges for a given node
def find_edges(node, graph):
    state = node.state
    zero_index = state.index(0)
    state_edges = []
    movements = [1, -1, -3, 3]  # Right, left, up, down

    for move in movements:
        new_zero_index = zero_index + move
        if 0 <= new_zero_index < 9 and abs(new_zero_index % 3 - zero_index % 3) <= 1:
            new_state = swap_elements(state, zero_index, new_zero_index)
            state_edges.append(find_node(new_state, graph))
    return state_edges


# Swaps elements at the given indices in a tuple
def swap_elements(tup, idx1, idx2):
    lst = [*tup]
    lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
    return tuple(lst)


# Finds and returns the node containing the given state
def find_node(state, graph):
    return graph.nodes.get(state)


# Prints the state of the puzzle
def print_state(state):
    for i in range(len(state)):
        print(state[i], end=' ')
        if (i + 1) % 3 == 0:
            print("")
    print()


# Calculates the Manhattan distance heuristic (used in BFS)
def calculate_manhattan_distance(node):
    goal_state = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    manhattan_distance = 0
    for i in range(1, 9):  # Exclude 0 as it represents the empty space
        current_index = node.state.index(i)
        goal_index = goal_state.index(i)
        manhattan_distance += abs(current_index % 3 - goal_index % 3) + abs(current_index // 3 - goal_index // 3)
    return manhattan_distance


# Calculates the out-of-place tiles heuristic (used in Astar)
def calculate_out_of_place_tiles(node):
    goal_state = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    misplaced_tiles = sum(1 for i, j in zip(node.state, goal_state) if i != j and i != 0)
    return misplaced_tiles


# Checks if the given node or state matches the goal state
def check_goal(var):
    goal_state = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    if isinstance(var, Node):
        return var.state == goal_state
    else:
        return var == goal_state


# Creates a graph for the 8-puzzle problem
def create_graph():
    g = Graph()
    puzzle = create_8_puzzle_problem()
    for permutation in create_permutations(puzzle):
        node = Node(permutation)
        g.add_node(node)
    create_edges(g)
    return g


# Creates the "graph" and puzzles and benchmarks the time
def create_puzzles(num_puzzles):
    graph = create_graph()
    puzzles = []
    timer_start = time.perf_counter()
    for i in range(num_puzzles):
        puzzles.append(create_8_puzzle_problem())
    print(f"{num_puzzles} puzzles created")
    print(f"Total time to generate {num_puzzles} puzzles: {(time.perf_counter() - timer_start):.4f} seconds")
    return graph, puzzles


def run_and_benchmark_algorithms(graph, puzzles):
    algorithms = ['DFS', 'UCS', 'BFS', 'ASTAR']
    print()
    for algorithm in algorithms:
        print(f"Running {algorithm} on {len(puzzles)} puzzles")
        best_time = float('inf')
        worst_time = float('-inf')
        total_time = 0
        best_nodes_visited = float('inf')
        worst_nodes_visited = float('-inf')
        total_nodes_visited = 0

        for index, puzzle in enumerate(puzzles):
            nodes_visited = 0
            timer_start = time.perf_counter()

            algorithm_func = None
            match algorithm:
                case 'DFS': algorithm_func = dfs
                case 'UCS': algorithm_func = ucs
                case 'BFS': algorithm_func = bfs
                case 'ASTAR': algorithm_func = astar

            nodes_visited += algorithm_func(graph, puzzle)

            elapsed_time = time.perf_counter() - timer_start

            total_time += elapsed_time
            total_nodes_visited += nodes_visited

            if elapsed_time < best_time:
                best_time = elapsed_time
            if elapsed_time > worst_time:
                worst_time = elapsed_time

            if nodes_visited < best_nodes_visited:
                best_nodes_visited = nodes_visited
            if nodes_visited > worst_nodes_visited:
                worst_nodes_visited = nodes_visited

            if (index + 1) % (len(puzzles) // 10) == 0:
                percentage = ((index + 1) / len(puzzles)) * 100
                if percentage == 10:
                    print(f'{percentage}% complete, estimated time to completion is ~{total_time * 10:.2f} seconds')
                else:
                    print(f"{percentage:.0f}%", end=", ")

        print(f"of the puzzles processed for {algorithm}")

        average_time = total_time / len(puzzles)
        average_nodes_visited = total_nodes_visited // len(puzzles)

        print(f"Algorithm: {algorithm}")
        print(f"Best case nodes visited: {best_nodes_visited}")
        print(f"Worst case nodes visited: {worst_nodes_visited}")
        print(f"Average nodes visited: {average_nodes_visited}")
        print(f"Best case execution time: {best_time:.4f}")
        print(f"Worst case execution time: {worst_time:.4f}")
        print(f"Average execution time: {average_time:.4f}")
        print(f"Total execution time: {total_time:.4f}")
        print()


# Depth First Search algorithm
# Performs depth-first search traversal on a graph
# It starts from the initial node and explores as far as possible along each branch before backtracking
# This implementation uses a stack (implemented as a deque) to keep track of the frontier,
# making it suitable for quickly exploring deeper nodes
# It maintains a set of visited nodes to avoid revisiting already explored states
# If the goal state is found, it returns the number of visited nodes
def dfs(g, puzzle):
    # Get the initial node from the graph
    node = g.get_node(puzzle)

    # Initialize the frontier with the initial node
    frontier = deque()
    frontier.append(node)

    # Create a set to store visited nodes
    visited = set()

    # While the frontier is not empty
    while frontier:
        # Pop the next node from the frontier
        node = frontier.pop()

        # If the current node is the goal state, return the number of visited nodes
        if check_goal(node.state):
            return len(visited)

        # If the node has not been visited yet
        if node not in visited:
            # Mark the node as visited
            visited.add(node)

            # Add unvisited adjacent nodes to the frontier
            for edge in node.edges:
                frontier.append(edge)


# Uniform Cost Search algorithm
# Explores nodes in increasing order of cost from the start node
# Uses a priority queue to prioritize nodes with lower costs
# Maintains a set of visited nodes to avoid revisiting already explored states
# If the goal state is found, returns the number of visited nodes
def ucs(g, puzzle):
    # Initialize frontier as a priority queue
    frontier = PriorityQueue()

    # Initialize the start node, cost, entry num (to break cost ties), and add the start node to the frontier
    node = g.get_node(puzzle)
    cost = 0
    entry_num = 0
    frontier.put((cost, entry_num, node))

    # Create a set of visited states
    visited_nodes = set()

    # While frontier is not empty
    while frontier:
        # Get the node with the lowest depth from the frontier
        cost, _, current_node = frontier.get()

        visited_nodes.add(current_node.state)
        # If the current node is the goal state, return the number of visited nodes
        if check_goal(current_node):
            return len(visited_nodes)

        # Increment cost for neighboring nodes
        cost += 1

        # Explore neighboring nodes
        for edge in current_node.edges:
            # If neighbor is not visited
            if edge.state not in visited_nodes:
                entry_num += 1

                # Push neighbor to the frontier with updated cost
                frontier.put((cost, entry_num, edge))


# Best First Search algorithm
# Explores nodes level by level, starting from the initial node
# Uses a priority queue for ordered insertion to ensure that nodes at shallower depths are explored first
# Maintains a set of visited nodes to avoid revisiting already explored states
# If the goal state is found, returns the number of visited nodes
def bfs(g, puzzle):
    # Initialize the frontier as a PriorityQueue
    frontier = PriorityQueue()

    # Initialize the start node, entry num (to break cost ties), and add the start node to the frontier
    node = g.get_node(puzzle)
    entry_num = 0
    frontier.put((calculate_manhattan_distance(node), entry_num, node))

    # Create a set of visited states
    visited_nodes = set()

    # While there are nodes in the frontier
    while not frontier.empty():
        # Get the node with the lowest manhattan distance from the frontier
        distance, _, current_node = frontier.get()

        # Mark the current node as visited
        visited_nodes.add(current_node)

        # If the current node is the goal state, return the number of visited nodes
        if check_goal(current_node):
            return len(visited_nodes)

        # For each neighbor, if it has not been visited, increment the entry num and add the neighbor to the frontier
        for edge in current_node.edges:
            if edge not in visited_nodes:
                entry_num += 1
                frontier.put((calculate_manhattan_distance(edge), entry_num, edge))


# A* algorithm
# Finds the shortest path from start to goal node
# Uses a priority queue to prioritize nodes based on their total cost f = g + h
# where g is the cost from the start node and h is the heuristic cost to the goal node
# Maintains a set of visited nodes to avoid revisiting already explored states
# If the goal state is found, returns the number of visited nodes
def astar(g, puzzle):
    # Initialize the frontier as a priority queue
    frontier = PriorityQueue()

    # Initialize the start node, entry num (to break cost ties), and add the start node to the frontier
    node = g.get_node(puzzle)
    entry_num = 0
    frontier.put((0, entry_num, node))

    # Create a set of visited states
    visited_nodes = set()

    # While frontier is not empty
    while frontier:
        # Get the node with the lowest f from the frontier
        g, _, current_node = frontier.get()

        # Increment g
        g += 1

        # Add the current node to visited set
        visited_nodes.add(current_node.state)

        # If the current node is the goal state, return the number of visited nodes
        if check_goal(current_node):
            return len(visited_nodes)

        # For each neighbor, calculate the cost 'g' from start to neighbor (this is the distance)
        for edge in current_node.edges:

            # If the neighbor has not been visited
            if edge.state not in visited_nodes:
                # Increment the entry num
                entry_num += 1

                # Calculate heuristic 'h' (this is the out-of-place tiles)
                h = calculate_out_of_place_tiles(edge)

                # Calculate the total cost f = g + h, and push the neighbor to the frontier
                f = g + h
                frontier.put((f, entry_num, edge))


# Main function to create graphs and run algorithms
def main():
    graph, puzzles = create_puzzles(100)
    run_and_benchmark_algorithms(graph, puzzles)


if __name__ == "__main__":
    main()
