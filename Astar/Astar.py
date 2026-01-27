import heapq

def a_star(graph, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_cost = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, cost in graph[current]:
            tentative_g = g_cost[current] + cost

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_cost, neighbor))

    return None


# ---------------- GRAPH ----------------
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

# ------------- HEURISTIC --------------
def heuristic(node, goal):
    h = {
        'A': 7,
        'B': 6,
        'C': 2,
        'D': 0
    }
    return h[node]


# ----------- USER INPUT ---------------
start = input("Enter start node: ").upper()
goal = input("Enter goal node: ").upper()

if start not in graph or goal not in graph:
    print("Invalid start or goal node")
else:
    path = a_star(graph, start, goal, heuristic)
    if path:
        print("Shortest Path:", path)
    else:
        print("No path found")

