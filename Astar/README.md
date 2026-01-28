A*(start, goal):

OPEN = {start}        // nodes to be explored
CLOSED = {}           // nodes already explored

g(start) = 0
h(start) = heuristic(start, goal)
f(start) = g(start) + h(start)

while OPEN is not empty:

    n = node in OPEN with lowest f(n)

    if n == goal:
        return path from start to goal

    remove n from OPEN
    add n to CLOSED

    for each neighbor m of n:

        if m in CLOSED:
            continue

        tentative_g = g(n) + cost(n, m)

        if m not in OPEN:
            add m to OPEN
        else if tentative_g >= g(m):
            continue

        parent(m) = n
        g(m) = tentative_g
        h(m) = heuristic(m, goal)
        f(m) = g(m) + h(m)

return "No path found"

