# rover_astar_sim.py
import heapq
import math
import time
from typing import Callable, Tuple, List, Optional, Any

def neighbors_8(u, H, W):
    r, c = u
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r+dr, c+dc
            if 0 <= rr < H and 0 <= cc < W:
                yield (rr, cc)

def euclid(a, b):
    ar, ac = a; br, bc = b
    return math.hypot(ar-br, ac-bc)

def reconstruct(parent, goal):
    path = []
    v = goal
    while v is not None:
        path.append(v)
        v = parent.get(v)
    path.reverse()
    return path

def astar(
    start: Tuple[int,int],
    goal: Tuple[int,int],
    neighbors_fn: Callable[[Tuple[int,int]], Any],
    edge_cost_fn: Callable[[Tuple[int,int], Tuple[int,int]], Optional[float]],
    heuristic_fn: Callable[[Tuple[int,int], Tuple[int,int]], float],
    *,
    weight: float = 1.0,          # >1.0 -> Weighted A* (f = g + weight*h)
    epsilon: Optional[float] = None, # (1+ε)-optimal early stop after a goal candidate is found
    max_expansions: Optional[int] = None,
    max_time_sec: Optional[float] = None,
    beam_width: Optional[int] = None # Keep only top-K frontier entries (aggressive)
):
    """
    Returns:
      path, total_cost, expansions, expanded_order(list), frontier_snaps(None placeholder)

    Speedups:
      - Closed set prevents re-expansion.
      - Weighted A* reduces expansions (sacrifices optimality if weight>1).
      - (1+epsilon)-early stop terminates once a good-enough goal path is seen.
      - Optional caps (time/expansions) and beam search.

    Notes:
      * With epsilon not None, we stop once best_goal_cost <= (1+ε) * min_f_in_OPEN.
      * With weight>1.0, solution may be suboptimal but usually found much faster.
    """
    if start == goal:
        return [start], 0.0, 0, [start], None

    t0 = time.time()
    counter = 0  # stable tie-breaker

    # open heap of tuples: (f, h, counter, node)
    openh: List[Tuple[float, float, int, Tuple[int,int]]] = []
    h0 = heuristic_fn(start, goal)
    heapq.heappush(openh, (h0 * weight, h0, counter, start))

    g = {start: 0.0}
    parent = {start: None}
    closed = set()

    expansions = 0
    expanded_order = []

    # Track best goal seen even if not yet popped
    best_goal_cost = None
    best_goal_node = None

    while openh:
        if max_time_sec is not None and (time.time() - t0) > max_time_sec:
            # Return best goal so far if we have one, else fail fast
            if best_goal_node is not None:
                return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None
            return None, float("inf"), expansions, expanded_order, None

        f, h, _, u = heapq.heappop(openh)

        # Early-stop check using (1+epsilon)-admissible bound if we already saw a goal path
        if epsilon is not None and best_goal_cost is not None:
            # min f in OPEN is current popped f (because we always pop the min)
            min_f = f
            if best_goal_cost <= (1.0 + epsilon) * min_f:
                return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None

        if u in closed:
            continue
        closed.add(u)

        expanded_order.append(u)
        expansions += 1
        if max_expansions is not None and expansions >= max_expansions:
            if best_goal_node is not None:
                return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None
            return None, float("inf"), expansions, expanded_order, None

        if u == goal:
            # Optimal for standard A*; with weight>1.0 this is just the best found at pop time
            return reconstruct(parent, u), g[u], expansions, expanded_order, None

        gu = g[u]

        # Expand neighbors
        for v in neighbors_fn(u):
            c = edge_cost_fn(u, v)
            if c is None:
                continue
            alt = gu + c

            # If we've already finalized v, skip
            if v in closed:
                continue

            # Relax
            old = g.get(v)
            if old is None or alt < old - 1e-12:
                g[v] = alt
                parent[v] = u
                hv = heuristic_fn(v, goal)
                counter += 1
                heapq.heappush(openh, (alt + weight*hv, hv, counter, v))

                # Track best goal cost as soon as goal is discovered (generated), not only when popped
                if v == goal:
                    if (best_goal_cost is None) or (alt < best_goal_cost - 1e-12):
                        best_goal_cost = alt
                        best_goal_node = v

        # Optional beam pruning (keep top-K in OPEN)
        if beam_width is not None and len(openh) > beam_width:
            # Take the best K entries; heap is not sorted, so nlargest with negative f or sort
            openh = heapq.nsmallest(beam_width, openh)
            heapq.heapify(openh)

    # If we drained OPEN, return best seen goal if any
    if best_goal_node is not None:
        return reconstruct(parent, best_goal_node), best_goal_cost, expansions, expanded_order, None

    return None, float("inf"), expansions, expanded_order, None
