import numpy as np
import matplotlib.pyplot as plt
from road import *
import math
from scipy import integrate,special
import time

start_time = time.perf_counter()

np.random.seed(14)

INTERSECTION_DISTANCE = 50 # Meters
VEHICLE_DENSITY = 0.18# Vehicles per meter
TRANSMISSION_RANGE = 20 # Meters
REP = 150# Times
DISTANCES = [50,100, 500, 900]
# DISTANCES = [50, 100, 150, 250, 300, 450,600,900,1000,1500,2000,2500] # Meterss

# Theory Helpers Start

def compute_gamma_params():

    """ Compute k and theta, parameters of the gamma distribution, as described in the paper. """
    
    X1 , _ = integrate.quad(lambda x: (VEHICLE_DENSITY * x * np.exp(-VEHICLE_DENSITY * x)) / (1 - np.exp(-VEHICLE_DENSITY * TRANSMISSION_RANGE)), 0 , TRANSMISSION_RANGE)
    X2 , _ = integrate.quad(lambda x: (VEHICLE_DENSITY * (x**2) * np.exp(-VEHICLE_DENSITY * x)) , 0 , TRANSMISSION_RANGE)

    mean = (1 - np.exp(-VEHICLE_DENSITY * TRANSMISSION_RANGE) * (VEHICLE_DENSITY * TRANSMISSION_RANGE + 1)) / (VEHICLE_DENSITY * np.exp(-VEHICLE_DENSITY * TRANSMISSION_RANGE))
    second_moment = (2 * mean * X1 + X2) / np.exp(-VEHICLE_DENSITY*TRANSMISSION_RANGE)
    # second_moment = 

    k = 1 / ((second_moment / mean**2) - 1) 
    theta = mean / k

    return k, theta

def bond_prob():
    
    """ This function computes the bond probability as derived in the paper. """

    k,theta = compute_gamma_params()
    function1 = lambda x,t:  ( ( x**(k-1) ) * ((np.exp(-x/theta))/((theta**k) * special.gamma(k))) ) * ((VEHICLE_DENSITY * np.exp(-VEHICLE_DENSITY * t))/ (1- np.exp(-VEHICLE_DENSITY * TRANSMISSION_RANGE)))
    p1 , _ = integrate.dblquad(function1, 0 , TRANSMISSION_RANGE ,lambda t: INTERSECTION_DISTANCE - t, np.inf)

    function2 = lambda x,t: (1- np.exp(-2 * VEHICLE_DENSITY * math.sqrt(TRANSMISSION_RANGE**2 -(INTERSECTION_DISTANCE - x - t)**2))) * ( x**(k-1) ) * ((np.exp(-x/theta))/((theta**k) * special.gamma(k))) * ((VEHICLE_DENSITY * np.exp(-VEHICLE_DENSITY * t))/ (1- np.exp(-VEHICLE_DENSITY * TRANSMISSION_RANGE)))
    p2 , _ = integrate.dblquad(function2, 0 , TRANSMISSION_RANGE ,lambda t: INTERSECTION_DISTANCE - t - TRANSMISSION_RANGE, lambda t: INTERSECTION_DISTANCE - t)

    return p1+p2

def recursive_theta(p,x):
    """ One of the recursive expressions involved in computing the theoretical probability """
    if x == 0:
        return 0
    else:
        return p * (p + recursive_theta(p, x-1) * (1 - p))

def recursive_P(p, x):
    """ One of the recursive expressions involved in computing the theoretical probability """
    if x == 0:
        return p
    else:
        return p * ( (p**x) + recursive_P(p, x-1) - (p**x) * recursive_theta(p,x))

# Theory Helpers End



# Simulation Helpers Start

def generate_vehicles():
    # We limit the simulation to a 2d-square lattice of size 2x15
    # This will mean that for each road out of the 30, we generate vehicles according to the exponential distribution
    roads =[]
    x = 0
    y = 0

    # Generate 15 vertical roads
    for _ in range(int(max(DISTANCES)/INTERSECTION_DISTANCE)):
        roads.append( Road( x , -1 , VEHICLE_DENSITY, TRANSMISSION_RANGE) )
        x += INTERSECTION_DISTANCE

    # Generate 2 horizontal roads
    for _ in range(2):
        roads.append( Road( -1 , y , VEHICLE_DENSITY, TRANSMISSION_RANGE) )
        y += INTERSECTION_DISTANCE

    return roads

def create_euclidean_graph(roads):

        # Create the list of vertices:
    vertices = []

    for road in roads:
        for vehicle in road.vehicles:
            if vehicle[1] <= INTERSECTION_DISTANCE:
                vertices.append(vehicle)

    # print(vertices)

    # Create the adgacency matrix: 
    edges = [[0 for _ in range(len(vertices))] for _ in range(len(vertices))]

    for i in range(len(vertices)):
        xi, yi = vertices[i]
        di = math.sqrt(xi**2 + yi**2)  # distance from origin

    for i in range(len(vertices)):
        for j in range(len(vertices)):
            if i == j:
                continue
            dx = vertices[j][0] - vertices[i][0]
            dy = vertices[j][1] - vertices[i][1]
            distance = math.sqrt(dx**2 + dy**2)

            # Only connect if within range *and* moving toward increasing x
            if distance <= TRANSMISSION_RANGE and dx >=0:
                edges[i][j] = 1  # Directed edge from i to j

    # print(edges)

    return vertices, edges

def dfs(vertices, edges, start, distance, visited = None):

    if visited is None:
        visited = set()
    
    # Mark the current node as visited
    visited.add(start)
    
    # Explore neighbors
    for neighbor, is_connected in enumerate(edges[start]):
        if is_connected and neighbor not in visited:
            if vertices[neighbor][0] >= distance and vertices[neighbor][1] == INTERSECTION_DISTANCE:
                return 1
            result = dfs(vertices, edges, neighbor, distance, visited)
            if result == 1:
                return 1  # Propagate the success up

    return 0  # No valid path found

# Simulation Helpers End

def main():

    p = bond_prob()
    theoretical_probs = []

    sim_probabilities = []

    for distance in DISTANCES:

        theoretical_probs.append(recursive_P(p , distance/INTERSECTION_DISTANCE))
    
        successes = 0

        for _ in range(REP):
            roads = generate_vehicles()


            vertices, edges = create_euclidean_graph(roads)

            # Run DFS to check delivery
            success = dfs(vertices, edges, 0 , distance)
            successes += success

        # Calculate and store success probability
        prob = successes / REP
        sim_probabilities.append(prob)

    end_time = time.perf_counter()
    print(f"Program executed in {end_time - start_time:.4f} seconds")


    plt.figure(figsize=(8, 5))
    plt.plot(DISTANCES, theoretical_probs, '--')
    plt.plot(DISTANCES, sim_probabilities, marker='o')
    plt.xlabel("Distance (m)")
    plt.ylabel("Message Delivery Probability")
    plt.title("Figure 7: Message Delivery Probability vs. Distance")
    plt.grid(True)
    # plt.xscale("log")
    plt.show()




if __name__ == '__main__':
    main()