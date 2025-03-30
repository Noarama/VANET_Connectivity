""" This program acts as a simulation for the expected cluster size as presented in [1] """

import numpy as np
from cluster import *
import matplotlib.pyplot as plt

np.random.seed(400)

VEHICLE_DENSITY = 0.015 # vehicles per meter
TRANSMISSION_RANGE = [10,20,40,50,70,100,200,300,500,600,700,800,900] # meters
REP = 10 # times. This is for repititon for each transmission range


def generate_vehicle_positions(r):

    """ This function generates inter-vehicle distances according to an exponential distribution. """

    positions = []
    current_position = 0

    while True:  # Infinite loop to keep generating vehicles
        inter_vehicle_distance = np.random.exponential(1 / VEHICLE_DENSITY)
        # print(inter_vehicle_distance)
        current_position += inter_vehicle_distance
        positions.append(current_position)

        # Stop condition: proportional to the transmission range to speed up algorithm. 
        if len(positions) > 1000 * r: 
            break

    return np.array(positions)

def one_dim_connectivity(r):

    clusters = [] # A list of the clusters for this instance
    cluster_sizes = []
    i = 0

    while i < REP:
        # Generating vehicles by drawing random numbers for the inter-vehicle distances from the exponential distribution.
        # inter_vehicle_distances = np.random.exponential(1/VEHICLE_DENSITY, int(ROAD_LENGTH*VEHICLE_DENSITY))

        # positions = np.cumsum(inter_vehicle_distances)
        # positions = positions[positions < ROAD_LENGTH]

        positions = generate_vehicle_positions(r)

        # At this point we have the list of vehicles. I now need to find the clusters for this instance.

        clusters.extend(identify_clusters(positions, r)) # Adding the clusters to the list of clusters corresponding to the current transmission range.
        i += 1

    # Here I have a list of a lot of clusters corresponding to one transmission range so I should produce the mean of these values and return it to the main 
    count = 0
    size_sum = 0
    for cluster in clusters:
        size_sum += cluster.cluster_size
        count += 1

    return size_sum/count


def identify_clusters(vehicle_positions,cur_transmission):

    """ This function identifies clusters based on the transmission range and the inter-vehicle distances. """

    clusters = [] # List that holds the clusters found for this instance of vehicle positions
    temp_cluster = Cluster()

    temp_cluster.add_to_cluster(vehicle_positions[0])   # Add the first vehicle to the cluster

    # Check remaining vehicles, if they belong to the same cluster as the one with the first vehicle
    for vehicle in vehicle_positions[1:]:
        if vehicle - temp_cluster.last_vehicle < cur_transmission:
            temp_cluster.add_to_cluster(vehicle)

        else:
            clusters.append(temp_cluster)
            temp_cluster = Cluster()
            temp_cluster.add_to_cluster(vehicle)

    clusters.append(temp_cluster)

    return clusters


def theoretical_cluster_size(R):

    """ This function computes the expected value of the cluster size according to the theory of the paper. """

    return (1 - np.exp(-VEHICLE_DENSITY * R) * (VEHICLE_DENSITY * R + 1)) / (VEHICLE_DENSITY * np.exp(-VEHICLE_DENSITY * R))

def main():

    sim_means = [one_dim_connectivity(r) for r in TRANSMISSION_RANGE] # A list storing the simulation cluster size means for each transmission range 
    theory_means = [theoretical_cluster_size(r) for r in TRANSMISSION_RANGE] # A list storing the theoretical cluster size means for each transmission range 

    # Plotting the data

    # Plot simulation results
    plt.figure(figsize=(8, 6))
    plt.plot(TRANSMISSION_RANGE, sim_means, marker="o", linestyle="", label="Simulation", color="blue")
    
    # Plot theoretical results
    plt.plot(TRANSMISSION_RANGE, theory_means, linestyle="--", label="Theoretical E[C]", color="red")

    # Formatting
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Transmission Range (m)")
    plt.ylabel("Mean Cluster Size")
    plt.title("Mean Cluster Size vs. Transmission Range")
    plt.legend()
    plt.grid()
    
    # Show plot
    plt.show()




if __name__ == '__main__':
    main()