import numpy as np
import matplotlib.pyplot as plt
from cluster import *

np.random.seed(400)

VEHICLE_DENSITY = 0.012  # vehicles per meter
TRANSMISSION_RANGE = [200]  # meters
REP = 10  # Repetitions per transmission range


def generate_vehicle_positions(r):
    """ Generate inter-vehicle distances from an exponential distribution. """
    positions = []
    current_position = 0

    while True:
        inter_vehicle_distance = np.random.exponential(1 / VEHICLE_DENSITY)
        current_position += inter_vehicle_distance
        positions.append(current_position)

        if len(positions) > 100 * r:
            break

    return np.array(positions)


def one_dim_connectivity(r):
    """ Identifies clusters and returns their sizes. """
    clusters = []
    
    for _ in range(REP):
        positions = generate_vehicle_positions(r)
        clusters.extend(identify_clusters(positions, r))

    return [cluster.cluster_size for cluster in clusters]


def identify_clusters(vehicle_positions, cur_transmission):
    """ Groups vehicles into clusters based on transmission range. """
    clusters = []
    temp_cluster = Cluster()
    temp_cluster.add_to_cluster(vehicle_positions[0])

    for vehicle in vehicle_positions[1:]:
        if vehicle - temp_cluster.last_vehicle < cur_transmission:
            temp_cluster.add_to_cluster(vehicle)
        else:
            clusters.append(temp_cluster)
            temp_cluster = Cluster()
            temp_cluster.add_to_cluster(vehicle)

    clusters.append(temp_cluster)
    return clusters


def compute_ccdf(data):
    """ Compute empirical CCDF. """
    sorted_data = np.sort(data)
    ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, ccdf


def main():
    # Collect and flatten all cluster sizes
    sim_sizes = [size for r in TRANSMISSION_RANGE for size in one_dim_connectivity(r)]

    # Compute and plot CCDF
    x_values, y_values = compute_ccdf(sim_sizes)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, 'o', markersize=5, label="Empirical CCDF", color="blue")
    plt.xlabel("Cluster Size (meters)")
    plt.ylabel("CCDF")
    plt.title("Empirical CCDF of Cluster Size")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
