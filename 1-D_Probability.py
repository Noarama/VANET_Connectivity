import numpy as np
import matplotlib.pyplot as plt
from cluster import *
from scipy.stats import gamma
from scipy import integrate
from scipy.special import gammainc

np.random.seed(12)

VEHICLE_DENSITY = [0.012,0.018,0.02]  # vehicles per meter
TRANSMISSION_RANGE = 200  # meters
REP = 100 # Repetitions per transmission range


def generate_vehicle_positions(density):
    """ Generate inter-vehicle distances from an exponential distribution. """
    positions = []
    current_position = 0

    while True:
        inter_vehicle_distance = np.random.exponential(1 / density)
        current_position += inter_vehicle_distance
        positions.append(current_position)

        if len(positions) > 10 * TRANSMISSION_RANGE:
            break

    return np.array(positions)


def one_dim_connectivity(r,d):

    """ Identifies clusters and returns their sizes. """
    clusters = []
    
    for _ in range(REP):
        positions = generate_vehicle_positions(d)
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


def compute_empirical_ccdf(sorted_data):

    """ Compute empirical CCDF. """

    n = len(sorted_data)
    ccdf_values = 1- (np.arange(1, n + 1) / n)

    return ccdf_values

def compute_gamma_ccdf(sorted_data , density):

    """ Compute gamma CCDF as described in the paper. """
    
    X1 , _ = integrate.quad(lambda x: (density * x * np.exp(-density * x)) / (1 - np.exp(-density * TRANSMISSION_RANGE)), 0 , TRANSMISSION_RANGE)
    X2 , _ = integrate.quad(lambda x: (density * (x**2) * np.exp(-density * x)) , 0 , TRANSMISSION_RANGE)

    mean = (1 - np.exp(-density * TRANSMISSION_RANGE) * (density * TRANSMISSION_RANGE + 1)) / (density * np.exp(-density * TRANSMISSION_RANGE))
    second_moment = (2 * mean * X1 + X2) / np.exp(-density*TRANSMISSION_RANGE)
    # second_moment = 

    k = 1 / ((second_moment / mean**2) - 1) 
    theta = mean / k
    print(k)

    return gamma.sf(sorted_data , k , scale = theta)

def main():
    
    plt.figure(figsize=(8, 6))

    for density in VEHICLE_DENSITY:
        cluster_sizes = one_dim_connectivity(TRANSMISSION_RANGE, density)
        
        data = np.sort(cluster_sizes)
        ccdf = compute_empirical_ccdf(data)
        plt.plot(data, ccdf, '--', markersize=4, label=f"Empirical CCDF (λ={density})")
        x = np.linspace(0,max(data),1000)
        gamma_ccdf = compute_gamma_ccdf(x , density)
        plt.plot(x, gamma_ccdf, '-', label=f"Gamma Approx (λ={density})")

    plt.xlabel("Cluster Size (meters)")
    plt.ylabel("CCDF")
    plt.title("Empirical vs Gamma CCDF of Cluster Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
