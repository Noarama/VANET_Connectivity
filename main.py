import numpy as np
from cluster import *

np.random.seed(42)

VEHICLE_DENSITY = 0.02 # vehicles per meter
TRANSMISSION_RANGE = 200 # meters
ROAD_LENGTH = 2000 # 2km
INTERSECTION_DISTANCE = 500 # meters

def one_dim_connectivity():

    # Generate vehicles along the main road according to the exponential distribution
    vehicle_positions = np.cumsum(np.random.exponential(1/VEHICLE_DENSITY, int(VEHICLE_DENSITY * ROAD_LENGTH)))

    # Remove any vehicles not within the defined road length
    vehicle_positions = vehicle_positions[vehicle_positions < ROAD_LENGTH]

    # Identify the clusters for this setup
    clusters = find_clusters(vehicle_positions)

    # print(clusters[0].cluster_size)


def find_clusters(vehicle_positions):

    clusters = []

    cur_cluster = Cluster()

    for vehicle in vehicle_positions:
        # Condition block to add the first vehicle to the first cluster
        if (vehicle == vehicle_positions[0]):
            cur_cluster.add_to_cluster(vehicle_positions[0])
        
        # This condition block identifies if a vehicle is connected to the current cluster. If so, the vehicle is added to the cluster.
        elif (vehicle <= cur_cluster.last_vehicle+TRANSMISSION_RANGE):
            cur_cluster.add_to_cluster(vehicle)

        # Finally, if there is a vehicle not connected to the cluster, a new cluster begins. 
        else:
            clusters.append(cur_cluster)
            cur_cluster = Cluster()
            cur_cluster.add_to_cluster(vehicle)
    
    # In the case where only one cluster:
    if(not clusters):
        clusters.append(cur_cluster)

    return clusters



def main():
    one_dim_connectivity()



if __name__ == '__main__':
    main()