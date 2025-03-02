class Cluster:
    connected_vehicles = []
    last_vehicle = -1
    first_vehicle = -1
    cluster_size = -1

    def __init__(self):
        self.connected_vehicles = []
        self.last_vehicle = -1
        self.first_vehicle = -1
        self.cluster_size = -1

    def add_to_cluster(self, new_vehicle):
        if(not self.connected_vehicles):
            self.first_vehicle = new_vehicle

        self.connected_vehicles.append(new_vehicle)
        self.last_vehicle = new_vehicle
        self.cluster_size = self.last_vehicle - self.first_vehicle

    def print_cluster(self):
        print(self.connected_vehicles)