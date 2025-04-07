import numpy as np

class Road:

    def __init__(self, x, y, density, range):
        self.vehicles = self.generate_vehicles( x , y , density, range)

    def generate_vehicles(self, x , y , density, range):

        """ Generate inter-vehicle distances from an exponential distribution. """
        positions = []
        current_position = 0

        while True:

            if y == -1:
                add = (x , current_position)
            else:
                add = (current_position, y)

            inter_vehicle_distance = np.random.exponential(1 / density)
            current_position += inter_vehicle_distance
            positions.append(add)

            if len(positions) > 10 * range:
                break
        return np.array(positions)