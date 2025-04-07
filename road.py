import numpy as np

class Road:

    def __init__(self, x, y, density, range, distance, INTERSECTION_DISTANCE):
        self.vehicles = self.generate_vehicles( x , y , density, range, distance, INTERSECTION_DISTANCE)

    def generate_vehicles(self, x , y , density, range , distance, INTERSECTION_DISTANCE):

        """ Generate inter-vehicle distances from an exponential distribution. """
        positions = []
        current_position = 0

        while True:

            if y == -1:
                add = (x , current_position)
                if current_position > 2 * INTERSECTION_DISTANCE:
                    break
            else:
                add = (current_position, y)
                if current_position > distance + INTERSECTION_DISTANCE:
                    break

            inter_vehicle_distance = np.random.exponential(1 / density)
            current_position += inter_vehicle_distance
            positions.append(add)

            
        return np.array(positions)