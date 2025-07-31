import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

# First, let's see what vehicles are available
print("Available vehicles:")
try:
    # Try to get vehicle names - this might not work in all AirSim versions
    vehicles = client.listVehicles()
    print(f"Vehicles: {vehicles}")
except:
    print("Could not list vehicles directly")

# Try to get lidar data without specifying vehicle name (uses default)
print("\nTrying to get lidar data from default vehicle...")

def min_distance_from_lidar(points):
    if not points:
        return None
    distances = [
        (points[i] ** 2 + points[i+1] ** 2 + points[i+2] ** 2) ** 0.5
        for i in range(0, len(points), 3)
    ]
    return min(distances) if distances else None

for i in range(10):
    try:
        # Try with the correct vehicle name "SimpleFlight"
        lidar_ground = client.getLidarData(lidar_name="Lidar1", vehicle_name="SimpleFlight")
        lidar_around = client.getLidarData(lidar_name="Lidar2", vehicle_name="SimpleFlight")

        ground_dist = min_distance_from_lidar(lidar_ground.point_cloud)
        around_dist = min_distance_from_lidar(lidar_around.point_cloud)

        print(f"[{i}] Ground Distance: {ground_dist:.2f} m | Nearest Horizontal Obstacle: {around_dist:.2f} m")
    except Exception as e:
        print(f"[{i}] Error: {e}")
    time.sleep(1)