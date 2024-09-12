import airsim

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Arm and takeoff the drone
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Fly the drone to a specific position
client.moveToPositionAsync(10, 10, -10, 5).join()

# Land the drone
client.landAsync().join()

# Disarm and release control
client.armDisarm(False)
client.enableApiControl(False)

