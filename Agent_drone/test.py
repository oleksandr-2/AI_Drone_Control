import airsim
import numpy as np
import tensorflow as tf
import time
import cv2
import keras
from collections import deque
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras import layers, models
import os
import logging
import random
from scipy.spatial import distance
import monitoring

# # Add these at the beginning of your script, after the imports
# LOOP_DETECTION_WINDOW = 100  # Number of frames to consider for loop detection
# SURVEY_GRID_SIZE = 100  # Size of the grid for surveying (100x100 meters)
# SURVEY_RESOLUTION = 5  # Resolution of the survey grid (5x5 meter cells)
NUM_WAYPOINTS = 50
MAX_ALTITUDE = 30
MIN_ALTITUDE = 2
MAX_AREA = 20000 # 20 km in meters
MAX_WAYPOINT_TIME = 300 # 5 minutes in seconds
WAYPOINTS = []
current_waypoint_index = 0
waypoint_start_time = 0

#flight_path = deque(maxlen=LOOP_DETECTION_WINDOW)
#surveyed_area = np.zeros((SURVEY_GRID_SIZE // SURVEY_RESOLUTION, SURVEY_GRID_SIZE // SURVEY_RESOLUTION), dtype=bool)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

class Metrics:
    def __init__(self, window_size=100):
        self.losses = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.distance = deque(maxlen=window_size)
        self.collision = deque(maxlen=window_size)
        self.episodes = 0
    
    def update(self, loss, reward, distance, collision):
        self.losses.append(loss)
        self.rewards.append(reward)
        self.distance.append(distance)
        self.collision.append(collision)

    def get_average_metrics(self):
        return {
            "avg_loss": np.mean(self.losses),
            "avg_reward": np.mean(self.rewards),
            "avg_distance": np.mean(self.distance),
            "collision_rate": np.mean(self.collision)
        }
    
    def new_episode(self):
        self.episodes += 1

metrics = Metrics()

# Creating AI model
def create_model():
    image_input = keras.Input(shape=(84, 84, 3))
    state_input = keras.Input(shape=(18,))
    # Image processing
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(image_input)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(120, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)
    # State processing
    y = keras.layers.Dense(87, activation='relu')(state_input)
    y= keras.layers.Dropout(0.4)(y)
    y = keras.layers.Dense(64, activation= 'relu')(y)
    combined = keras.layers.concatenate([x, y])
    # LSTM layers for temporal dependencies
    lstm_out = keras.layers.LSTM(128, return_sequences=True)(keras.layers.Reshape((1, -1))(combined))
    lstm_out = keras.layers.LSTM(86, activation= 'relu',return_sequences= True)(lstm_out)
    lstm_out = keras.layers.Dropout(0.2)(lstm_out)
    lstm_out = keras.layers.LSTM(32)(lstm_out)
    #RNN layers 
    # Dense layers
    z = keras.layers.Dense(64, activation='relu')(lstm_out)
    z = keras.layers.Dense(32, activation='relu')(z)
    output = keras.layers.Dense(5, activation='tanh')(z)

    model = keras.Model(inputs=[image_input, state_input], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def interpret_action(action):
    MIN_SAFE_HEIGHT = 2.0
    MAX_HEIGHT = 20.0
    vx, vy, vz, yaw_rate, throttle = action
    vx *= 5
    vy *= 5
    vz *= 2
    # Allow some backward motion without forcing a turn
    if vx < 0:
        vx = max(vx, -2)

    current_height = client.getMultirotorState().kinematics_estimated.position.z_val
    if current_height <= MIN_SAFE_HEIGHT and vz < 0:
        vz = 0
    elif current_height >= MAX_HEIGHT and vz > 0:
        vz = min(0, vz) # Force downward or zero vertical velocity 
    yaw_rate *= 45
    throttle = (throttle + 1) / 2
    return vx, vy, vz, yaw_rate, throttle

def generate_waypoints():
    global WAYPOINTS
    WAYPOINTS = []
    area_side = int(np.sqrt(MAX_AREA))
    for _ in range(NUM_WAYPOINTS):
        x = random.uniform(-area_side / 2, area_side / 2)
        y = random.uniform(- area_side / 2, area_side / 2)
        z = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
        WAYPOINTS.append((x, y, z))
    random.shuffle(WAYPOINTS)

def get_next_waypoint():
    global current_waypoint_index, waypoint_start_time
    if current_waypoint_index >= len(WAYPOINTS):
        generate_waypoints()
        current_waypoint_index = 0
    waypoint = WAYPOINTS[current_waypoint_index]
    current_waypoint_index += 1
    waypoint_start_time = time.time()
    return waypoint

def distance_to_waypoint(position, waypoint):
    return distance.euclidean(position[:3], waypoint)

def get_drone_state():
    try:
        state = client.getMultirotorState()
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = cv2.resize(img_rgb, (84, 84))
        
        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 3:
            lidar_points = np.zeros((5,))
            ground_distance = float('inf')  # No data available
        else:
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_points = np.mean(points[:5], axis=0)
            lidar_points = np.pad(lidar_points, (0, 5 - len(lidar_points)), 'constant')
            ground_distance = np.min(points[:, 2])  # Minimum Z value is the ground distance
        
        position = state.kinematics_estimated.position
        velocity = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        waypoint = get_next_waypoint()
        state_vector = np.array([
            position.x_val, position.y_val, ground_distance,  # Use LiDAR ground distance for Z
            velocity.x_val, velocity.y_val, velocity.z_val,
            orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val,
            *lidar_points
        ])
        state_vector = np.pad(state_vector, (0, 15 - len(state_vector)), 'constant')[:15]
        return img_rgb, state_vector, ground_distance
    except Exception as e:
        logging.error(f"Error getting drone state: {e}")
        return np.zeros((84, 84, 3)), np.zeros((18,)), float('inf')


def calculate_reward(state, prev_state, action, last_action, last_action_count, start_time, current_waypoint):
    global waypoint_start_time
    reward = 0
    try:
        position = state[:3]
        velocity = state[3:6]
        orientation = state[6:10]
    except IndexError as e:
        logging.error(f"Error accessing state components: {e}")
        position = [0, 0, 0]
        velocity = [0, 0, 0]
        orientation = [0, 0, 0, 0]
    
    prev_position = prev_state[:3]
    prev_velocity = prev_state[3:6]
    prev_orientation = prev_state[6:10]
    
    height = position[2]
    collision = client.simGetCollisionInfo().has_collided

    dist_to_waypoint = distance_to_waypoint(position, current_waypoint)
    route_reward = max(-50, min(50, 50 - dist_to_waypoint))
    reward += route_reward
    logging.info(f"Route following reward: {route_reward}")

    # Waypoint reaching reward
    if dist_to_waypoint < 5:
        waypoit_reward = 100
        reward += waypoit_reward
        logging.info(f"Waypoint reached reward: {waypoit_reward}")

        # Time reward
        time_take = time.time() - waypoint_start_time
        time_reward = max(0, 40 * (1 - time_take / MAX_WAYPOINT_TIME))
        reward += time_reward
        logging.info(f"Time reward: {time_reward}")

        current_waypoint = get_next_waypoint()

    # Altitude control
    if height < MIN_ALTITUDE:
        altitude_penalty = -50 * (MIN_ALTITUDE - height) / MIN_ALTITUDE
        reward += altitude_penalty
        logging.info(f"Low altitude penalty: {altitude_penalty}")
    elif height > MAX_ALTITUDE:
        altitude_penalty = -50 * (height - MAX_ALTITUDE) / MAX_ALTITUDE
        reward += altitude_penalty
        logging.info(f"High altitude penalty: {altitude_penalty}")
        if height > MAX_ALTITUDE + 3:
            client.moveToZAsync(MAX_ALTITUDE, 1).join()
            logging.info("Forced descent to maximum allowed altitude")

    # flight_path.append(position)
    # # Update surveyed area
    # update_surveyed_area(position)

    # # REward for completing a loop
    # if detect_loop(flight_path):
    #     loop_reward = 500 # Significant reward for completing a loop
    #     reward += loop_reward
    #     logging.info(f"Reward for completing a loop: {loop_reward}")
    #     flight_path.clear() # Reset flighr path after detecting a loop

     # Reward for surveying new area 
    # survey_completion = calculate_survey_completion()
    # survey_reward = survey_completion * 100  # Example reward calculation
    # reward += survey_reward  # Adding survey reward to the total reward
    # logging.info(f"Rewrd for survey completion ({survey_completion:.2%}): {survey_reward}")

    # Constants
    MIN_SAFE_HEIGHT = 2.0
    #MAX_HEIGHT = 50.0
    MIN_MOVEMENT_DISTANCE = 0.5
    TIME_PENALTY_FACTOR = 0.01
    STABILITY_REWARD_FACTOR = 35.0
    SMOOTH_FLIGHT_REWARD_FACTOR = 35.5

    # # Height control 
    # if height < MIN_SAFE_HEIGHT:
    #     height_penalty = -10 * (MIN_SAFE_HEIGHT - height)
    #     reward += height_penalty
    #     logging.info(f"Penalty for low heighr: {height_penalty}")
    # elif height > MAX_HEIGHT:
    #     height_penalty = -20 * (height - MAX_HEIGHT)
    #     reward += height_penalty
    #     logging.info(f"Penalty for exceeding max height: {height_penalty}")
    # else:
    #     height_penalty = 5 * (1 - abs(height - ((MAX_HEIGHT + MIN_SAFE_HEIGHT) / 2)) / ((MAX_HEIGHT - MIN_SAFE_HEIGHT) / 2))
    #     reward += height_penalty
    #     logging.info(f"Reward for optimal hight: {height_penalty}")

    # # Reward for moving towards the goal
    # distance_to_goal = np.linalg.norm(position - np.array(goal_position))
    # prev_distance_to_goal = np.linalg.norm(prev_position - np.array(goal_position))
    # reward += (prev_distance_to_goal - distance_to_goal) * 5
    # logging.info(f"Reward for moving towards goal: {(prev_distance_to_goal - distance_to_goal) * 5}")

    # Reward for moving a significant distance
    distance_moved = np.linalg.norm(position - prev_position)
    movement_reward = min(distance_moved / MIN_MOVEMENT_DISTANCE, 1) * 3
    reward += movement_reward
    logging.info(f"Reward for movement: {movement_reward}")

    # Penalize for collisions
    if collision:
        reward -= 30
        logging.info(f"Penalty for collision")

    # # Penalize for repeating the same action
    # if np.all(action == last_action):
    #     last_action_count += 1
    #     if last_action_count > 5:  # Threshold for inactivity
    #         penalty = last_action_count * 0.5
    #         reward -= penalty
    #         logging.info(f"Penalty for inactivity: {penalty}")
    # else:
    #     last_action_count = 0
    
    # Reward forward motion and penalize excessive bacrward motion
    forwara_velocity = velocity[0]
    if forwara_velocity > 0:
        forward_reward = 30 * forwara_velocity
        reward += forward_reward
        logging.info(f"Reward for forward motion: {forward_reward}")
    elif forwara_velocity < -2:
        backward_penalty = -20 * abs(forwara_velocity)
        reward += backward_penalty
        logging.info(f"Penalty for excessive bacckward motion: {backward_penalty}")

    # Time penalty to encourage efficiency
    time_penalty = (time.time() - start_time) * TIME_PENALTY_FACTOR
    reward -= time_penalty
    logging.info(f"Penalty for time: {time_penalty}")

    # Reward for long-distance flights
    total_distance_traveled = np.linalg.norm(position - np.array([0, 0, MIN_SAFE_HEIGHT]))
    distance_reward = 0
    if total_distance_traveled > 1000: # 1km
        distance_reward += 100
    elif total_distance_traveled > 500: # 500 m
        distance_reward += 50
    reward += distance_reward
    logging.info(f"Reward for long-distance flight: {distance_reward}")

    # # Reward for specific maneuvers (simplified example, you may need to implement more complex login)
    # if abs(position[0]) > 1000 and abs(position[1]) < 100: # Starting flight
    #     reward += 50
    #     logging.info("Reward for straight flight : 50")
    # elif abs(position[1]) > 1000 and abs(position[0]) < 100: # Side flight
    #     reward += 55
    #     logging.info("Reward for side flight: 55")
    
    # # Loop maneuver detection (simplified, you may nedd more complex logic)
    # if abs(orientation[1]) > 0.9:
    #     reward += 100
    #     logging.info("Reward for loop maneuver: 1000")

    # Reward for stable altitude
    altitude_stability = STABILITY_REWARD_FACTOR * (1 / (1 + abs(height - prev_position[2])))
    reward += altitude_stability
    logging.info(f"Reward for altitude stability: {altitude_stability}")

    # Reward for smooth changes in velocity
    velocity_smoothness = SMOOTH_FLIGHT_REWARD_FACTOR * (1 / (1 + np.linalg.norm(np.array(velocity) - np.array(prev_velocity))))
    reward += velocity_smoothness
    logging.info(f"Reward for velocity smoothness: {velocity_smoothness}")

    # Reward for avoiding rapid changes in orientation
    orientation_smoothness = SMOOTH_FLIGHT_REWARD_FACTOR * (1 / (1 + np.linalg.norm(np.array(orientation) - np.array(prev_orientation))))
    reward += orientation_smoothness
    logging.info(f"Reward for orientation smoothness: {orientation_smoothness}")

    return reward, distance_moved, collision, last_action_count

def lift_drone(target_height):
    while True:
        _, _, current_height = get_drone_state()
        if current_height >= target_height:
            break
        client.moveByVelocityAsync(0, 0, -1, 1).join()  # Note: negative z velocity is upward
        logging.info(f"Current height: {current_height}")

    logging.info(f"Drone lifted to approximately {target_height} meters.")

# def safe_divide(x, y):
#     return x/ y if y != 0 else 0

# # def detect_loop(flight_path):
# #     if len(flight_path) < LOOP_DETECTION_WINDOW:
# #         return False
    
#     start_pos = flight_path[0]
#     end_pos = flight_path[-1]
#     distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    
#     # Check if the drone has returned close to its starting position
#     if distance < 5:  # 5 meters threshold
#         # Check if the path forms a loop
#         path_length = sum(np.linalg.norm(np.array(flight_path[i+1]) - np.array(flight_path[i])) for i in range(len(flight_path)-1))
#         if path_length > 50:  # Minimum loop size of 50 meters
#             return True
#     return False

# def update_surveyed_area(position):
#     x, y = position[:2]
#     grid_x = int((x + SURVEY_GRID_SIZE / 2) // SURVEY_RESOLUTION)
#     grid_y = int((y + SURVEY_GRID_SIZE / 2) // SURVEY_RESOLUTION)
    
#     if 0 <= grid_x < surveyed_area.shape[0] and 0 <= grid_y < surveyed_area.shape[1]:
#         surveyed_area[grid_x, grid_y] = True

# def calculate_survey_completion():
#     return np.sum(surveyed_area) / surveyed_area.size


def train_drone(model, num_episodes=5000):
    global  current_waypoint_index, waypoint_start_time # surveyed_area,
    _, prev_state, _ = get_drone_state()
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.01
    best_reward = float('-inf')

    lift_drone(5)  # Lift drone to 5 meters before starting training
    logging.info("Drone is now under neural network control.")

    generate_waypoints()
    
    for episode in range(num_episodes):
        #surveyed_area.fill(False)
        metrics.new_episode()
        img_rgb, state, ground_distance = get_drone_state()
        last_state = state
        episode_reward = 0
        episode_distance = 0
        collision_count = 0
        last_action = np.zeros(5)
        last_action_count = 0
        done = False
        start_time = time.time()
        current_waypoint = get_next_waypoint()
        
        metrics.new_episode()
        
        while True:
            img_rgb, state = get_drone_state()
            
            logging.info(f"Current state shape: {state.shape}")
            logging.info(f"Image shape: {img_rgb.shape}")
            
            if np.random.rand() < epsilon:
                action = np.random.uniform(-1, 1, size=5)
            else:
                action = model.predict([np.expand_dims(img_rgb, axis=0), np.expand_dims(state, axis=0)])[0]
                action = np.squeeze(action)
            
            vx, vy, vz, throttle = interpret_action(action)
            client.moveByVelocityAsync(vx, vy, vz, throttle).join()
            
            reward, distance_moved, collision, last_action_count = calculate_reward(state, prev_state, action, last_action, last_action_count, start_time, [0, 0, 5])
            episode_reward += reward
            episode_distance += distance_moved
            collision_count += int(collision)
            
            try:
                dummy_targets = np.random.rand(1, 5)  # Create dummy targets for evaluation
                eval_result = model.evaluate(
                    [np.expand_dims(img_rgb, axis=0), np.expand_dims(state, axis=0)],
                    dummy_targets,
                    verbose=0
                )
                loss = eval_result[0]
                mae = eval_result[1]
                logging.info(f"Evaluation - Loss: {loss}, MAE: {mae}")
            except Exception as e:
                loss = float('nan')
                logging.error(f"Error during model evaluation: {e}")
            
            metrics.update(loss, reward, distance_moved, collision)
            
            if collision:
                break
            
            prev_state = state
            last_action = action
        


        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Save model with best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_save_path = "best_model.keras"
            save_model(model, model_save_path)
            logging.info(f"Saved best model with reward: {best_reward}")

        # Save model every 100 episodes
        if (episode + 1) % 100 == 0:
            model_save_path = f"model_episode_{episode + 1}.keras"
            save_model(model, model_save_path)
            logging.info(f"Saved model at episode {episode + 1}")

        logging.info(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward}, Distance: {episode_distance}, Collisions: {collision_count}")
        
        # Log average metrics
        avg_metrics = metrics.get_average_metrics()
        logging.info(f"Average Metrics - Loss: {avg_metrics['avg_loss']}, Reward: {avg_metrics['avg_reward']}, Distance: {avg_metrics['avg_distance']}, Collision Rate: {avg_metrics['collision_rate']}")
        
        metrics.new_episode()
    
    client.armDisarm(False)
    client.enableApiControl(False)
    logging.info("Training complete.")

if __name__ == "__main__":
    if os.path.exists("best_model.keras"):
        model = load_model("model_episode_1700.keras")
        logging.info(f"Loaded existing model: {model}")
    else:
        model = create_model()
        logging.info("Created new model.")
    
    logging.info(f"Model summary:\n{model.summary()}")
    train_drone(model)