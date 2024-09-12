import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting program")
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
logging.info("Imports complete")

logging.basicConfig(level=logging.INFO)

# Constants
NUM_WAYPOINTS = 50
MAX_ALTITUDE = 30
MIN_ALTITUDE = 2
MAX_AREA = 3000 # 20 km in meters
MAX_WAYPOINT_TIME = 300  # 5 minutes in seconds
WAYPOINTS = []
current_waypoint_index = 0
waypoint_start_time = 0


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Attempting to connect to AirSim")
# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
barometer_data = client.getBarometerData()
altitude = barometer_data.altitude
gps_data = client.getGpsData()
altitude_gps = gps_data.gnss.geo_point.altitude



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


def create_model():
    image_input = keras.Input(shape=(84, 84, 3))
    state_input = keras.Input(shape=(18,))
    # Image processing
    x = keras.layers.Conv2D(123, (3, 3), activation='relu')(image_input)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(120, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)
    # State processing
    y = keras.layers.Dense(120, activation='relu')(state_input)
    y= keras.layers.Dropout(0.4)(y)
    y = keras.layers.Dense(84, activation= 'relu')(y)
    y = keras.layers.Dropout(0.3)(y)
    y = keras.layers.Dense(72, activation= 'relu')(y)
    y = keras.layers.Dropout(0.2)(y)
    y = keras.layers.Dense(65, activation= 'relu')(y)
    combined = keras.layers.concatenate([x, y])
    # LSTM layers for temporal dependencies
    lstm_out = keras.layers.LSTM(128, return_sequences=True)(keras.layers.Reshape((1, -1))(combined))
    lstm_out = keras.layers.LSTM(86, activation= 'relu',return_sequences= True)(lstm_out)
    lstm_out = keras.layers.Dropout(0.2)(lstm_out)
    lstm_out = keras.layers.LSTM(32)(lstm_out)

     # Reshape to add the time dimension back
    reshaped_lstm_out = keras.layers.Reshape((1, -1))(lstm_out)
    
    #RNN layers 
    rnn_out = keras.layers.SimpleRNN(64, activation= 'relu', return_sequences= True)(reshaped_lstm_out)
    rnn_out = keras.layers.SimpleRNN(32, activation= 'relu')(rnn_out)
    # Dense layers
    z = keras.layers.Dense(128, activation='relu')(rnn_out)
    z = keras.layers.Dense(64, activation='relu')(z)
    z = keras.layers.Dense(32, activation= 'relu')(z)
    z = keras.layers.Dropout(0.2)(z) 

    output = keras.layers.Dense(5, activation='tanh')(z)

    model = keras.Model(inputs=[image_input, state_input], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

@monitoring.track_function
def clean_and_normalize_data(state_vector, img_rgb):
    state_vector = np.where(state_vector == 0, np.finfo(float).eps, state_vector)

    # Handle infinite values
    state_vector = np.where(np.isinf(state_vector), np.finfo(float).max, state_vector)

    # Normalize state vector
    state_vector = (state_vector - np.mean(state_vector)) / (np.std(state_vector) + np.finfo(float).eps)

    # Normalize image 
    img_rgb = img_rgb / 255.0
    return state_vector, img_rgb

@monitoring.track_function
def interpret_action(action):
    MIN_SAFE_HEIGHT = 2.0
    MAX_HEIGHT = 20.0
    vx, vy, vz, yaw_rate, throttle = action
    vx *= 5
    vy *= 5
    vz *= 2
    if vx < 0:
        vx = max(vx, -2)

    current_height = client.getMultirotorState().kinematics_estimated.position.z_val
    if current_height <= MIN_SAFE_HEIGHT and vz < 0:
        vz = 0
    elif current_height >= MAX_HEIGHT and vz > 0:
        vz = min(0, vz)
    yaw_rate *= 45
    throttle = (throttle + 1) / 2
    return vx, vy, vz, yaw_rate, throttle

@monitoring.track_function
def generate_waypoints():
    global WAYPOINTS
    WAYPOINTS = []
    area_side = int(np.sqrt(MAX_AREA))
    for _ in range(NUM_WAYPOINTS):
        x = random.uniform(-area_side / 2, area_side / 2)
        y = random.uniform(-area_side / 2, area_side / 2)
        z = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
        if np.isinf(x) or np.isinf(y) or np.isinf(z):
            logging.error(f"Infinite value generated for waypoint: {x}, {y}, {z}")
            continue
        WAYPOINTS.append((x, y, z))
    random.shuffle(WAYPOINTS)

@monitoring.track_function
def get_next_waypoint():
    global current_waypoint_index, waypoint_start_time
    if current_waypoint_index >= len(WAYPOINTS):
        generate_waypoints()
        current_waypoint_index = 0
    waypoint = WAYPOINTS[current_waypoint_index]
    current_waypoint_index += 1
    waypoint_start_time = time.time()
    return waypoint

@monitoring.track_function
def distance_to_waypoint(position, waypoint):
    if np.any(np.isnan(position[:3])) or np.any(np.isnan(waypoint)):
        return np.finfo(float).max
    
    if np.any(np.isinf(position[:3])) or np.any(np.isinf(waypoint)):
        return np.finfo(float).max
    
    return distance.euclidean(position[:3], waypoint)

@monitoring.track_function
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
            lidar_points = np.zeros((5,), np.finfo(float).eps)
            ground_distance = np.finfo(float).max
        else:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            lidar_points = np.mean(points[:5], axis=0)
            lidar_points = np.pad(lidar_points, (0, 5 - len(lidar_points)), 'constant', constant_values=np.finfo(float).eps)
            ground_distance = np.clip(np.min(points[:, 2]), 0, np.finfo(float).max)
        
        position = state.kinematics_estimated.position
        velocity = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        state_vector = np.array([
            position.x_val, position.y_val, ground_distance,
            velocity.x_val, velocity.y_val, velocity.z_val,
            orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val,
            *lidar_points
        ])

        state_vector = np.pad(state_vector, (0, 18 - len(state_vector)), 'constant', constant_values=np.finfo(float).eps)
        state_vector, img_rgb = clean_and_normalize_data(state_vector, img_rgb)

        # if np.any(np.isinf(state_vector)):
        #     logging.error(f"Infinite value detected in state vector: {state_vector}")
        #     return np.zeros((84, 84, 3)), np.zeros((18, )), float('inf')

        #state_vector = np.pad(state_vector, (0, 18 - len(state_vector)), 'constant')
        return img_rgb, state_vector, ground_distance
    except Exception as e:
        logging.error(f"Error getting drone state: {e}")
        return np.zeros((84, 84, 3)), np.full((18,), np.finfo(float).eps), np.finfo(float).max

@monitoring.track_function 
def calculate_reward(state, prev_state, last_action_count, start_time, current_waypoint):
    global waypoint_start_time
    reward = 0
    position = state[:3]
    velocity = state[3:6]
    orientation = state[6:10]
    prev_position = prev_state[:3]
    prev_velocity = prev_state[3:6]
    prev_orientation = prev_state[6:10]
    
    height = max(position[2], 0) 
    collision = client.simGetCollisionInfo().has_collided

    dist_to_waypoint = distance_to_waypoint(position, current_waypoint)
    route_reward = np.clip(50 - dist_to_waypoint, -50, 50)
    reward += route_reward
    logging.info(f"Route following reward: {route_reward}")

    if dist_to_waypoint < 5:
        waypoint_reward = 100
        reward += waypoint_reward
        logging.info(f"Waypoint reached reward: {waypoint_reward}")

        time_taken = time.time() - waypoint_start_time
        time_reward = max(0, 40 * (1 - time_taken / MAX_WAYPOINT_TIME))
        reward += time_reward
        logging.info(f"Time reward: {time_reward}")

        current_waypoint = get_next_waypoint()
        logging.info(f"Current waypoint: {current_waypoint}")

    if height < MIN_ALTITUDE:
        altitude_penalty = -35 * (MIN_ALTITUDE - height) / MIN_ALTITUDE
        reward += altitude_penalty
        logging.info(f"Low altitude penalty: {altitude_penalty}")
    elif height > MAX_ALTITUDE:
        altitude_penalty = -50 * (height - MAX_ALTITUDE) / MAX_ALTITUDE
        reward += altitude_penalty
        logging.info(f"High altitude penalty: {altitude_penalty}")
        if height > MAX_ALTITUDE + 3:
            client.moveToZAsync(MAX_ALTITUDE, 1).join()
            logging.info("Forced descent to maximum allowed altitude")

    MIN_MOVEMENT_DISTANCE = 0.5
    TIME_PENALTY_FACTOR = 0.01
    STABILITY_REWARD_FACTOR = 25.9
    SMOOTH_FLIGHT_REWARD_FACTOR = 25.5

    distance_moved = np.linalg.norm(position - prev_position)
    movement_reward = min(distance_moved / MIN_MOVEMENT_DISTANCE, 1) * 3
    reward += movement_reward
    logging.info(f"Reward for movement: {movement_reward}")

    if collision:
        reward -= 30
        logging.info(f"Penalty for collision")

    forward_velocity = velocity[0]
    if forward_velocity > 0:
        forward_reward = 25 * forward_velocity
        reward += forward_reward
        logging.info(f"Reward for forward motion: {forward_reward}")
    elif forward_velocity < -2:
        backward_penalty = -30 * abs(forward_velocity)
        reward += backward_penalty
        logging.info(f"Penalty for excessive backward motion: {backward_penalty}")

    time_penalty = (time.time() - start_time) * TIME_PENALTY_FACTOR
    reward -= time_penalty
    logging.info(f"Penalty for time: {time_penalty}")

    total_distance_traveled = np.linalg.norm(position - np.array([0, 0, MIN_ALTITUDE]))
    distance_reward = 0
    if total_distance_traveled > 1000:
        distance_reward += 100
    elif total_distance_traveled > 500:
        distance_reward += 50
    reward += distance_reward
    logging.info(f"Reward for long-distance flight: {distance_reward}")

    altitude_stability = STABILITY_REWARD_FACTOR * (1 / (1 + abs(height - prev_position[2])))
    reward += altitude_stability
    logging.info(f"Reward for altitude stability: {altitude_stability}")

    velocity_smoothness = SMOOTH_FLIGHT_REWARD_FACTOR * (1 / (1 + np.linalg.norm(np.array(velocity) - np.array(prev_velocity))))
    reward += velocity_smoothness
    logging.info(f"Reward for velocity smoothness: {velocity_smoothness}")

    orientation_smoothness = SMOOTH_FLIGHT_REWARD_FACTOR * (1 / (1 + np.linalg.norm(np.array(orientation) - np.array(prev_orientation))))
    reward += orientation_smoothness
    logging.info(f"Reward for orientation smoothness: {orientation_smoothness}")

    return reward, np.linalg.norm(position - prev_position), collision, last_action_count, current_waypoint

def lift_drone(target_height):
    while True:
        _, _, current_height = get_drone_state()
        if current_height >= target_height:
            break
        client.moveByVelocityAsync(0, 0, -1, 1).join()
        logging.info(f"Current height: {current_height}")
        time.sleep(0.1)  # Small sleep to prevent high CPU usage

    logging.info(f"Drone lifted to approximately {target_height} meters.")

logging.info("Starting drone training")
@monitoring.track_function
def train_drone(model, num_episodes=5000):
    global current_waypoint_index, waypoint_start_time, WAYPOINTS
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.01
    best_reward = float('-inf')

    metrics = Metrics()

    lift_drone(5)
    logging.info("Drone is now under neural network control.")

    generate_waypoints()
    
    for episode in range(num_episodes):
        _, state, _ = get_drone_state()
        monitoring.state_machine.transition("EPISODE_START")
        monitoring.log_event(f"Starting episode {episode + 1}")
        last_state = state
        episode_reward = 0
        episode_distance = 0
        collision_count = 0
        last_action_count = 0
        start_time = time.time()
        current_waypoint = get_next_waypoint()
        
        metrics.new_episode()
        
        while True:
            img_rgb, state, ground_distance = get_drone_state()
            monitoring.data_tracker.log_data_transfer("get_drone_state", "train_drone", 
                                                  {"img_rgb": img_rgb.shape, "state": state, "ground_distance": ground_distance})
            
            logging.info(f"Current state shape: {state.shape}")
            logging.info(f"Image shape: {img_rgb.shape}")
            logging.info(f"Ground distance: {ground_distance}")
            
            if np.random.rand() < epsilon:
                action = np.random.uniform(-1, 1, size=5)
            else:
                action = model.predict([np.expand_dims(img_rgb, axis=0), np.expand_dims(state, axis=0)])[0]
                monitoring.data_tracker.log_data_transfer("model.predict", "train_drone", {"action": action})
                action = np.squeeze(action)
            
            vx, vy, vz, yaw_rate, throttle = interpret_action(action)
            monitoring.data_tracker.log_data_transfer("interpret_action", "train_drone", 
                                                  {"vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate, "throttle": throttle})
            client.moveByVelocityAsync(vx, vy, vz, 1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)).join()
            
            reward, distance_moved, collision, last_action_count, current_waypoint = calculate_reward(
                state, last_state, last_action_count, start_time, current_waypoint
            )
            monitoring.update_monitoring(state, reward, distance_moved)
            metrics.update(0, reward, distance_to_waypoint(state[:3], WAYPOINTS[current_waypoint_index]), collision)
           
            episode_reward += reward
            episode_distance += distance_moved
            collision_count += int(collision)
            
            try:
                dummy_targets = np.random.rand(1, 5)
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
            
            if collision or ground_distance < 0.5:  # Add a check for minimum safe ground distance
                monitoring.log_event("Collision detected", level='warning', details=f"At position: {state[:3]}")
                break
            
            last_state = state
            #last_action = action

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_save_path = "best_model.keras"
            save_model(model, model_save_path)
            logging.info(f"Saved best model with reward: {best_reward}")

        if (episode + 1) % 100 == 0:
            model_save_path = f"model_episode_{episode + 1}.keras"
            save_model(model, model_save_path)
            logging.info(f"Saved model at episode {episode + 1}")

        logging.info(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward}, Distance: {episode_distance}, Collisions: {collision_count}")
        
        avg_metrics = metrics.get_average_metrics()
        logging.info(f"Average Metrics - Loss: {avg_metrics['avg_loss']}, Reward: {avg_metrics['avg_reward']}, Distance: {avg_metrics['avg_distance']}, Collision Rate: {avg_metrics['collision_rate']}")
        
        metrics.new_episode()
    
    client.armDisarm(False)
    client.enableApiControl(False)
    logging.info("Training complete.")

if __name__ == "__main__":
    if os.path.exists("best_model.keras"):
        logging.info("Attempting to load model")
        model = load_model("best_model.keras")
        logging.info("Model loaded successfully")
        logging.info(f"Loaded existing model: {model}")
    else:
        model = create_model()
        logging.info("Created new model.")
    
    logging.info(f"Model summary:\n{model.summary()}")
    train_drone(model)