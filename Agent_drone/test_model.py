import logging
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
import msgpack
logging.info("Imports complete")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_ALTITUDE = 30
MIN_ALTITUDE = 2
MIN_MOVEMENT_DISTANCE = 0.5
STABILITY_REWARD_FACTOR = 5
SMOOTH_FLIGHT_REWARD_FACTOR = 5
SQUARE_SIZE = 100
COVERAGE_RESOLUTION = 10
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LIDAR_RANGE = 20

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Attempting to connect to AirSim")
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

class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
    
    def store(self, experience):
        self.memory.append(experience)
    
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        img_rgb, states, lidar_data, actions, rewards, next_img_rgb, next_states, next_lidar_data = zip(*experiences)
        return img_rgb, states, lidar_data, actions, rewards, next_img_rgb, next_states, next_lidar_data
    
    def __len__(self):
        return len(self.memory)
    
# def collect_experience(state, action, reward, next_state, done):
#     experience = (state, action, reward, next_state, done)
#     experience_replay.store(experience)




def create_model():
    # Inputs
    image_input = keras.Input(shape=(84, 84, 3))
    state_input = keras.Input(shape=(20,))
    
    # Image processing branch
    x1 = keras.layers.Conv2D(123, (3, 3), activation='relu')(image_input)
    x2 = keras.layers.MaxPooling2D((2, 2))(x1)
    x3 = keras.layers.Conv2D(120, (3, 3), activation='relu')(x2)
    x4 = keras.layers.MaxPooling2D((2, 2))(x3)
    x5 = keras.layers.Conv2D(64, (3, 3), activation='relu')(x4)
    x6 = keras.layers.Flatten()(x5)
    x7 = keras.layers.Dropout(0.3)(x6)
    
    # State processing branch
    y1 = keras.layers.Dense(120, activation='relu')(state_input)
    y2 = keras.layers.Dropout(0.4)(y1)
    y3 = keras.layers.Dense(84, activation='relu')(y2)
    y3 = keras.layers.Dense(97, activation= 'relu')(y3)
    y4 = keras.layers.Dropout(0.3)(y3)
    y5 = keras.layers.Dense(72, activation='relu')(y4)
    y6 = keras.layers.Dropout(0.2)(y5)
    y7 = keras.layers.Dense(65, activation='relu')(y6)
    
    # Combining image and state branches
    combined = keras.layers.concatenate([x7, y7])
    
    # LSTM layers for temporal dependencies
    lstm_in = keras.layers.Reshape((1, -1))(combined)
    lstm_out1 = keras.layers.LSTM(130, return_sequences=True)(lstm_in)
    lstm_out1 = keras.layers.LSTM(120, return_sequences= True)(lstm_out1)
    lstm_out2 = keras.layers.LSTM(92, return_sequences=True)(lstm_out1)
    lstm_out3 = keras.layers.Dropout(0.2)(lstm_out2)
    lstm_out4 = keras.layers.LSTM(45)(lstm_out3)
    
    # Reshape to add the time dimension back
    reshaped_lstm_out = keras.layers.Reshape((1, -1))(lstm_out4)
    
    # RNN layers
    rnn_out1 = keras.layers.SimpleRNN(124, activation='relu', return_sequences=True)(reshaped_lstm_out)
    rnn_out1 = keras.layers.SimpleRNN(80, activation= 'relu', return_sequences= True)(rnn_out1)
    rnn_out2 = keras.layers.SimpleRNN(65, activation='relu')(rnn_out1)
    
    # Dense layers
    z1 = keras.layers.Dense(128, activation='relu')(rnn_out2)
    z2 = keras.layers.Dense(80, activation='relu')(z1)
    z3 = keras.layers.Dense(45, activation='relu')(z2)
    z4 = keras.layers.Dropout(0.2)(z3)
    
    # Output layer
    output = keras.layers.Dense(5, activation='tanh')(z4)
    
    # Model creation
    model = keras.Model(inputs=[image_input, state_input], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

@monitoring.track_function
def clean_and_normalize_data(state_vector, img_rgb, lidar_data):
    state_vector = np.clip(state_vector, -1e6, 1e6)  # Clip to prevent extreme values
    state_vector = (state_vector - np.mean(state_vector)) / (np.std(state_vector) + 1e-8)
    img_rgb = img_rgb / 255.0
    lidar_data = lidar_data / LIDAR_RANGE  # Normalize lidar data to [0, 1] range
    return state_vector, img_rgb, lidar_data

@monitoring.track_function
def interpret_action(action):
    MIN_SAFE_HEIGHT = 2.0
    MAX_HEIGHT = 20.0
    vx, vy, vz, yaw_rate, throttle = action
    vx = np.clip(vx * 2, -2, 2)  # Limit max speed
    vy = np.clip(vy * 2, -2, 2)
    vz = np.clip(vz, -1, 1)
    if vx < 0:
        vx = max(vx, -2)

    current_height = client.getMultirotorState().kinematics_estimated.position.z_val
    if current_height <= MIN_SAFE_HEIGHT and vz < 0:
        vz = 0
    elif current_height >= MAX_HEIGHT and vz > 0:
        vz = min(0, vz)
    yaw_rate = np.clip(yaw_rate * 45, -45, 45)
    throttle = (throttle + 1) / 2  # Convert from [-1, 1] to [0, 1]
    return vx, vy, vz, yaw_rate, throttle

# @monitoring.track_function
# def generate_waypoints():
#     global WAYPOINTS
#     WAYPOINTS = []
#     area_side = int(np.sqrt(MAX_AREA))
#     for _ in range(NUM_WAYPOINTS):
#         x = random.uniform(-area_side / 2, area_side / 2)
#         y = random.uniform(-area_side / 2, area_side / 2)
#         z = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
#         if np.isinf(x) or np.isinf(y) or np.isinf(z):
#             logging.error(f"Infinite value generated for waypoint: {x}, {y}, {z}")
#             continue
#         WAYPOINTS.append((x, y, z))
#     random.shuffle(WAYPOINTS)

# @monitoring.track_function
# def get_next_waypoint():
#     global current_waypoint_index, waypoint_start_time
#     if current_waypoint_index >= len(WAYPOINTS):
#         generate_waypoints()
#         current_waypoint_index = 0
#     waypoint = WAYPOINTS[current_waypoint_index]
#     current_waypoint_index += 1
#     waypoint_start_time = time.time()
#     return waypoint

# @monitoring.track_function
# def distance_to_waypoint(position, waypoint):
#     if np.any(np.isnan(position[:3])) or np.any(np.isnan(waypoint)):
#         return np.finfo(float).max
    
#     if np.any(np.isinf(position[:3])) or np.any(np.isinf(waypoint)):
#         return np.finfo(float).max
    
#     return distance.euclidean(position[:3], waypoint)
@monitoring.track_function
def generate_patrol_square(drone_position):
    center_x = drone_position.x_val
    center_y = drone_position.y_val
    half_size = SQUARE_SIZE / 2
    return {
        'min_x': center_x - half_size,
        'max_x': center_x + half_size,
        'min_y': center_y - half_size,
        'max_y': center_y + half_size
    }

def numpy_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def visualize_patrol_square(client, current_square):
    """Visualize the patrol square in the AirSim environment."""
    # Create the vertices of the patrol square
    vertices = np.array([
        [current_square['min_x'], current_square['min_y'], 0],
        [current_square['max_x'], current_square['min_y'], 0],
        [current_square['max_x'], current_square['max_y'], 0],
        [current_square['min_x'], current_square['max_y'], 0],
        [current_square['min_x'], current_square['min_y'], 0]  # Close the polygon
    ])

    # Convert vertices to AirSim Vector3r format
    airsim_vertices = [airsim.Vector3r(v[0], v[1], v[2]) for v in vertices]

    # Plot the line strip in the AirSim environment
    client.simPlotLineStrip(airsim_vertices, color_rgba=[1, 0, 0, 0.5], thickness=0.5, duration=0)


@monitoring.track_function
def get_drone_state(client, current_square):
    try:
        state = client.getMultirotorState()
        gps_data = client.getGpsData()
        barometer_data = client.getBarometerData()
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = cv2.resize(img_rgb, (84, 84))

        lidar_data = client.getLidarData()
        lidar_distances = np.array([point.distance for point in lidar_data.point_cloud if point.is_valid])

        position = state.kinematics_estimated.position
        velocity = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        gps_location = gps_data.gnss.geo_point
        altitude = barometer_data.altitude

        relative_x = (position.x_val - current_square['min_x']) / SQUARE_SIZE
        relative_y = (position.y_val - current_square['min_y']) / SQUARE_SIZE

        state_vector = np.array([
            relative_x, relative_y, gps_location.latitude, gps_location.longitude, altitude,
            position.x_val, position.y_val, position.z_val,
            velocity.x_val, velocity.y_val, velocity.z_val,
            orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val,
            *lidar_distances  # Placeholder for lidar data
        ])

        state_vector, img_rgb, lidar_data = clean_and_normalize_data(state_vector, img_rgb, lidar_distances)
        return img_rgb, state_vector, lidar_data, position.z_val
    except Exception as e:
        logging.error(f"Error getting drone state: {e}")
        return np.zeros((84, 84, 3)), np.zeros((20,)), np.zeros(LIDAR_RANGE), np.inf

def extract_scalars(arr):
    # Ensure the input is a numpy array or list
    if isinstance(arr, np.ndarray) or isinstance(arr, list):
        flat_list = []
        for item in np.ravel(arr):  # Flatten the input
            if isinstance(item, (int, float)):  # Check for scalars
                flat_list.append(float(item))  # Convert to float
            else:
                raise ValueError("Element in input array is not a recognized scalar type.")
        return np.array(flat_list)
    else:
        raise ValueError("Input is not a recognized list or array.")

@monitoring.track_function
def calculate_reward(state, prev_state, current_square, lidar_data):
    reward = 0
    position = state[5:8]
    velocity = state[8:11]
    prev_position = prev_state[5:8]
    
    # Reward for moving
    distance_moved = np.linalg.norm(position - prev_position)
    reward += distance_moved * 10  # Increase reward for movement
    
    # Penalize for being too high or too low
    altitude = position[2]
    if altitude < MIN_ALTITUDE:
        reward -= (MIN_ALTITUDE - altitude) * 5
    elif altitude > MAX_ALTITUDE:
        reward -= (altitude - MAX_ALTITUDE) * 5
    
    # Reward for staying within the patrol square
    if (current_square['min_x'] <= position[0] <= current_square['max_x'] and
        current_square['min_y'] <= position[1] <= current_square['max_y']):
        reward += 5
    else:
        reward -= 10
    
    # Reward for forward velocity
    forward_velocity = velocity[0]
    reward += forward_velocity * 2
    
    # Penalize for collisions
    if client.simGetCollisionInfo().has_collided:
        reward -= 100

    if np.any(lidar_data < 5):
        reward -= 10
    
    return reward

@monitoring.track_function
def lift_drone(client, target_height):
    while True:
        drone_state = client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        current_square = generate_patrol_square(drone_position)
        _, _, _, current_height = get_drone_state(client, current_square)
        if current_height >= target_height:
            break
        client.moveByVelocityAsync(0, 0, -1, 1).join() # Note: negative z velocity is upward
        logging.info(f"Current height: {current_height}")

    logging.info(f"Drone lifted to approximately {target_height} meters.")

logging.info("Starting drone training")
@monitoring.track_function
def train_drone(model, client, num_episodes=5000):
    epsilon = EPSILON_START
    best_reward = float('-inf')
    experience_replay = ExperienceReplay(MEMORY_SIZE, BATCH_SIZE)

    # Generate the initial patrol square
    drone_state = client.getMultirotorState()
    drone_position = drone_state.kinematics_estimated.position
    current_square = generate_patrol_square(drone_position)

    for episode in range(num_episodes):
        lift_drone(client, 5)
        client.reset()
        visualize_patrol_square(client, current_square)
        coverage = np.zeros((SQUARE_SIZE // COVERAGE_RESOLUTION, SQUARE_SIZE // COVERAGE_RESOLUTION))
        img_rgb, state, lidar_data, _ = get_drone_state(client, current_square)

        monitoring.state_machine.transition("EPISODE_START")
        monitoring.log_event(f"Starting episode {episode + 1}")

        episode_reward = 0
        step = 0
        start_time = time.time()
        
        while time.time() - start_time < 300:  # 5-minute timeout
            try:
                # Get the current drone state
                drone_state = client.getMultirotorState()
                drone_position = drone_state.kinematics_estimated.position

                # Get the drone state, image, and lidar data
                img_rgb, state, lidar_data, _ = get_drone_state(client, current_square)

                # Choose and perform an action
                if np.random.rand() < epsilon:
                    action = np.random.uniform(-1, 1, size=5)
                else:
                    action = model.predict([np.expand_dims(img_rgb, axis=0), np.expand_dims(state, axis=0), np.expand_dims(lidar_data, axis=0)])[0]

                vx, vy, vz, yaw_rate, throttle = interpret_action(action)
                client.moveByVelocityAsync(vx, vy, vz, 1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)).join()

                # Get the new state and calculate reward
                new_img_rgb, new_state, new_lidar_data, _ = get_drone_state(client, current_square)
                reward = calculate_reward(new_state, state, current_square, new_lidar_data)

                # Store the experience
                experience = (img_rgb, state, lidar_data, action, reward, new_img_rgb, new_state, new_lidar_data)
                experience_replay.store(experience)

                episode_reward += reward
                step += 1
                
                if step % 10 == 0:
                    logging.info(f"Episode {episode}, Step {step}, Reward {episode_reward}")
                
                state = new_state
                img_rgb = new_img_rgb
                lidar_data = new_lidar_data
                
                if client.simGetCollisionInfo().has_collided:
                    logging.warning("Collision detected. Ending episode.")
                    break
                
            except Exception as e:
                logging.error(f"Error during neural network control: {e}")
                break
        
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_save_path = "best_model.keras"
            model.save(model_save_path)
            logging.info(f"Saved best model with reward: {best_reward}")
        
        # Train the model using the experience replay buffer
        if len(experience_replay) >= BATCH_SIZE:
            batch = experience_replay.sample()
            train_model(model, batch)

        logging.info(f"Episode {episode + 1}/{num_episodes} completed. Total reward: {episode_reward}")
        
        if episode % 100 == 0:
            model.save(f"drone_model_episode_{episode}.keras")
    
    client.armDisarm(False)
    client.enableApiControl(False)
    logging.info("Training complete.")

def train_model(model, batch):
    img_rgb, states, lidar_data, actions, rewards, next_img_rgb, next_states, next_lidar_data = batch
    
    # Convert lists to numpy arrays
    img_rgb = np.array(img_rgb)
    states = np.array(states)
    lidar_data = np.array(lidar_data)
    next_img_rgb = np.array(next_img_rgb)
    next_states = np.array(next_states)
    next_lidar_data = np.array(next_lidar_data)
    
    # Predict Q-values for current and next states
    current_q_values = model.predict([img_rgb, states, lidar_data])
    next_q_values = model.predict([next_img_rgb, next_states, next_lidar_data])
    
    # Compute target Q-values
    targets = current_q_values.copy()
    for i in range(len(rewards)):
        targets[i][actions[i]] = rewards[i] + GAMMA * np.max(next_q_values[i])
    
    # Train the model
    model.fit([img_rgb, states, lidar_data], targets, epochs=1, verbose=0)

MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99

memory = deque(maxlen=MEMORY_SIZE)


if __name__ == "__main__":
    if os.path.exists("best_model.keras"):
        logging.info("Attempting to load model")
        model = load_model("best_model.keras")
        logging.info("Model loaded successfully")
    else:
        model = create_model()
        logging.info("Created new model.")

    logging.info(f"Model summary:\n{model.summary()}")

    train_drone(model, client, num_episodes=5000)
