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
    state_input = keras.Input(shape=(15,))
    # Image processing
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(120, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.3)(x)
    # State processing
    y = keras.layers.Dense(87, activation='relu')(state_input)
    combined = keras.layers.concatenate([x, y])
    # LSTM layers for temporal dependencies
    lstm_out = keras.layers.LSTM(128, return_sequences=True)(keras.layers.Reshape((1, -1))(combined))
    lstm_out = keras.layers.LSTM(56)(lstm_out)
    # Dense layers
    z = keras.layers.Dense(112, activation='relu')(lstm_out)
    z = keras.layers.Dense(40, activation='relu')(z)
    output = keras.layers.Dense(5, activation='tanh')(z)

    model = keras.Model(inputs=[image_input, state_input], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

model = create_model()

def interpret_action(action):
    vx, vy, vz, yaw_rate, throttle = action
    vx *= 5
    vy *= 5
    vz *= 2
    MIN_SAFE_HEIGHT = 2.0
    current_height = client.getMultirotorState().kinematics_estimated.position.z_val
    if current_height <= MIN_SAFE_HEIGHT and vz < 0:
        vz = 0
    yaw_rate *= 45
    throttle = (throttle + 1) / 2
    return vx, vy, vz, yaw_rate, throttle

def get_drone_state():
    try:
        state = client.getMultirotorState()
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = cv2.resize(img_rgb, (84, 84))
        
        # Display the camera image
        # cv2.imshow('Camera Image', img_rgb)
        # cv2.waitKey(1)

        lidar_data = client.getLidarData()
        if len(lidar_data.point_cloud) < 15:  # Ensure enough lidar points
            lidar_points = np.zeros((5,))
        else:
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_points = np.mean(points[:5], axis=0)
            # Ensure lidar_points has 5 elements
            lidar_points = np.pad(lidar_points, (0, 5 - len(lidar_points)), 'constant')
        
        # Display the lidar data
        # print(f"Lidar Data: {lidar_points}")

        position = state.kinematics_estimated.position
        velocity = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        state_vector = np.array([
            position.x_val, position.y_val, position.z_val,
            velocity.x_val, velocity.y_val, velocity.z_val,
            orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val,
            *lidar_points
        ])
        # Truncate or pad the state_vector to ensure it has 15 elements
        state_vector = np.pad(state_vector, (0, 15 - len(state_vector)), 'constant')[:15]
        # print("State vector shape:", state_vector.shape)  # Debugging print
        return img_rgb, state_vector
    except Exception as e:
        print(f"Error getting drone state: {e}")
        return np.zeros((84, 84, 3)), np.zeros((15,))
    
def calculate_reward(state, prev_state, action, last_action, last_action_count, start_time, goal_position):
    position = state[:3]
    prev_position = prev_state[:3]
    velocity = state[3:6]
    height = position[2]
    collision = client.simGetCollisionInfo().has_collided

    # Constants
    MANEUVER_THRESHOLD = 0.2
    INACTIVITY_THRESHOLD = 4
    HEIGHT_LIMIT = 20
    MIN_MOVEMENT_DISTANCE = 0.5
    TIME_PENALTY_FACTOR = 0.01
    MIN_SAFE_HEIGHT = 2.0  # meters

    # Initialize reward
    reward = 0

    # Reward for maintaining minimum height
    if height < MIN_SAFE_HEIGHT:
        reward -= 50 * (MIN_SAFE_HEIGHT - height)
    else:
        reward += (height - MIN_SAFE_HEIGHT) * 10  # Reward for maintaining height above minimum

    # Reward for moving towards the goal
    distance_to_goal = np.linalg.norm(position - goal_position)
    prev_distance_to_goal = np.linalg.norm(prev_position - goal_position)
    reward += (prev_distance_to_goal - distance_to_goal) * 10

    # Reward for completing a loop maneuver
    if np.all(np.abs(velocity) > 0.5) and np.sign(velocity[0]) != np.sign(velocity[1]):
        reward += 5  # Reward for complex maneuver

    # Gradual reward for continuous maneuvering
    maneuver_reward = np.sum(np.abs(action) > MANEUVER_THRESHOLD) * 0.5
    reward += maneuver_reward

    # Penalize for height limits (gradual penalty)
    if height > HEIGHT_LIMIT:
        reward -= (height - HEIGHT_LIMIT) * 0.5

    # Reward for maintaining a consistent height (gradual reward)
    height_stability_reward = 1 / (1 + abs(height - prev_state[2]))
    reward += height_stability_reward

    # Gradual reward for moving a significant distance
    distance_moved = np.linalg.norm(position - prev_position)
    movement_reward = min(distance_moved / MIN_MOVEMENT_DISTANCE, 1) * 2
    reward += movement_reward

    # Penalize for collisions
    if collision:
        reward -= 20

    # Penalize for performing the same action repeatedly (gradual penalty)
    if np.all(action == last_action):
        last_action_count += 1
        if last_action_count > INACTIVITY_THRESHOLD:
            reward -= last_action_count * 0.1
    else:
        last_action_count = 0

    # Time penalty to encourage efficiency
    time_penalty = (time.time() - start_time) * TIME_PENALTY_FACTOR
    reward -= time_penalty

    return reward, distance_moved, collision, last_action_count

def train_drone(model, num_episodes=10000):
    _, prev_state = get_drone_state()
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.01
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        episode_reward = 0
        episode_distance = 0
        last_action = np.zeros(5)
        last_action_count = 0
        
        goal_position = np.random.uniform(-50, 50, 3)
        
        while time.time() - episode_start_time < 300:
            img_rgb, current_state = get_drone_state()
            current_height = current_state[2]
            MIN_SAFE_HEIGHT = 2.0
            if current_height < MIN_SAFE_HEIGHT:
                client.moveByVelocityAsync(0, 0, 2, 0.5)
                time.sleep(0.5)
                continue
            
            if np.random.rand() < epsilon:
                action = np.random.uniform(-1, 1, 5)
            else:
                action = model.predict([np.array([img_rgb]), np.array([current_state])])[0]
            
            vx, vy, vz, yaw_rate, throttle = interpret_action(action)
            client.moveByVelocityAsync(vx, vy, vz, 0.1)
            client.rotateByYawRateAsync(yaw_rate, 0.1)
            client.moveByMotorPWMsAsync(throttle, throttle, throttle, throttle, 0.1)
            
            reward, distance, collision, last_action_count = calculate_reward(
                current_state, prev_state, action, last_action, last_action_count, 
                episode_start_time, goal_position
            )
            episode_reward += reward
            episode_distance += distance
            target = reward + action
            history = model.train_on_batch([np.array([img_rgb]), np.array([current_state])], np.array([target]))
            metrics.update(history[0], reward, distance, collision)
            prev_state = current_state
            last_action = action

        print(f"Episode {episode + 1} ended. Total Reward: {episode_reward:.2f}, Total Distance: {episode_distance:.2f}")
        metrics.new_episode()
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_model(model, 'new_best_model.h5')
            print(f"New best model saved with reward: {best_reward:.2f}")
        
        if (episode + 1) % 100 == 0:
            save_model(model, f'model_episode_{episode+1}.h5')
            print(f"Model saved at episode {episode+1}")
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("Training completed.")



def save_best_model(model):
    try:
        save_model(model,  'new_best_model.h5')
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_best_model():
    if os.path.exists('new_best_model.h5'):
        try:
            print("Load model acces")
            return load_model('new_best_model.h5')
        except Exception as e:
            print(f"Error loading model: {e}")
            return create_model()
    else:
        print("Model file not found. Creating a new model.")
        model = create_model()
        save_best_model(model)
        return model

# Train drone
model = load_best_model()  # Load or create the best model
train_drone(model)

