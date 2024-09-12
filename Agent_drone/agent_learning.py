import airsim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import time
import random

class DroneNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DroneNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))

def get_drone_state(client):
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    velocity = state.kinematics_estimated.linear_velocity
    image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    image = np.frombuffer(image.image_data_uint8, dtype=np.uint8).reshape(image.height, image.width, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (84, 84)) / 255.0
    state_vector = np.concatenate([
        [position.x_val, position.y_val, position.z_val],
        [velocity.x_val, velocity.y_val, velocity.z_val],
        image.flatten()
    ])
    return state_vector, position, velocity

def get_reward(prev_position, position, velocity, target_altitude):
    distance = np.sqrt((position.x_val - prev_position.x_val) ** 2 +
                       (position.y_val - prev_position.y_val) ** 2 +
                       (position.z_val - prev_position.z_val) ** 2)
    
    reward = 0
    
    # Reward for maintaining target altitude
    altitude_difference = abs(position.z_val - target_altitude)
    if altitude_difference < 0.5:
        reward += 100
    elif position.z_val < 1:
        reward -= 200
    else:
        reward -= altitude_difference * 20

    # Reward for stability (minimal horizontal movement during pre-training)
    horizontal_speed = np.sqrt(velocity.x_val**2 + velocity.y_val**2)
    reward -= horizontal_speed * 10

    # Penalty for excessive vertical speed
    reward -= abs(velocity.z_val) * 5

    return reward

def safe_takeoff(client, target_altitude):
    client.takeoffAsync().join()
    client.moveToZAsync(target_altitude, 1).join()

def pre_train_stability(net, client, optimizer, criterion, episodes=100):
    print("Starting pre-training for stability...")
    target_altitude = 5.0
    for episode in range(episodes):
        safe_takeoff(client, target_altitude)
        state, prev_position, _ = get_drone_state(client)
        total_reward = 0

        start_time = time.time()
        while time.time() - start_time < 30:  # 30 seconds per pre-training episode
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action = net(state_tensor).numpy().flatten()

            # During pre-training, we only allow small vertical adjustments
            action[0] = 0  # No horizontal movement
            action[1] = 0
            action[2] = np.clip(action[2], -0.1, 0.1)  # Limited vertical movement
            action[3] = 0  # No yaw

            vx, vy, vz, yaw_rate = action
            duration = 0.1

            client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration, 
                                       yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)))
            time.sleep(duration)

            new_state, new_position, new_velocity = get_drone_state(client)
            reward = get_reward(prev_position, new_position, new_velocity, target_altitude)
            total_reward += reward

            new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)
            q_values = net(state_tensor)
            next_q_values = net(new_state_tensor)

            target = q_values.clone()
            for i in range(4):
                target[0, i] = reward + 0.99 * next_q_values[0, i]

            loss = criterion(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = new_state
            prev_position = new_position

        print(f"Pre-training Episode {episode + 1}, Total Reward: {total_reward:.2f}")

def train_drone_nn(model_path='best_model.pth'):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    net = DroneNN(7062, 4)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("Loaded the best model from disk.")
    else:
        pre_train_stability(net, client, optimizer, criterion)

    episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.01
    target_altitude = 5.0

    best_reward = -float('inf')

    for episode in range(episodes):
        client.reset()
        safe_takeoff(client, target_altitude)
        state, prev_position, _ = get_drone_state(client)
        total_reward = 0

        start_time = time.time()
        step = 0
        while time.time() - start_time < 120:  # Run each episode for 120 seconds
            step += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if np.random.rand() < epsilon:
                action = np.random.uniform(-1, 1, 4)
            else:
                with torch.no_grad():
                    action = net(state_tensor).numpy().flatten()

            # Apply the action
            vx, vy, vz, yaw_rate = action
            duration = 0.1  # Apply the action for 0.1 seconds

            client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration, 
                                       yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)))
            time.sleep(duration)

            new_state, new_position, new_velocity = get_drone_state(client)
            reward = get_reward(prev_position, new_position, new_velocity, target_altitude)
            total_reward += reward

            new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)
            q_values = net(state_tensor)
            next_q_values = net(new_state_tensor)

            target = q_values.clone()
            for i in range(4):
                target[0, i] = reward + 0.99 * next_q_values[0, i]

            loss = criterion(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = new_state
            prev_position = new_position

            if step % 10 == 0:
                print(f"Episode {episode+1}, Step {step}, Position: ({new_position.x_val:.2f}, {new_position.y_val:.2f}, {new_position.z_val:.2f}), "
                      f"Velocity: ({new_velocity.x_val:.2f}, {new_velocity.y_val:.2f}, {new_velocity.z_val:.2f}), "
                      f"Action: {action}, Reward: {reward:.2f}")

            if new_position.z_val < 1 or abs(new_position.z_val - target_altitude) > 10:
                print(f"Episode ended early due to unsafe altitude: {new_position.z_val:.2f}")
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1} completed, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(net.state_dict(), model_path)
            print("Saved the best model to disk.")

    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    train_drone_nn()