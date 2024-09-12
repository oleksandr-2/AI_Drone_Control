import logging 
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sqlite3
import threading
import time
import inspect
import functools

# Enhanced logging 
def setup_logger():
    logger = logging.getLogger('drone_training')
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler('drone_training.log', maxBytes=10*1024*1024, backupCount=5)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# Function call Traker
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering function: {func.__name__}")
        logger.debug(f"Arguments: {args}, Keyword arguments: {kwargs}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting function: {func.__name__}")
        logger.debug(f"Return value: {result}")
        return result
    return wrapper

# Real-time Data Visualization
class Visualizer:
    def __init__(self):
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 20))
        self.rewards = []
        self.distances = []
        self.altitudes = []
        self.velocities = []

    def update(self, frame):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax1.plot(self.rewards)
        self.ax2.plot(self.distances)
        self.ax3.plot(self.altitudes)
        self.ax4.plot(self.velocities)
        self.ax1.set_title('Rewards')
        self.ax2.set_title('Distances')
        self.ax3.set_title('Altitudes')
        self.ax4.set_title('Velocities')

    def start(self):
        self.ani = FuncAnimation(self.fig, self.update, interval= 1000)
        plt.show()

    def update_data(self, reward, distance, altitude, velocity):
        self.rewards.append(reward)
        self.distances.append(distance)
        self.altitudes.append(altitude)
        self.velocities.append(velocity)

visualizer = Visualizer()

# Telemetry Database
class TelemetryDB:
    def __init__(self, db_name='telemetry.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS drone_state
            (timestamp REAL, x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, yaw REAL, pitch REAL, roll REAL)
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS rewards
            (timestamp REAL, reward REAL, distance REAL, altitude REAL, velocity REAL)
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events
            (timestamp REAL, event TEXT, details TEXT)
        ''')
        self.conn.commit()

    def log_state(self, timestamp, x, y, z, vx, vy, vz, yaw, pitch, roll):
        self.cursor.execute('INSERT INTO drone_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                            (timestamp, x, y, z, vx, vy, vz, yaw, pitch, roll))
        self.conn.commit()

    def log_reward(self, timestamp, reward, distance, altitude, velocity):
        self.cursor.execute('INSERT INTO rewards VALUES (?, ?, ?, ?, ?)',
                            (timestamp, reward, distance, altitude, velocity))
        self.conn.commit()

    def log_event(self, timestamp, event, details):
        self.cursor.execute('INSERT INTO events VALUES (?, ?, ?)',
                            (timestamp, event, details))
        self.conn.commit()

telemetry_db = TelemetryDB()

# Watchdog Timer 
class Watchdog:
    def __init__(self, timeout = 60):
        self.timeout = timeout
        self.timer = None
        self.last_reset = time.time()

    def reset(self):
        self.last_reset = time.time()

    def start(self):
        def check():
            while True:
                if time.time() - self.last_reset > self.timeout:
                    logger.error("Watchdog timer expired. Program may be frozen")
                    telemetry_db.log_event(time.time(), "WATCHDOG_EXPIRED", "Program may be frozen")
                    # Add code here to dump the current state or take other actions
                time.sleep(1)
        self.timer = threading.Thread(target= check)
        self.timer.daemon = True
        self.timer.start()

watchdog = Watchdog()

# State Machine Tracker
class StateMachine:
    def __init__(self):
        self.state = "INITIALIZING"

    def transition(self, new_state):
        logger.info(f"State transition: {self.state} -> {new_state}")
        telemetry_db.log_event(time.time(), "STATE_TRANSITION", f"{self.state} -> {new_state}")
        self.state = new_state

state_machine = StateMachine()

# Data Transfer Tracker
class DataTransferTracker:
    @staticmethod
    def log_data_transfer(source, destination, data):
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame)
        logger.debug(f"Data transfer from {source} to {destination} at {caller_info.filename}:{caller_info.lineno}")
        logger.debug(f"Data: {data}")
        telemetry_db.log_event(time.time(), "DATA_TRANSFER", f"From {source} to {destination}")

data_tracker = DataTransferTracker()

# Function to start all monitoring tools
def start_monitoring():
    watchdog.start()
    viz_thread = threading.Thread(target=visualizer.start)
    viz_thread.daemon = True
    viz_thread.start()

# Function to update monitoring tools with new data
def update_monitoring(state, reward, distance_moved):
    timestamp = time.time()
    x, y, z, vx, vy, vz = state[:6]
    velocity = (vx**2 + vy**2 + vz**2)**0.5
    telemetry_db.log_state(timestamp, x, y, z, vx, vy, vz, 0, 0, 0)  # Assuming yaw, pitch, roll are not available
    telemetry_db.log_reward(timestamp, reward, distance_moved, z, velocity)
    visualizer.update_data(reward, distance_moved, z, velocity)
    watchdog.reset()

# Function to log important events
def log_event(event, level='info', details=None):
    if level == 'info':
        logger.info(event)
    elif level == 'warning':
        logger.warning(event)
    elif level == 'error':
        logger.error(event)
    else:
        logger.debug(event)
    telemetry_db.log_event(time.time(), event, details)

# Decorator for tracking function calls
def track_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering function: {func.__name__}")
        logger.debug(f"Arguments: {args}, Keyword arguments: {kwargs}")
        telemetry_db.log_event(time.time(), "FUNCTION_CALL", f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting function: {func.__name__}")
        logger.debug(f"Return value: {result}")
        telemetry_db.log_event(time.time(), "FUNCTION_RETURN", f"Exiting {func.__name__}")
        return result
    return wrapper

