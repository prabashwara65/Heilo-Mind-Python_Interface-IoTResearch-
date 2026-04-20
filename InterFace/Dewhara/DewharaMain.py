#!/usr/bin/env python3
"""
Battery Prediction Main Program
Handles Arduino serial data, prediction requests, and MQTT responses for Dewhara
Stores prediction results in DynamoDB
"""

import os
import json
import time
import threading
import queue
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import boto3
import ssl
import signal
import sys
import psutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
from decimal import Decimal
import serial
import serial.tools.list_ports
from collections import deque

# AWS IoT Core SDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Paho MQTT for client publishing
import paho.mqtt.client as mqtt

# Suppress warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =============================
# CONFIGURATION
# =============================
class Config:
    """Central configuration class"""
    
    # AWS IoT Core Settings (Server/Listener)
    SERVER_CLIENT_ID = "battery_predictor_server"
    AWS_IOT_ENDPOINT = "ajmja1mzmi1j4-ats.iot.eu-north-1.amazonaws.com"
    REQUEST_TOPIC = "battery/prediction/request"
    RESPONSE_TOPIC = "battery/prediction/response"
    
    # Client Publisher Settings
    CLIENT_ID = "dewhara_client"
    CLIENT_PUBLISH_TOPIC = "battery/prediction/result"
    
    # DynamoDB
    TABLE_NAME = "DewharaBatteryPredictionResults"
    REGION = "eu-north-1"
    ENABLE_DYNAMODB = True
    
    # Paths
    ROOT_DIR = Path(__file__).resolve().parent
    CERTS_DIR = ROOT_DIR / "certs"
    MODEL_DIR = ROOT_DIR / "models"
    DATA_DIR = ROOT_DIR / "data"
    RESULTS_DIR = ROOT_DIR / "results"
    
    # Certificate paths (Server)
    SERVER_ROOT_CA = CERTS_DIR / "AmazonRootCA1.pem"
    SERVER_CERT = CERTS_DIR / "certificate.pem.crt"
    SERVER_PRIVATE_KEY = CERTS_DIR / "private.pem.key"
    
    # Certificate paths (Client)
    CLIENT_CERT = CERTS_DIR / "6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-certificate.pem.crt"
    CLIENT_KEY = CERTS_DIR / "6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-private.pem.key"
    CLIENT_ROOT_CA = CERTS_DIR / "AmazonRootCA1.pem"
    
    # Model paths
    KERAS_MODEL_PATH = MODEL_DIR / "battery_soc_model.keras"
    SCALER_PATH = MODEL_DIR / "scaler_led.pkl"
    
    # Battery data paths
    BATTERY_SOC_PATH = DATA_DIR / "battery SOC.csv"
    COMBINED_DATA_PATH = DATA_DIR / "combined.csv"
    
    # Performance monitoring
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_LOG_INTERVAL = 3600
    
    # Threading
    NUM_WORKER_THREADS = 4
    REQUEST_QUEUE_SIZE = 100
    
    # Prediction settings
    NUM_WARMUP_PREDICTIONS = 20
    
    # Arduino Serial Settings
    ARDUINO_BAUD_RATE = 9600
    ARDUINO_DATA_POINTS = 20  # TIME_STEPS
    ARDUINO_DATA_COLS = 3      # [voltage, current, time]
    SERIAL_TIMEOUT = 1

# =============================
# LOGGING SETUP
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('battery_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================
# DYNAMODB HELPER
# =============================
def convert_floats_to_decimal(obj):
    """
    Recursively convert floats to Decimal for DynamoDB compatibility
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_floats_to_decimal(item) for item in obj]
    elif isinstance(obj, np.floating):
        return Decimal(str(float(obj)))
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

# =============================
# BATTERY OPTIMIZER
# =============================
class BatteryOptimizer:
    """Battery optimization logic"""
    
    def decide_action(self, soc: float, soh: float, runtime_sec: float) -> str:
        """Decide action based on battery state"""
        if soc < 15:
            return "FORCE_SHED_LOAD"
        elif soc < 30:
            return "SHED_NON_CRITICAL_LOAD"
        elif soh < 70:
            return "LIMIT_CHARGE_RATE"
        elif runtime_sec < 3600:
            return "DELAY_NON_ESSENTIAL_TASKS"
        else:
            return "NORMAL_OPERATION"

# =============================
# DEVICE PRIORITY MANAGER
# =============================
class DevicePriorityManager:
    """Manages device priorities and shedding decisions"""
    
    def __init__(self):
        # Lower number = more critical
        # load_weight: percentage of total current consumed by this device
        self.devices = [
            {"name": "SmokeDetector", "priority": 1, "load_weight": 0.10, "pin": 4},  # 10% load
            {"name": "LEDs",          "priority": 2, "load_weight": 0.30, "pin": 17}, # 30% load
            {"name": "Fan",           "priority": 3, "load_weight": 0.20, "pin": 27}, # 20% load
            {"name": "Heater",        "priority": 4, "load_weight": 0.40, "pin": 22}  # 40% load
        ]
        
        # LED current mapping
        self.LED_CURRENT_MAP = {1: 0.02, 2: 0.04, 3: 0.06}
        
    def get_shed_devices(self, soc: float) -> List[str]:
        """
        Returns a list of device names that should be SHED (turned OFF) based on SOC.
        Tiered approach:
        - SOC < 50%: Shed Priority 4 (Heater)
        - SOC < 40%: Shed Priority 3 (Fan)
        - SOC < 30%: Shed Priority 2 (LEDs)
        - SOC < 20%: Keep only Priority 1 (Smoke Detector)
        """
        shed_list = []
        # Sort devices by priority (low priority first for shedding)
        devices_sorted = sorted(self.devices, key=lambda d: d["priority"], reverse=True)
        
        for device in devices_sorted:
            if device["priority"] == 1:
                continue  # Never shed critical
                
            if device["priority"] == 4 and soc < 50:
                shed_list.append(device["name"])
            elif device["priority"] == 3 and soc < 40:
                shed_list.append(device["name"])
            elif device["priority"] == 2 and soc < 30:
                shed_list.append(device["name"])
            elif soc < 20:  # Critical low
                shed_list.append(device["name"])
                
        return list(set(shed_list))  # unique list
    
    def calculate_load_factor(self, shed_devices: List[str]) -> float:
        """Calculate load reduction factor based on shed devices"""
        factor = 1.0
        for device in self.devices:
            if device["name"] in shed_devices:
                factor -= device["load_weight"]
        return max(factor, 0.1)  # Minimum 10% load for critical only
    
    def get_led_current(self, led_count: int) -> float:
        """Get current draw for given LED count"""
        return self.LED_CURRENT_MAP.get(led_count, 0.02)

# =============================
# ARDUINO DATA COLLECTOR
# =============================

class ArduinoDataCollector:
    """Handles reading and buffering data from Arduino via serial - Works on any port"""
    
    def __init__(self):
        self.serial_connection = None
        self.data_buffer = deque(maxlen=Config.ARDUINO_DATA_POINTS)
        self.voltage_buffer = deque(maxlen=Config.ARDUINO_DATA_POINTS)
        self.current_buffer = deque(maxlen=Config.ARDUINO_DATA_POINTS)
        self.time_buffer = deque(maxlen=Config.ARDUINO_DATA_POINTS)
        self.port = self._find_arduino_port()
        self.running = False
        self.collection_thread = None
        self.led_count = 1
        self.log_counter = 0
        
    def _find_arduino_port(self):
        """Automatically find Arduino port on any system (Windows/Linux/Raspberry Pi)"""
        try:
            # Method 1: Use pySerial's comports (most reliable)
            ports = list(serial.tools.list_ports.comports())
            
            if ports:
                logger.debug(f"Found {len(ports)} serial port(s)")
            
            for port in ports:
                port_name = port.device
                port_desc = port.description.lower()
                port_hwid = port.hwid.lower()
                
                # Check for Arduino or common USB-serial adapters
                keywords = ['arduino', 'usb', 'serial', 'tty', 'acm', 
                           'ch340', 'ch341', 'cp210', 'ftdi', 'pl2303']
                
                if any(keyword in port_desc or keyword in port_hwid for keyword in keywords):
                    logger.info(f"✅ Found Arduino on port: {port_name}")
                    return port_name
            
            # Method 2: Check common ports for different OS
            common_ports = [
                # Linux / Raspberry Pi
                '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',
                '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3',
                '/dev/ttyS0', '/dev/ttyAMA0', '/dev/serial0', '/dev/serial1',
                # Windows
                'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8',
                # Mac
                '/dev/cu.usbserial', '/dev/cu.usbmodem'
            ]
            
            for port in common_ports:
                if os.path.exists(port):
                    # Test if it's actually an Arduino by trying to open it
                    try:
                        test_serial = serial.Serial(port, 9600, timeout=0.5)
                        test_serial.close()
                        logger.info(f"✅ Found active serial port: {port}")
                        return port
                    except:
                        continue
            
            # Method 3: Use system commands to find USB serial devices (Linux/Raspberry Pi)
            try:
                import subprocess
                result = subprocess.run(['lsusb'], capture_output=True, text=True)
                usb_devices = result.stdout.lower()
                
                if any(keyword in usb_devices for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
                    logger.info("🔍 USB serial device detected. Scanning for port...")
                    
                    # Try to get port from dmesg
                    dmesg = subprocess.run(['dmesg', '|', 'grep', '-i', 'tty'], 
                                          shell=True, capture_output=True, text=True)
                    for line in dmesg.stdout.split('\n'):
                        if 'ttyUSB' in line or 'ttyACM' in line:
                            import re
                            match = re.search(r'(ttyUSB\d+|ttyACM\d+)', line)
                            if match:
                                port = f"/dev/{match.group(1)}"
                                if os.path.exists(port):
                                    logger.info(f"✅ Found port from dmesg: {port}")
                                    return port
            except:
                pass
            
            # No Arduino found
            logger.warning("⚠️ No Arduino found. Using simulation mode.")
            logger.info("💡 Tips for connecting Arduino:")
            logger.info("   1. Check USB connection: lsusb (Linux) or Device Manager (Windows)")
            logger.info("   2. Check serial ports: ls -l /dev/tty* (Linux)")
            logger.info("   3. Set permissions: sudo chmod 666 /dev/ttyUSB0 (Linux)")
            logger.info("   4. Add user to dialout: sudo usermod -a -G dialout $USER (Linux)")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Error finding Arduino port: {e}")
            return None
    
    def set_led_count(self, count: int):
        """Set the number of LEDs ON"""
        if count in [1, 2, 3]:
            self.led_count = count
            logger.info(f"✅ LED count set to: {count}")
        else:
            logger.warning(f"⚠️ Invalid LED count: {count}, using default 1")
    
    def connect(self):
        """Connect to Arduino"""
        if not self.port:
            logger.warning("⚠️ No Arduino port available. Running in simulation mode.")
            return False
        
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=Config.ARDUINO_BAUD_RATE,
                timeout=Config.SERIAL_TIMEOUT
            )
            time.sleep(2)
            self.serial_connection.reset_input_buffer()
            logger.info(f"✅ Connected to Arduino on {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Arduino: {e}")
            return False
    
    def read_arduino_data(self):
        """Read Arduino data - expects format: voltage,current,time"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return None

        try:
            if self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if not line:
                    return None

                # Skip lines that don't contain exactly three comma-separated values
                parts = line.split(',')
                if len(parts) != 3:
                    # Only log malformed lines occasionally
                    self.log_counter += 1
                    if self.log_counter % 100 == 0:
                        logger.debug(f"Skipping malformed line: {line[:100]}")
                    return None

                # Attempt conversion
                try:
                    voltage = float(parts[0])
                    current = float(parts[1])
                    time_val = float(parts[2])
                    
                    # Validate ranges
                    if 0 <= voltage <= 15 and 0 <= current <= 5:
                        return [voltage, current, time_val]
                    else:
                        if self.log_counter % 100 == 0:
                            logger.debug(f"Values out of range: V={voltage}, I={current}")
                        return None
                        
                except ValueError:
                    if self.log_counter % 100 == 0:
                        logger.debug(f"Skipping non-numeric line: {line[:100]}")
                    return None

        except Exception as e:
            logger.error(f"Error reading Arduino: {e}")

        return None
    
    def _collection_loop(self):
        """Background thread to continuously collect data from Arduino - SILENT"""
        time_counter = 0
        
        while self.running:
            data_point = self.read_arduino_data()
            
            if data_point:
                voltage, current, time_val = data_point
                self.voltage_buffer.append(voltage)
                self.current_buffer.append(current)
                self.time_buffer.append(time_val)
                self.data_buffer.append([voltage, current, time_val])
            else:
                # Generate simulated data
                time_counter += 1
                simulated_data = self._generate_simulated_data_point(time_counter)
                voltage, current, time_val = simulated_data
                
                self.voltage_buffer.append(voltage)
                self.current_buffer.append(current)
                self.time_buffer.append(time_val)
                self.data_buffer.append(simulated_data)
            
            time.sleep(1)
    
    def _generate_simulated_data_point(self, time_counter: int) -> List[float]:
        """Generate simulated battery data point"""
        base_voltage = 12.5 - (time_counter * 0.001)
        voltage = max(10.5, base_voltage + np.random.normal(0, 0.1))
        
        device_manager = DevicePriorityManager()
        current = device_manager.get_led_current(self.led_count) + np.random.normal(0, 0.001)
        
        return [float(voltage), float(current), float(time_counter)]
    
    def start_collection(self):
        """Start background data collection"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("📊 Arduino data collection started")
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            logger.info("✅ Disconnected from Arduino")
    
    def get_current_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current data buffers as numpy arrays"""
        voltage_array = np.array(list(self.voltage_buffer), dtype=np.float32)
        current_array = np.array(list(self.current_buffer), dtype=np.float32)
        time_array = np.array(list(self.time_buffer), dtype=np.float32)
        
        if len(voltage_array) < Config.ARDUINO_DATA_POINTS:
            padding = Config.ARDUINO_DATA_POINTS - len(voltage_array)
            voltage_array = np.pad(voltage_array, (padding, 0), 'constant', constant_values=12.0)
            current_array = np.pad(current_array, (padding, 0), 'constant', constant_values=0.02)
            time_array = np.pad(time_array, (padding, 0), 'constant', constant_values=0)
            if padding > 0 and len(self.voltage_buffer) == 0:
                logger.debug(f"Buffer not full. Padded with {padding} values.")
        
        voltage_array = voltage_array[-Config.ARDUINO_DATA_POINTS:]
        current_array = current_array[-Config.ARDUINO_DATA_POINTS:]
        time_array = time_array[-Config.ARDUINO_DATA_POINTS:]
        
        return voltage_array, current_array, time_array
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information"""
        return {
            "size": len(self.data_buffer),
            "target": Config.ARDUINO_DATA_POINTS,
            "ready": len(self.data_buffer) >= Config.ARDUINO_DATA_POINTS,
            "source": "arduino" if self.serial_connection else "simulated",
            "led_count": self.led_count,
            "port": self.port if self.port else "none"
        }

# =============================
# MODEL MANAGER
# =============================
class ModelManager:
    """Manages battery SOC prediction model"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.scaler = None
        self.device_manager = DevicePriorityManager()
        self.optimizer = BatteryOptimizer()
        self.model_size_mb = 0
        self.load_time = 0
        
        # Performance metrics
        self.inference_times = []
        self.total_predictions = 0
        self.cpu_usage_samples = []
        self.ram_usage_samples = []
        
        # Battery constants
        self.BATTERY_CAPACITY_AH = 1.8  # 1800 mAh
        self.TIME_STEPS = 20
        
        self._initialized = True
    
    def load_models(self):
        """Load all models and scalers"""
        logger.info("=" * 50)
        logger.info("LOADING BATTERY PREDICTION MODEL")
        logger.info("=" * 50)
        
        try:
            # Create directories if they don't exist
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            os.makedirs(Config.RESULTS_DIR, exist_ok=True)
            
            # Check if model files exist
            if not Config.KERAS_MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {Config.KERAS_MODEL_PATH}")
            if not Config.SCALER_PATH.exists():
                raise FileNotFoundError(f"Scaler not found at {Config.SCALER_PATH}")
            
            # Load model with timing
            start_time = time.time()
            
            logger.info(f"Loading Keras model from: {Config.KERAS_MODEL_PATH}")
            self.model = tf.keras.models.load_model(Config.KERAS_MODEL_PATH, compile=False)
            
            logger.info(f"Loading scaler from: {Config.SCALER_PATH}")
            self.scaler = joblib.load(Config.SCALER_PATH)
            
            self.load_time = time.time() - start_time
            self.model_size_mb = os.path.getsize(Config.KERAS_MODEL_PATH) / (1024 * 1024)
            
            logger.info(f"Ã¢Å“â€¦ Model loaded successfully in {self.load_time:.2f} seconds")
            logger.info(f"Ã°Å¸â€œÅ  Model size: {self.model_size_mb:.2f} MB")
            
            # Load and prepare training data for reference
            self._load_training_data()
            
            # Perform warmup predictions
            self._warmup_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Failed to load model: {e}")
            return False
    
    def _load_training_data(self):
        """Load and prepare training data for reference"""
        try:
            if Config.BATTERY_SOC_PATH.exists() and Config.COMBINED_DATA_PATH.exists():
                soc_df = pd.read_csv(Config.BATTERY_SOC_PATH)
                combined_df = pd.read_csv(Config.COMBINED_DATA_PATH)
                
                soc_df.columns = soc_df.columns.str.strip()
                combined_df.columns = combined_df.columns.str.strip()
                
                soc_df = soc_df.rename(columns={"Battery_SoC_%": "SOC"})
                combined_df = combined_df.rename(columns={
                    "Voltage_measured": "Voltage",
                    "Current_load": "Current_load",
                    "Time": "Time"
                })
                
                logger.info("Ã¢Å“â€¦ Training data loaded successfully")
            else:
                logger.warning("Ã¢Å¡Â Ã¯Â¸Â Training data files not found")
        except Exception as e:
            logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â Could not load training data: {e}")
    
    def _warmup_model(self):
        """Perform warmup predictions"""
        logger.info("Ã°Å¸â€Â¥ Warming up model...")
        
        dummy_data = np.random.randn(self.TIME_STEPS, 3).astype(np.float32)
        dummy_scaled = self.scaler.transform(dummy_data)
        dummy_seq = dummy_scaled.reshape(1, self.TIME_STEPS, 3)
        
        for i in range(Config.NUM_WARMUP_PREDICTIONS):
            self.model.predict(dummy_seq, verbose=0)
            if (i + 1) % 5 == 0:
                logger.info(f"  Warmup {i + 1}/{Config.NUM_WARMUP_PREDICTIONS} complete")
        
        logger.info("Ã¢Å“â€¦ Model warmup complete")
    
    def prepare_input(self, voltage: np.ndarray, current: np.ndarray, time_vals: np.ndarray) -> np.ndarray:
        """Prepare input data for prediction"""
        try:
            # Create feature matrix
            features = np.column_stack([voltage, current, time_vals])
            
            # Ensure correct shape
            if len(features.shape) == 2 and features.shape[0] == self.TIME_STEPS:
                features = features.reshape(1, self.TIME_STEPS, 3)
            
            # Scale features
            original_shape = features.shape
            features_reshaped = features.reshape(-1, 3)
            scaled = self.scaler.transform(features_reshaped)
            
            return scaled.reshape(original_shape)
            
        except Exception as e:
            logger.error(f"Error preparing input: {e}")
            raise
    
    def create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for prediction"""
        if len(X) <= self.TIME_STEPS:
            return np.array([])
        
        Xs = []
        for i in range(len(X) - self.TIME_STEPS):
            Xs.append(X[i:i + self.TIME_STEPS])
        return np.array(Xs)
    
    def simulate_battery(self, time_vals: np.ndarray, led_count: int) -> Dict[str, Any]:
        """
        Run battery simulation with baseline and optimized paths
        This uses physics-based simulation without relying on model predictions
        """
        BATTERY_CAPACITY_AH = 1.8
        selected_current = self.device_manager.get_led_current(led_count)
        
        # Initialize tracking arrays
        soc_baseline_list = []
        soc_optimized_list = []
        soh_list = []
        runtime_base_list = []
        runtime_opt_list = []
        decisions = []
        final_decisions = []
        shed_devices_history = []
        
        # Initial state (starting at 25% SOC as per your original code)
        initial_soc_percent = 25.0
        remaining_baseline_ah = (initial_soc_percent / 100) * BATTERY_CAPACITY_AH
        remaining_optimized_ah = remaining_baseline_ah
        initial_capacity = BATTERY_CAPACITY_AH
        
        logger.info(f"Running battery simulation with {led_count} LED(s)...")
        logger.info(f"Initial SOC: {initial_soc_percent}%, Current draw: {selected_current*1000:.2f} mA")
        
        # Run simulation for each time step
        for i in range(len(time_vals) - 1):
            delta_t = max(time_vals[i + 1] - time_vals[i], 1)
            
            # 1. Baseline Simulation (Constant load)
            discharge_baseline = selected_current * (delta_t / 3600)
            remaining_baseline_ah = max(remaining_baseline_ah - discharge_baseline, 0)
            soc_baseline_val = (remaining_baseline_ah / BATTERY_CAPACITY_AH) * 100
            soc_baseline_list.append(soc_baseline_val)
            curr_runtime_base = (remaining_baseline_ah / selected_current) * 3600 if selected_current > 0 else 0
            runtime_base_list.append(curr_runtime_base)
            
            # 2. Optimized Simulation (with priority shedding)
            curr_soc_opt = (remaining_optimized_ah / BATTERY_CAPACITY_AH) * 100
            
            # Get devices to shed based on current SOC
            devices_off = self.device_manager.get_shed_devices(curr_soc_opt)
            shed_devices_history.append(devices_off)
            
            # Calculate load reduction factor
            factor = self.device_manager.calculate_load_factor(devices_off)
            
            # Apply optimized discharge
            discharge_optimized = (selected_current * factor) * (delta_t / 3600)
            remaining_optimized_ah = max(remaining_optimized_ah - discharge_optimized, 0)
            soc_opt_val = (remaining_optimized_ah / BATTERY_CAPACITY_AH) * 100
            soc_optimized_list.append(soc_opt_val)
            
            # Calculate SoH (State of Health)
            curr_soh = (remaining_optimized_ah / initial_capacity) * 100
            soh_list.append(curr_soh)
            
            # Calculate runtime
            curr_runtime_opt = (remaining_optimized_ah / (selected_current * factor)) * 3600 if selected_current > 0 and factor > 0 else 0
            runtime_opt_list.append(curr_runtime_opt)
            
            # Get optimization decision
            action = self.optimizer.decide_action(curr_soc_opt, curr_soh, curr_runtime_opt)
            
            # Safety validator
            if curr_soc_opt < 15 and action == "NORMAL_OPERATION":
                safe_action = "FORCE_SHED_LOAD"
            elif curr_soh < 70 and action == "NORMAL_OPERATION":
                safe_action = "LIMIT_CHARGE_RATE"
            elif curr_runtime_opt < 3600 and action == "NORMAL_OPERATION":
                safe_action = "DELAY_NON_ESSENTIAL_TASKS"
            else:
                safe_action = action
            
            final_text = f"Priority Shedding: {', '.join(devices_off)}" if devices_off else "NORMAL_OPERATION"
            
            decisions.append(safe_action)
            final_decisions.append(final_text)
            
            # Stop if battery is empty
            if remaining_baseline_ah <= 0.001 and remaining_optimized_ah <= 0.001:
                break
        
        # Ensure we have at least one value
        if len(soc_baseline_list) == 0:
            soc_baseline_list = [initial_soc_percent]
            soc_optimized_list = [initial_soc_percent]
            soh_list = [100.0]
            runtime_base_list = [24 * 3600]  # 24 hours
            runtime_opt_list = [24 * 3600]
            decisions = ["NORMAL_OPERATION"]
            final_decisions = ["NORMAL_OPERATION"]
            shed_devices_history = [[]]
        
        return {
            "soc_baseline": np.array(soc_baseline_list),
            "soc_optimized": np.array(soc_optimized_list),
            "soh": np.array(soh_list),
            "runtime_base": np.array(runtime_base_list),
            "runtime_opt": np.array(runtime_opt_list),
            "decisions": decisions,
            "final_decisions": final_decisions,
            "shed_devices": shed_devices_history
        }
    
    def predict(self, voltage: np.ndarray, current: np.ndarray, time_vals: np.ndarray, led_count: int) -> Dict[str, Any]:
        """
        Run battery prediction and simulation
        """
        try:
            # Track performance
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)
            
            # Prepare input for prediction
            start_time = time.time()
            scaled_input = self.prepare_input(voltage, current, time_vals)
            
            # Run SOC prediction (this gives us the model's prediction)
            soc_pred_raw = self.model.predict(scaled_input, verbose=0).flatten()
            
            # Use the prediction to influence the simulation
            # The prediction gives us expected SOC values which we can use
            avg_predicted_soc = float(np.mean(soc_pred_raw))
            logger.info(f"Ã°Å¸â€œÅ  Model predicted average SOC: {avg_predicted_soc:.2f}%")
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Store metrics
            self.inference_times.append(inference_time_ms)
            self.total_predictions += 1
            
            memory_after = process.memory_info().rss / (1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Run battery simulation (physics-based)
            simulation_results = self.simulate_battery(time_vals, led_count)
            
            # Get final values
            final_soc_baseline = float(simulation_results["soc_baseline"][-1]) if len(simulation_results["soc_baseline"]) > 0 else 0
            final_soc_optimized = float(simulation_results["soc_optimized"][-1]) if len(simulation_results["soc_optimized"]) > 0 else 0
            final_soh = float(simulation_results["soh"][-1]) if len(simulation_results["soh"]) > 0 else 100
            final_runtime_base = float(simulation_results["runtime_base"][-1]) if len(simulation_results["runtime_base"]) > 0 else 0
            final_runtime_opt = float(simulation_results["runtime_opt"][-1]) if len(simulation_results["runtime_opt"]) > 0 else 0
            
            # Get the last shed devices
            last_shed_devices = simulation_results["shed_devices"][-1] if simulation_results["shed_devices"] else []
            
            result = {
                "led_count": led_count,
                "soc_baseline_percent": final_soc_baseline,
                "soc_optimized_percent": final_soc_optimized,
                "soc_improvement": final_soc_optimized - final_soc_baseline,
                "soh_percent": final_soh,
                "runtime_baseline_hours": final_runtime_base / 3600,
                "runtime_optimized_hours": final_runtime_opt / 3600,
                "optimizer_decision": simulation_results["decisions"][-1] if simulation_results["decisions"] else "UNKNOWN",
                "shed_devices": last_shed_devices,
                "model_predicted_avg_soc": avg_predicted_soc,
                "soc_timeline": {
                    "baseline": simulation_results["soc_baseline"].tolist() if len(simulation_results["soc_baseline"]) > 0 else [],
                    "optimized": simulation_results["soc_optimized"].tolist() if len(simulation_results["soc_optimized"]) > 0 else []
                },
                "performance": {
                    "inference_time_ms": inference_time_ms,
                    "avg_inference_time_ms": float(np.mean(self.inference_times[-100:])) if self.inference_times else inference_time_ms,
                    "ram_usage_mb": memory_after,
                    "cpu_percent": cpu_percent,
                    "total_predictions": self.total_predictions
                }
            }
            
            # Save results to CSV
            self._save_results(simulation_results, led_count)
            
            # Log the final results
            logger.info(f"Ã°Å¸â€œÅ  Final Results - SOC Baseline: {final_soc_baseline:.2f}%, "
                       f"SOC Optimized: {final_soc_optimized:.2f}%, "
                       f"Improvement: {final_soc_optimized - final_soc_baseline:.2f}%, "
                       f"SoH: {final_soh:.2f}%, "
                       f"Shed Devices: {last_shed_devices}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _save_results(self, simulation_results: Dict[str, Any], led_count: int):
        """Save simulation results to CSV"""
        try:
            df_results = pd.DataFrame({
                "TimeStep": np.arange(len(simulation_results["soc_optimized"])),
                "SOC_Baseline": simulation_results["soc_baseline"],
                "SOC_Optimized": simulation_results["soc_optimized"],
                "SoH": simulation_results["soh"],
                "Runtime_Base_sec": simulation_results["runtime_base"],
                "Runtime_Opt_sec": simulation_results["runtime_opt"],
                "OptimizerDecision": simulation_results["decisions"],
                "FinalDecision": simulation_results["final_decisions"]
            })
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Config.RESULTS_DIR / f"battery_simulation_led{led_count}_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            logger.info(f"Ã¢Å“â€¦ Simulation results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Failed to save results: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        process = psutil.Process(os.getpid())
        
        return {
            "model_size_mb": self.model_size_mb,
            "load_time_seconds": self.load_time,
            "ram_usage_mb": process.memory_info().rss / (1024 * 1024),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "thread_count": threading.active_count(),
            "total_predictions": self.total_predictions,
            "avg_inference_time_ms": float(np.mean(self.inference_times[-100:])) if self.inference_times else 0
        }

# =============================
# MQTT SERVER (Request Handler)
# =============================
class BatteryPredictionServer:
    """Handles incoming prediction requests via AWS IoT Core"""
    
    def __init__(self, model_manager: ModelManager, arduino_collector: ArduinoDataCollector):
        self.model_manager = model_manager
        self.arduino_collector = arduino_collector
        self.mqtt_client = None
        self.running = False
        self.request_queue = queue.Queue(maxsize=Config.REQUEST_QUEUE_SIZE)
        self.worker_threads = []
        self.dynamodb = None
        self.table = None
        
        # Initialize DynamoDB
        if Config.ENABLE_DYNAMODB:
            try:
                self.dynamodb = boto3.resource("dynamodb", region_name=Config.REGION)
                self._ensure_table_exists()
            except Exception as e:
                logger.warning(f"Ã¢Å¡Â Ã¯Â¸Â DynamoDB initialization failed: {e}")
                self.dynamodb = None
                self.table = None
        else:
            logger.info("Ã¢â€žÂ¹Ã¯Â¸Â DynamoDB logging is disabled")
    
    def _ensure_table_exists(self):
        """Ensure DynamoDB table exists, create if it doesn't"""
        if not self.dynamodb:
            return False
        
        try:
            # Check if table exists
            existing_tables = self.dynamodb.meta.client.list_tables()['TableNames']
            if Config.TABLE_NAME not in existing_tables:
                logger.info(f"Creating DynamoDB table: {Config.TABLE_NAME}")
                
                table = self.dynamodb.create_table(
                    TableName=Config.TABLE_NAME,
                    KeySchema=[{'AttributeName': 'requestId', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': 'requestId', 'AttributeType': 'S'}],
                    BillingMode='PAY_PER_REQUEST'
                )
                
                # Wait for table to be created
                table.wait_until_exists()
                logger.info(f"Ã¢Å“â€¦ Table {Config.TABLE_NAME} created successfully")
            
            self.table = self.dynamodb.Table(Config.TABLE_NAME)
            logger.info("Ã¢Å“â€¦ DynamoDB initialized and ready")
            return True
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Failed to ensure DynamoDB table exists: {e}")
            return False
    
    def connect(self):
        """Connect to AWS IoT Core"""
        try:
            self.mqtt_client = AWSIoTMQTTClient(Config.SERVER_CLIENT_ID)
            
            self.mqtt_client.configureEndpoint(Config.AWS_IOT_ENDPOINT, 8883)
            self.mqtt_client.configureCredentials(
                str(Config.SERVER_ROOT_CA),
                str(Config.SERVER_PRIVATE_KEY),
                str(Config.SERVER_CERT)
            )
            
            self.mqtt_client.configureOfflinePublishQueueing(-1)
            self.mqtt_client.configureDrainingFrequency(2)
            self.mqtt_client.configureConnectDisconnectTimeout(10)
            self.mqtt_client.configureMQTTOperationTimeout(5)
            self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
            
            logger.info("Ã°Å¸â€œÂ¡ Connecting to AWS IoT Core...")
            self.mqtt_client.connect()
            logger.info("Ã¢Å“â€¦ Connected to AWS IoT Core")
            
            self.mqtt_client.subscribe(Config.REQUEST_TOPIC, 1, self._message_callback)
            logger.info(f"Ã°Å¸â€œÂ¡ Subscribed to topic: {Config.REQUEST_TOPIC}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Failed to connect to AWS IoT Core: {e}")
            return False
    
    def _message_callback(self, client, userdata, message):
        """Callback for incoming MQTT messages"""
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            request_id = payload.get('requestId') or payload.get('request_id')
            logger.info(f"Ã°Å¸â€œÂ© Request received with requestId: {request_id}")
            
            self.request_queue.put({
                'topic': message.topic,
                'payload': payload,
                'timestamp': time.time()
            })
            
        except json.JSONDecodeError as e:
            logger.error(f"Ã¢ÂÅ’ Invalid JSON payload: {e}")
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Error processing message: {e}")
    
    def _process_request(self, request: Dict[str, Any]):
        """Process a single prediction request"""
        try:
            payload = request['payload']
            
            request_id = payload.get('requestId') or payload.get('request_id')
            if not request_id:
                request_id = f"auto_{int(time.time() * 1000)}"
            
            device_id = payload.get('deviceId', 'unknown')
            
            # Get LED count from request
            led_count = payload.get('led_count') or payload.get('ledCount')
            if led_count:
                try:
                    led_count = int(led_count)
                    self.arduino_collector.set_led_count(led_count)
                except:
                    led_count = 1
            else:
                led_count = 1
            
            logger.info(f"Ã°Å¸â€â€ž Processing request {request_id} from device {device_id} (LEDs: {led_count})")
            
            # Get buffer status
            buffer_status = self.arduino_collector.get_buffer_status()
            logger.info(f"Ã°Å¸â€œÅ  Buffer status: {buffer_status['size']}/20 points (Source: {buffer_status['source']})")
            
            # Get current data from Arduino
            voltage, current, time_vals = self.arduino_collector.get_current_data()
            
            # Log sample data
            logger.info(f"Ã°Å¸â€œÅ  Latest readings - Voltage: {voltage[-5:].tolist()}, Current: {current[-5:].tolist()}")
            
            # Run prediction
            result = self.model_manager.predict(voltage, current, time_vals, led_count)
            
            # Prepare response
            response = {
                "requestId": request_id,
                "deviceId": device_id,
                "timestamp": time.time(),
                "status": "success",
                "led_count": led_count,
                "soc_baseline_percent": result["soc_baseline_percent"],
                "soc_optimized_percent": result["soc_optimized_percent"],
                "soc_improvement": result["soc_improvement"],
                "soh_percent": result["soh_percent"],
                "runtime_baseline_hours": result["runtime_baseline_hours"],
                "runtime_optimized_hours": result["runtime_optimized_hours"],
                "optimizer_decision": result["optimizer_decision"],
                "shed_devices": result["shed_devices"],
                "model_predicted_avg_soc": result["model_predicted_avg_soc"],
                "performance": result["performance"],
                "data_source": buffer_status['source'],
                "buffer_size": buffer_status['size'],
                "buffer_ready": buffer_status['ready']
            }
            
            # Publish response
            self.mqtt_client.publish(Config.RESPONSE_TOPIC, json.dumps(response), 1)
            logger.info(f"Ã¢Å“â€¦ Response sent for request {request_id}")
            
            self.mqtt_client.publish(Config.CLIENT_PUBLISH_TOPIC, json.dumps(response), 1)
            logger.info(f"Ã¢Å“â€¦ Result published to {Config.CLIENT_PUBLISH_TOPIC}")
            
            # ============= DYNAMODB LOGGING =============
            # Log to DynamoDB if available
            if self.dynamodb and self.table:
                try:
                    # Add timestamp as string for DynamoDB
                    response['timestamp_str'] = datetime.fromtimestamp(response['timestamp']).isoformat()
                    
                    # Add readable datetime
                    response['datetime'] = datetime.fromtimestamp(response['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add all the detailed battery information
                    dynamodb_item = {
                        'requestId': request_id,
                        'deviceId': device_id,
                        'timestamp': Decimal(str(response['timestamp'])),
                        'timestamp_str': response['timestamp_str'],
                        'datetime': response['datetime'],
                        'led_count': led_count,
                        'soc_baseline_percent': Decimal(str(result['soc_baseline_percent'])),
                        'soc_optimized_percent': Decimal(str(result['soc_optimized_percent'])),
                        'soc_improvement': Decimal(str(result['soc_improvement'])),
                        'soh_percent': Decimal(str(result['soh_percent'])),
                        'runtime_baseline_hours': Decimal(str(result['runtime_baseline_hours'])),
                        'runtime_optimized_hours': Decimal(str(result['runtime_optimized_hours'])),
                        'optimizer_decision': result['optimizer_decision'],
                        'shed_devices': ', '.join(result['shed_devices']) if result['shed_devices'] else 'NONE',
                        'model_predicted_avg_soc': Decimal(str(result['model_predicted_avg_soc'])),
                        'data_source': buffer_status['source'],
                        'buffer_size': buffer_status['size'],
                        'status': 'success'
                    }
                    
                    # Add performance metrics
                    if 'performance' in result:
                        dynamodb_item['inference_time_ms'] = Decimal(str(result['performance']['inference_time_ms']))
                        dynamodb_item['avg_inference_time_ms'] = Decimal(str(result['performance']['avg_inference_time_ms']))
                        dynamodb_item['ram_usage_mb'] = Decimal(str(result['performance']['ram_usage_mb']))
                        dynamodb_item['cpu_percent'] = Decimal(str(result['performance']['cpu_percent']))
                    
                    # Add to DynamoDB
                    self.table.put_item(Item=dynamodb_item)
                    logger.info(f"Ã°Å¸â€™Â¾ Battery prediction logged to DynamoDB for request {request_id}")
                    
                    # Log the saved values for verification
                    logger.info(f"Ã°Å¸â€œÅ  DynamoDB saved values - SOC Baseline: {result['soc_baseline_percent']:.2f}%, "
                              f"SOC Optimized: {result['soc_optimized_percent']:.2f}%, "
                              f"Improvement: {result['soc_improvement']:.2f}%, "
                              f"SoH: {result['soh_percent']:.2f}%, "
                              f"Shed Devices: {', '.join(result['shed_devices']) if result['shed_devices'] else 'NONE'}")
                    
                except Exception as e:
                    logger.error(f"Ã¢ÂÅ’ DynamoDB logging failed: {e}")
                    # Disable DynamoDB if table doesn't exist
                    if "ResourceNotFoundException" in str(e):
                        logger.warning("Ã¢Å¡Â Ã¯Â¸Â DynamoDB table not found - disabling DynamoDB logging")
                        self.dynamodb = None
                        self.table = None
            
            return response  # Return response for testing
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Error processing request: {e}")
            error_request_id = payload.get('requestId') or payload.get('request_id', 'unknown') if 'payload' in locals() else 'unknown'
            error_device_id = payload.get('deviceId', 'unknown') if 'payload' in locals() else 'unknown'
            self._send_error_response(error_request_id, error_device_id, str(e))
    
    def _send_error_response(self, request_id: str, device_id: str, error_message: str):
        """Send error response"""
        try:
            error_response = {
                "requestId": request_id,
                "deviceId": device_id,
                "timestamp": time.time(),
                "status": "error",
                "error": error_message
            }
            
            self.mqtt_client.publish(Config.RESPONSE_TOPIC, json.dumps(error_response), 1)
            self.mqtt_client.publish(Config.CLIENT_PUBLISH_TOPIC, json.dumps(error_response), 1)
            
            logger.info(f"Ã¢Å¡Â Ã¯Â¸Â Error response sent for request {request_id}")
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Failed to send error response: {e}")
    
    def _worker_thread(self, thread_id: int):
        """Worker thread for processing requests"""
        logger.info(f"Ã°Å¸Â§Âµ Worker thread {thread_id} started")
        
        while self.running:
            try:
                request = self.request_queue.get(timeout=1)
                self._process_request(request)
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Worker thread {thread_id} error: {e}")
        
        logger.info(f"Ã°Å¸Â§Âµ Worker thread {thread_id} stopped")
    
    def start(self):
        """Start the server"""
        if not self.connect():
            logger.error("Ã¢ÂÅ’ Failed to start server")
            return False
        
        self.running = True
        
        for i in range(Config.NUM_WORKER_THREADS):
            thread = threading.Thread(
                target=self._worker_thread,
                args=(i,),
                name=f"Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"Ã°Å¸Å¡â‚¬ Server started with {Config.NUM_WORKER_THREADS} worker threads")
        return True
    
    def stop(self):
        """Stop the server"""
        logger.info("Ã°Å¸â€ºâ€˜ Stopping server...")
        self.running = False
        
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
                logger.info("Ã¢Å“â€¦ Disconnected from AWS IoT Core")
            except:
                pass
        
        logger.info("Ã¢Å“â€¦ Server stopped")

# =============================
# MQTT CLIENT (Data Publisher)
# =============================
class BatteryDataPublisher:
    """Publishes battery data to AWS IoT Core"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to AWS IoT Core as a publisher"""
        try:
            self.client = mqtt.Client(
                client_id=Config.CLIENT_ID,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            
            self.client.tls_set(
                ca_certs=str(Config.CLIENT_ROOT_CA),
                certfile=str(Config.CLIENT_CERT),
                keyfile=str(Config.CLIENT_KEY),
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLSv1_2
            )
            
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            
            logger.info("Ã°Å¸â€œÂ¡ Connecting publisher to AWS IoT Core...")
            self.client.connect(Config.AWS_IOT_ENDPOINT, 8883, 60)
            self.client.loop_start()
            
            timeout = 10
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info("Ã¢Å“â€¦ Publisher connected to AWS IoT Core")
                return True
            else:
                logger.error("Ã¢ÂÅ’ Publisher connection timeout")
                return False
            
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Publisher connection failed: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Connection callback"""
        if rc == 0:
            self.connected = True
            logger.info("Ã¢Å“â€¦ Publisher connected successfully")
        else:
            logger.error(f"Ã¢ÂÅ’ Publisher connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Disconnection callback"""
        self.connected = False
        logger.warning("Ã¢Å¡Â Ã¯Â¸Â Publisher disconnected")
    
    def _on_publish(self, client, userdata, mid, rc, properties=None):
        """Publish callback"""
        logger.debug(f"Ã°Å¸â€œÂ¤ Message {mid} published")
    
    def publish_battery_data(self, voltage: float, current: float, time_val: float, device_id: str = "Dewhara") -> bool:
        """Publish battery data to AWS IoT Core"""
        if not self.connected:
            logger.error("Ã¢ÂÅ’ Publisher not connected")
            return False
        
        try:
            payload = {
                "deviceId": device_id,
                "timestamp": time.time(),
                "voltage": voltage,
                "current": current,
                "time": time_val
            }
            
            result = self.client.publish(
                Config.CLIENT_PUBLISH_TOPIC,
                json.dumps(payload),
                qos=1
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Ã°Å¸â€œÂ¤ Battery data published to {Config.CLIENT_PUBLISH_TOPIC}")
                return True
            else:
                logger.error(f"Ã¢ÂÅ’ Publish failed with code: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"Ã¢ÂÅ’ Failed to publish battery data: {e}")
            return False
    
    def disconnect(self):
        """Disconnect publisher"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Ã¢Å“â€¦ Publisher disconnected")

# =============================
# PERFORMANCE MONITOR
# =============================
class PerformanceMonitor:
    """Monitors and logs system performance"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.running = False
        self.monitor_thread = None
        self.metrics_history = []
        
    def start(self):
        """Start performance monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Ã°Å¸â€œÅ  Performance monitor started")
        
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.metrics_history:
            metrics_file = Path(f"battery_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Ã°Å¸â€œÅ  Performance metrics saved to {metrics_file}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self.model_manager.get_performance_metrics()
                metrics['timestamp'] = time.time()
                metrics['timestamp_str'] = datetime.now().isoformat()
                
                self.metrics_history.append(metrics)
                
                logger.info("Ã°Å¸â€œÅ  Performance Metrics:")
                logger.info(f"  RAM Usage: {metrics['ram_usage_mb']:.2f} MB")
                logger.info(f"  CPU Usage: {metrics['cpu_percent']:.1f}%")
                logger.info(f"  Threads: {metrics['thread_count']}")
                logger.info(f"  Avg Inference: {metrics['avg_inference_time_ms']:.2f} ms")
                
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                for _ in range(Config.PERFORMANCE_LOG_INTERVAL):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Ã¢ÂÅ’ Performance monitor error: {e}")
                time.sleep(60)

# =============================
# MAIN APPLICATION
# =============================
class BatteryPredictionApp:
    """Main application coordinating all components"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.arduino_collector = ArduinoDataCollector()
        self.server = None
        self.publisher = None
        self.monitor = None
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("BATTERY PREDICTION SYSTEM - INITIALIZING")
        logger.info("=" * 60)
        
        # Load model
        if not self.model_manager.load_models():
            return False
        
        # Connect to Arduino
        if self.arduino_collector.connect():
            logger.info("Ã¢Å“â€¦ Arduino detected - using real sensor data")
        else:
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â No Arduino detected - running in simulation mode")
        
        # Start Arduino data collection
        self.arduino_collector.start_collection()
        
        # Initialize server with Arduino collector
        self.server = BatteryPredictionServer(self.model_manager, self.arduino_collector)
        
        # Initialize publisher
        self.publisher = BatteryDataPublisher()
        
        # Initialize performance monitor
        if Config.ENABLE_PERFORMANCE_MONITORING:
            self.monitor = PerformanceMonitor(self.model_manager)
        
        logger.info("=" * 60)
        logger.info("Ã¢Å“â€¦ SYSTEM INITIALIZED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
    
    def start(self):
        """Start all components"""
        logger.info("Ã°Å¸Å¡â‚¬ STARTING BATTERY PREDICTION SYSTEM")
        
        # Start server
        if not self.server.start():
            logger.error("Ã¢ÂÅ’ Failed to start server")
            return
        
        # Connect publisher
        if not self.publisher.connect():
            logger.warning("Ã¢Å¡Â Ã¯Â¸Â Publisher not connected - continuing without publishing")
        
        # Start performance monitor
        if self.monitor:
            self.monitor.start()
        
        self.running = True
        
        # Print system info
        self._print_system_info()
        
        # Main loop
        try:
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop all components"""
        logger.info("Ã°Å¸â€ºâ€˜ SHUTTING DOWN BATTERY PREDICTION SYSTEM")
        
        self.running = False
        
        # Stop Arduino collection
        if self.arduino_collector:
            self.arduino_collector.stop_collection()
        
        # Stop server
        if self.server:
            self.server.stop()
        
        # Disconnect publisher
        if self.publisher:
            self.publisher.disconnect()
        
        # Stop monitor
        if self.monitor:
            self.monitor.stop()
        
        logger.info("Ã°Å¸â€˜â€¹ SYSTEM SHUTDOWN COMPLETE")
    
    def _print_system_info(self):
        """Print system information"""
        buffer_status = self.arduino_collector.get_buffer_status()
        
        logger.info("=" * 60)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 60)
        logger.info(f"Ã°Å¸â€œÂ¡ Server listening on: {Config.REQUEST_TOPIC}")
        logger.info(f"Ã°Å¸â€œÂ¤ Publishing results to: {Config.RESPONSE_TOPIC}")
        logger.info(f"Ã°Å¸â€œÂ¤ Also publishing to: {Config.CLIENT_PUBLISH_TOPIC}")
        logger.info(f"Ã°Å¸Â§Âµ Worker threads: {Config.NUM_WORKER_THREADS}")
        logger.info(f"Ã°Å¸â€œÅ  Performance monitoring: {'Enabled' if Config.ENABLE_PERFORMANCE_MONITORING else 'Disabled'}")
        logger.info(f"Ã°Å¸â€™Â¾ DynamoDB logging: {'Enabled' if Config.ENABLE_DYNAMODB else 'Disabled'}")
        logger.info(f"Ã°Å¸â€Å’ Arduino: {buffer_status['source'].upper()}")
        logger.info(f"Ã°Å¸â€œÅ  Data buffer: {buffer_status['size']}/20 points")
        logger.info(f"Ã°Å¸â€™Â¡ Default LED count: {buffer_status['led_count']}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to shutdown")
        logger.info("=" * 60)

# =============================
# UTILITY FUNCTIONS
# =============================
def generate_test_battery_data() -> Dict[str, Any]:
    """Generate test battery data for 20 time steps"""
    time_steps = 20
    voltage = []
    current = []
    time_vals = []
    
    for i in range(time_steps):
        # Simulate battery discharge
        v = 12.5 - (i * 0.05) + np.random.normal(0, 0.1)
        voltage.append(float(max(10.5, v)))
        current.append(float(0.02 + np.random.normal(0, 0.001)))
        time_vals.append(float(i))
    
    return {
        "voltage": voltage,
        "current": current,
        "time": time_vals
    }

def send_test_prediction_request():
    """Send a test prediction request"""
    publisher = BatteryDataPublisher()
    if publisher.connect():
        test_data = generate_test_battery_data()
        request_id = f"test_{int(time.time() * 1000)}"
        
        request = {
            "requestId": request_id,
            "deviceId": "Dewhara_Test",
            "led_count": 2,  # Test with 2 LEDs
            "voltage": test_data["voltage"],
            "current": test_data["current"],
            "time": test_data["time"]
        }
        
        client = mqtt.Client(client_id="test_client")
        client.tls_set(
            ca_certs=str(Config.CLIENT_ROOT_CA),
            certfile=str(Config.CLIENT_CERT),
            keyfile=str(Config.CLIENT_KEY),
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )
        
        client.connect(Config.AWS_IOT_ENDPOINT, 8883)
        client.loop_start()
        
        print(f"\nÃ°Å¸â€œÂ¤ Sending test request with requestId: {request_id}")
        result = client.publish(Config.REQUEST_TOPIC, json.dumps(request), qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"Ã¢Å“â€¦ Test prediction request sent with requestId: {request_id}")
            print(f"Ã¢Å“â€¦ Request sent. Check response on topic: {Config.RESPONSE_TOPIC}")
        else:
            logger.error("Ã¢ÂÅ’ Failed to send test request")
        
        time.sleep(2)
        client.loop_stop()
        client.disconnect()
        publisher.disconnect()

def test_prediction_with_sample_data():
    """Test the battery prediction logic with sample data"""
    print("\n" + "="*60)
    print("TESTING BATTERY PREDICTION WITH SAMPLE DATA")
    print("="*60)
    
    model_manager = ModelManager()
    if not model_manager.load_models():
        print("Ã¢ÂÅ’ Failed to load models")
        return
    
    # Generate test data
    test_data = generate_test_battery_data()
    voltage = np.array(test_data["voltage"], dtype=np.float32)
    current = np.array(test_data["current"], dtype=np.float32)
    time_vals = np.array(test_data["time"], dtype=np.float32)
    
    print(f"\nÃ°Å¸â€œÅ  Input data shape: voltage={voltage.shape}, current={current.shape}")
    
    # Test with different LED counts
    for led_count in [1, 2, 3]:
        print(f"\n{'='*40}")
        print(f"TESTING WITH {led_count} LED(S)")
        print(f"{'='*40}")
        
        result = model_manager.predict(voltage, current, time_vals, led_count)
        
        print(f"\nÃ°Å¸â€œË† Results:")
        print(f"  SOC Baseline: {result['soc_baseline_percent']:.2f}%")
        print(f"  SOC Optimized: {result['soc_optimized_percent']:.2f}%")
        print(f"  SOC Improvement: {result['soc_improvement']:.2f}%")
        print(f"  SoH: {result['soh_percent']:.2f}%")
        print(f"  Runtime Baseline: {result['runtime_baseline_hours']:.2f} hours")
        print(f"  Runtime Optimized: {result['runtime_optimized_hours']:.2f} hours")
        print(f"  Optimizer Decision: {result['optimizer_decision']}")
        print(f"  Shed Devices: {result['shed_devices']}")
        print(f"  Model Predicted Avg SOC: {result['model_predicted_avg_soc']:.2f}%")
        print(f"\nÃ¢ÂÂ±Ã¯Â¸Â  Inference Time: {result['performance']['inference_time_ms']:.2f} ms")

# =============================
# CREATE DYNAMODB TABLE FUNCTION
# =============================
def create_dynamodb_table():
    """Create DynamoDB table for battery predictions"""
    try:
        dynamodb = boto3.resource('dynamodb', region_name=Config.REGION)
        table_name = Config.TABLE_NAME
        
        # Check if table already exists
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        if table_name in existing_tables:
            print(f"Ã¢Å“â€¦ Table {table_name} already exists")
            return True
        
        # Create the table
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {
                    'AttributeName': 'requestId',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'requestId',
                    'AttributeType': 'S'
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        
        print(f"Ã¢ÂÂ³ Creating table {table_name}...")
        table.wait_until_exists()
        print(f"Ã¢Å“â€¦ Table {table_name} created successfully!")
        
        return True
        
    except Exception as e:
        print(f"Ã¢ÂÅ’ Error creating table: {e}")
        return False

# =============================
# MAIN ENTRY POINT
# =============================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Battery Prediction System')
    parser.add_argument('--test', action='store_true', help='Send test prediction request')
    parser.add_argument('--test-local', action='store_true', help='Run local test with sample data')
    parser.add_argument('--publish-test-data', action='store_true', help='Publish test battery data')
    parser.add_argument('--no-dynamodb', action='store_true', help='Disable DynamoDB logging')
    parser.add_argument('--create-table', action='store_true', help='Create DynamoDB table')
    parser.add_argument('--simulate-arduino', action='store_true', help='Force simulation mode even if Arduino connected')
    parser.add_argument('--led-count', type=int, default=1, choices=[1,2,3], help='Default LED count')
    args = parser.parse_args()
    
    if args.no_dynamodb:
        Config.ENABLE_DYNAMODB = False
        logger.info("Ã¢â€žÂ¹Ã¯Â¸Â DynamoDB logging disabled by command line")
    
    if args.create_table:
        create_dynamodb_table()
        return
    
    if args.test_local:
        test_prediction_with_sample_data()
        return
    
    if args.test:
        send_test_prediction_request()
        return
    
    if args.publish_test_data:
        publisher = BatteryDataPublisher()
        if publisher.connect():
            test_data = generate_test_battery_data()
            # Publish each data point
            for i in range(len(test_data["voltage"])):
                publisher.publish_battery_data(
                    test_data["voltage"][i],
                    test_data["current"][i],
                    test_data["time"][i],
                    "Dewhara_Test"
                )
                time.sleep(0.1)
            publisher.disconnect()
        return
    
    # Run main application
    app = BatteryPredictionApp()
    
    # Set default LED count if provided
    if args.led_count:
        app.arduino_collector.set_led_count(args.led_count)
    
    if app.initialize():
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            app.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        app.start()
    else:
        logger.error("Ã¢ÂÅ’ Failed to initialize application")
        sys.exit(1)

if __name__ == "__main__":
    main()
