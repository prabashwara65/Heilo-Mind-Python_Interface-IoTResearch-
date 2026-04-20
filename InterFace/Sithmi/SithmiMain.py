#!/usr/bin/env python3
"""
Solar Prediction Main Program
Handles Arduino serial data, prediction requests, and MQTT responses
"""

import os
import json
import time
import threading
import queue
import numpy as np
import tensorflow as tf
import joblib
import boto3
import ssl
import signal
import sys
import psutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from decimal import Decimal
import serial
import serial.tools.list_ports
from collections import deque

# AWS IoT Core SDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Paho MQTT for client publishing
import paho.mqtt.client as mqtt

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =============================
# CONFIGURATION
# =============================
class Config:
    """Central configuration class"""
    
    # AWS IoT Core Settings (Server/Listener)
    SERVER_CLIENT_ID = "solar_predictor_server"
    AWS_IOT_ENDPOINT = "ajmja1mzmi1j4-ats.iot.eu-north-1.amazonaws.com"
    REQUEST_TOPIC = "solar/prediction/request"
    RESPONSE_TOPIC = "solar/prediction/response"
    
    # Client Publisher Settings
    CLIENT_ID = "hasara_client"
    CLIENT_PUBLISH_TOPIC = "solar/prediction/result"
    
    # DynamoDB
    TABLE_NAME = "SithmiSolarPredictResults"
    REGION = "eu-north-1"
    ENABLE_DYNAMODB = True
    
    # Paths
    ROOT_DIR = Path(__file__).resolve().parent
    CERTS_DIR = ROOT_DIR / "certs"
    MODEL_DIR = ROOT_DIR / "model"
    
    # Certificate paths (Server)
    SERVER_ROOT_CA = CERTS_DIR / "AmazonRootCA1.pem"
    SERVER_CERT = CERTS_DIR / "certificate.pem.crt"
    SERVER_PRIVATE_KEY = CERTS_DIR / "private.pem.key"
    
    # Certificate paths (Client)
    CLIENT_CERT = CERTS_DIR / "6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-certificate.pem.crt"
    CLIENT_KEY = CERTS_DIR / "6c2a210110a2809a43a9da4b7f2c58bb1ae4fc5e4cc7d35a5f9747eb84709ce8-private.pem.key"
    CLIENT_ROOT_CA = CERTS_DIR / "AmazonRootCA1.pem"
    
    # Model paths
    KERAS_MODEL_PATH = MODEL_DIR / "solar_lstm.keras"
    SCALER_X_PATH = MODEL_DIR / "scaler_X.save"
    SCALER_Y_PATH = MODEL_DIR / "scaler_y.save"
    
    # Performance monitoring
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_LOG_INTERVAL = 3600
    
    # Threading
    NUM_WORKER_THREADS = 4
    REQUEST_QUEUE_SIZE = 100
    
    # Prediction settings
    NUM_WARMUP_PREDICTIONS = 50
    
    # Arduino Serial Settings
    ARDUINO_BAUD_RATE = 9600
    ARDUINO_DATA_POINTS = 24  # 24 hours of data
    ARDUINO_DATA_COLS = 3      # [irradiance, temperature, humidity]
    SERIAL_TIMEOUT = 1

# =============================
# LOGGING SETUP
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solar_predictor.log'),
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
# ARDUINO DATA COLLECTOR
# =============================
class ArduinoDataCollector:
    """Handles reading and buffering data from Arduino via serial - COMPLETELY SILENT"""
    
    def __init__(self):
        self.serial_connection = None
        self.data_buffer = deque(maxlen=Config.ARDUINO_DATA_POINTS)
        self.port = self._find_arduino_port()
        self.running = False
        self.collection_thread = None
        
    def _find_arduino_port(self):
        """Automatically find Arduino port - SILENT"""
        try:
            ports = list(serial.tools.list_ports.comports())
            
            for port in ports:
                if 'Arduino' in port.description or 'USB Serial' in port.description:
                    logger.info(f"✅ Arduino found: {port.device}")
                    return port.device
            
            common_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', 'COM3', 'COM4']
            for port in common_ports:
                if os.path.exists(port):
                    logger.info(f"✅ Using port: {port}")
                    return port
        except:
            pass
        
        logger.warning("⚠️ No Arduino - using simulation mode")
        return None
    
    def connect(self):
        """Connect to Arduino"""
        if not self.port:
            return False
        
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=Config.ARDUINO_BAUD_RATE,
                timeout=Config.SERIAL_TIMEOUT
            )
            time.sleep(2)
            logger.info(f"✅ Connected to Arduino")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    def read_arduino_data(self):
        """Read Arduino data - NO LOGS"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return None
        
        try:
            if self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode('utf-8').strip()
                
                if not line:
                    return None
                
                # Parse key-value pairs
                data_parts = line.split(',')
                sensor_data = {}
                
                for part in data_parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        try:
                            sensor_data[key] = float(value)
                        except:
                            pass
                
                # Extract solar values
                if 'LUX' in sensor_data:
                    irradiance = min(1.0, sensor_data['LUX'] / 100000.0)
                elif 'SOLAR' in sensor_data:
                    irradiance = min(1.0, sensor_data['SOLAR'] / 5.0)
                else:
                    irradiance = 0.0
                
                temperature = sensor_data.get('TEMP', 25.0)
                humidity = sensor_data.get('HUM', 50.0)
                
                # Constrain
                irradiance = max(0.0, min(1.0, irradiance))
                temperature = max(0.0, min(50.0, temperature))
                humidity = max(0.0, min(100.0, humidity))
                
                return [irradiance, temperature, humidity]
                
        except:
            pass
        
        return None
    
    def _collection_loop(self):
        """Background data collection - COMPLETELY SILENT"""
        while self.running:
            data_point = self.read_arduino_data()
            
            if data_point:
                self.data_buffer.append(data_point)
            else:
                # Silent simulation fallback
                simulated_data = self._generate_simulated_data_point()
                self.data_buffer.append(simulated_data)
            
            time.sleep(1)
    
    def _generate_simulated_data_point(self):
        """Generate simulated data - SILENT"""
        hour = datetime.now().hour
        if 6 <= hour <= 18:
            irradiance = np.sin((hour - 6) * np.pi / 12) ** 2
        else:
            irradiance = 0
            
        temperature = 25 + 10 * irradiance + np.random.normal(0, 0.5)
        humidity = 60 - 30 * irradiance + np.random.normal(0, 2)
        
        temperature = max(0, min(50, temperature))
        humidity = max(0, min(100, humidity))
        
        return [float(irradiance), float(temperature), float(humidity)]
    
    def start_collection(self):
        """Start background data collection"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("📊 Data collection started")
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            logger.info("✅ Disconnected")
    
    def get_current_data_array(self):
        """Get current data buffer as numpy array"""
        data_list = list(self.data_buffer)
        
        if len(data_list) < Config.ARDUINO_DATA_POINTS:
            padding = Config.ARDUINO_DATA_POINTS - len(data_list)
            data_list = [[0, 25, 50]] * padding + data_list
        
        data_list = data_list[-Config.ARDUINO_DATA_POINTS:]
        return np.array(data_list, dtype=np.float32)
    
    def is_buffer_ready(self):
        """Check if we have enough data"""
        return len(self.data_buffer) >= Config.ARDUINO_DATA_POINTS
    
    def get_buffer_status(self):
        """Get buffer status - SILENT"""
        return {
            "size": len(self.data_buffer),
            "target": Config.ARDUINO_DATA_POINTS,
            "ready": len(self.data_buffer) >= Config.ARDUINO_DATA_POINTS,
            "source": "arduino" if self.serial_connection else "simulated"
        }
# =============================
# MODEL MANAGER
# =============================
class ModelManager:
    """Singleton class to manage model and scalers"""
    
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
        self.scaler_X = None
        self.scaler_Y = None
        self.model_size_mb = 0
        self.load_time = 0
        
        # Performance metrics
        self.inference_times = []
        self.total_predictions = 0
        self.cpu_usage_samples = []
        self.ram_usage_samples = []
        
        self._initialized = True
    
    def load_models(self):
        """Load all models and scalers"""
        logger.info("=" * 50)
        logger.info("LOADING SOLAR PREDICTION MODEL")
        logger.info("=" * 50)
        
        try:
            # Check if model files exist
            if not Config.KERAS_MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {Config.KERAS_MODEL_PATH}")
            if not Config.SCALER_X_PATH.exists():
                raise FileNotFoundError(f"Scaler X not found at {Config.SCALER_X_PATH}")
            if not Config.SCALER_Y_PATH.exists():
                raise FileNotFoundError(f"Scaler Y not found at {Config.SCALER_Y_PATH}")
            
            # Load model with timing
            start_time = time.time()
            
            logger.info(f"Loading Keras model from: {Config.KERAS_MODEL_PATH}")
            self.model = tf.keras.models.load_model(Config.KERAS_MODEL_PATH, compile=False)
            
            logger.info(f"Loading X scaler from: {Config.SCALER_X_PATH}")
            self.scaler_X = joblib.load(Config.SCALER_X_PATH)
            
            logger.info(f"Loading Y scaler from: {Config.SCALER_Y_PATH}")
            self.scaler_Y = joblib.load(Config.SCALER_Y_PATH)
            
            self.load_time = time.time() - start_time
            self.model_size_mb = os.path.getsize(Config.KERAS_MODEL_PATH) / (1024 * 1024)
            
            logger.info(f"✅ Model loaded successfully in {self.load_time:.2f} seconds")
            logger.info(f"📊 Model size: {self.model_size_mb:.2f} MB")
            
            # Log model summary
            self.model.summary(print_fn=lambda x: logger.info(f"  {x}"))
            
            # Perform warmup predictions for accurate timing
            self._warmup_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def _warmup_model(self):
        """Perform warmup predictions to stabilize performance measurements"""
        logger.info("🔥 Warming up model for accurate performance measurements...")
        
        # Create dummy data for warmup
        dummy_data = np.zeros((1, 24, 3), dtype=np.float32)
        dummy_scaled = self.scaler_X.transform(dummy_data.reshape(24, 3)).reshape(1, 24, 3)
        
        # Run warmup predictions
        for i in range(Config.NUM_WARMUP_PREDICTIONS):
            self.model.predict(dummy_scaled, verbose=0)
            if (i + 1) % 10 == 0:
                logger.info(f"  Warmup {i + 1}/{Config.NUM_WARMUP_PREDICTIONS} complete")
        
        logger.info("✅ Model warmup complete")
    
    def prepare_input(self, recent_data: np.ndarray) -> np.ndarray:
        """Scale and reshape input data"""
        try:
            # Ensure correct shape
            if len(recent_data.shape) == 2:
                recent_data = recent_data.reshape(1, 24, 3)
            
            # Validate shape
            if recent_data.shape != (1, 24, 3):
                raise ValueError(f"Invalid input shape: {recent_data.shape}, expected (1, 24, 3)")
            
            # Reshape to (24, 3) for scaling
            reshaped = recent_data.reshape(24, 3)
            
            # Scale the data
            scaled = self.scaler_X.transform(reshaped)
            
            # Return to original shape (1, 24, 3)
            return scaled.reshape(1, 24, 3)
            
        except Exception as e:
            logger.error(f"Error preparing input: {e}")
            raise
    
    def predict(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Run prediction on input data with detailed performance metrics
        
        Args:
            data: numpy array of shape (24, 3) or (1, 24, 3)
            
        Returns:
            Dictionary containing prediction results and performance metrics
        """
        try:
            # Track performance
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)
            
            # Prepare input
            scaled_data = self.prepare_input(data)
            
            # Run prediction with timing
            start_time = time.time()
            pred_norm = self.model.predict(scaled_data, verbose=0)
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Store inference time for statistics
            self.inference_times.append(inference_time_ms)
            self.total_predictions += 1
            
            # Get memory after prediction
            memory_after = process.memory_info().rss / (1024 * 1024)
            ram_usage_mb = memory_after
            self.ram_usage_samples.append(ram_usage_mb)
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage_samples.append(cpu_percent)
            
            # Inverse transform predictions to get energy in kWh
            energy_kwh = self.scaler_Y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
            
            # Extract irradiance from original data
            if len(data.shape) == 3:
                irradiance = data[0, :, 0].flatten()
            else:
                irradiance = data[:, 0]
            
            # Apply physics constraints
            # No energy at night (irradiance = 0)
            energy_kwh[irradiance == 0] = 0
            # No negative energy
            energy_kwh[energy_kwh < 0] = 0
            
            # Calculate total energy
            total_energy = float(np.sum(energy_kwh))
            
            # Prepare hourly breakdown
            hourly_energy = []
            hourly_irradiance = []
            for i in range(24):
                hourly_energy.append(float(energy_kwh[i]))
                hourly_irradiance.append(float(irradiance[i]))
            
            # Calculate performance statistics
            avg_inference_time = np.mean(self.inference_times[-100:]) if self.inference_times else inference_time_ms
            
            result = {
                "total_energy_kwh": total_energy,
                "hourly_energy": hourly_energy,
                "hourly_irradiance": hourly_irradiance,
                "performance": {
                    "inference_time_ms": inference_time_ms,
                    "avg_inference_time_ms": float(avg_inference_time),
                    "ram_usage_mb": ram_usage_mb,
                    "cpu_percent": cpu_percent,
                    "memory_delta_mb": memory_after - memory_before,
                    "total_predictions": self.total_predictions
                }
            }
            
            logger.debug(f"Prediction complete - Total energy: {total_energy:.3f} kWh")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.inference_times:
            return {}
        
        return {
            "model_size_mb": self.model_size_mb,
            "load_time_seconds": self.load_time,
            "total_predictions": self.total_predictions,
            "inference_time_ms": {
                "mean": float(np.mean(self.inference_times[-100:])),
                "median": float(np.median(self.inference_times[-100:])),
                "std": float(np.std(self.inference_times[-100:])),
                "min": float(np.min(self.inference_times[-100:])),
                "max": float(np.max(self.inference_times[-100:]))
            },
            "ram_usage_mb": {
                "mean": float(np.mean(self.ram_usage_samples[-100:])),
                "current": self.ram_usage_samples[-1] if self.ram_usage_samples else 0
            },
            "cpu_percent": {
                "mean": float(np.mean(self.cpu_usage_samples[-100:])),
                "current": self.cpu_usage_samples[-1] if self.cpu_usage_samples else 0
            },
            "thread_count": threading.active_count()
        }
    
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
class SolarPredictionServer:
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
        
        # Initialize DynamoDB only if enabled
        if Config.ENABLE_DYNAMODB:
            try:
                self.dynamodb = boto3.resource("dynamodb", region_name=Config.REGION)
                self.table = self.dynamodb.Table(Config.TABLE_NAME)
                logger.info("✅ DynamoDB initialized")
            except Exception as e:
                logger.warning(f"⚠️ DynamoDB initialization failed: {e}")
                self.dynamodb = None
        else:
            logger.info("ℹ️ DynamoDB logging is disabled")
    
    def connect(self):
        """Connect to AWS IoT Core"""
        try:
            self.mqtt_client = AWSIoTMQTTClient(Config.SERVER_CLIENT_ID)
            
            # Configure endpoint
            self.mqtt_client.configureEndpoint(Config.AWS_IOT_ENDPOINT, 8883)
            
            # Configure credentials
            self.mqtt_client.configureCredentials(
                str(Config.SERVER_ROOT_CA),
                str(Config.SERVER_PRIVATE_KEY),
                str(Config.SERVER_CERT)
            )
            
            # Configure MQTT client
            self.mqtt_client.configureOfflinePublishQueueing(-1)
            self.mqtt_client.configureDrainingFrequency(2)
            self.mqtt_client.configureConnectDisconnectTimeout(10)
            self.mqtt_client.configureMQTTOperationTimeout(5)
            self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
            
            # Connect
            logger.info("📡 Connecting to AWS IoT Core...")
            self.mqtt_client.connect()
            logger.info("✅ Connected to AWS IoT Core")
            
            # Subscribe to request topic
            self.mqtt_client.subscribe(Config.REQUEST_TOPIC, 1, self._message_callback)
            logger.info(f"📡 Subscribed to topic: {Config.REQUEST_TOPIC}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to AWS IoT Core: {e}")
            return False
    
    def _message_callback(self, client, userdata, message):
        """Callback for incoming MQTT messages"""
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            request_id = payload.get('requestId') or payload.get('request_id')
            logger.info(f"📩 Request received with requestId: {request_id}")
            
            # Add to processing queue
            self.request_queue.put({
                'topic': message.topic,
                'payload': payload,
                'timestamp': time.time()
            })
            
            logger.debug(f"Queue size: {self.request_queue.qsize()}")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON payload: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def _process_request(self, request: Dict[str, Any]):
        """Process a single prediction request"""
        try:
            payload = request['payload']
            
            # Get requestId
            request_id = payload.get('requestId') or payload.get('request_id')
            if not request_id:
                request_id = f"auto_{int(time.time() * 1000)}"
                logger.warning(f"No requestId provided, generated: {request_id}")
            
            device_id = payload.get('deviceId', 'unknown')
            
            logger.info(f"🔄 Processing request {request_id} from device {device_id}")
            
            # Get buffer status
            buffer_status = self.arduino_collector.get_buffer_status()
            logger.info(f"📊 Buffer status: {buffer_status['size']}/24 points (Source: {buffer_status['source']})")
            
            # Use data from Arduino collector
            sensor_array = self.arduino_collector.get_current_data_array()
            
            # Log sample of the data being used
            logger.info(f"📊 Latest readings (last 5): {sensor_array[-5:].tolist()}")
            
            # Run prediction
            result = self.model_manager.predict(sensor_array)
            
            # Prepare response
            response = {
                "requestId": request_id,
                "deviceId": device_id,
                "timestamp": time.time(),
                "status": "success",
                "total_energy_kwh": result["total_energy_kwh"],
                "hourly_energy": result["hourly_energy"],
                "hourly_irradiance": result["hourly_irradiance"],
                "performance": result["performance"],
                "data_source": buffer_status['source'],
                "buffer_size": buffer_status['size'],
                "buffer_ready": buffer_status['ready']
            }
            
            # Publish response to response topic
            self.mqtt_client.publish(
                Config.RESPONSE_TOPIC,
                json.dumps(response),
                1
            )
            logger.info(f"✅ Response sent for request {request_id} (inference: {result['performance']['inference_time_ms']:.2f}ms)")
            
            # Also publish to result topic
            self.mqtt_client.publish(
                Config.CLIENT_PUBLISH_TOPIC,
                json.dumps(response),
                1
            )
            logger.info(f"✅ Result published to {Config.CLIENT_PUBLISH_TOPIC}")
            
            # Log to DynamoDB if available
            if self.dynamodb and self.table:
                try:
                    response['timestamp_str'] = datetime.fromtimestamp(response['timestamp']).isoformat()
                    dynamodb_item = convert_floats_to_decimal(response)
                    self.table.put_item(Item=dynamodb_item)
                    logger.info(f"💾 Prediction logged to DynamoDB for request {request_id}")
                except Exception as e:
                    logger.error(f"❌ DynamoDB logging failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            # Get request_id safely
            error_request_id = 'unknown'
            if 'payload' in locals():
                error_request_id = payload.get('requestId') or payload.get('request_id', 'unknown')
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
            
            logger.info(f"⚠️ Error response sent for request {request_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send error response: {e}")
    
    def _worker_thread(self, thread_id: int):
        """Worker thread for processing requests"""
        logger.info(f"🧵 Worker thread {thread_id} started")
        
        while self.running:
            try:
                request = self.request_queue.get(timeout=1)
                self._process_request(request)
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Worker thread {thread_id} error: {e}")
        
        logger.info(f"🧵 Worker thread {thread_id} stopped")
    
    def start(self):
        """Start the server"""
        if not self.connect():
            logger.error("❌ Failed to start server")
            return False
        
        self.running = True
        
        # Start worker threads
        for i in range(Config.NUM_WORKER_THREADS):
            thread = threading.Thread(
                target=self._worker_thread,
                args=(i,),
                name=f"Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"🚀 Server started with {Config.NUM_WORKER_THREADS} worker threads")
        return True
    
    def stop(self):
        """Stop the server"""
        logger.info("🛑 Stopping server...")
        self.running = False
        
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
                logger.info("✅ Disconnected from AWS IoT Core")
            except:
                pass
        
        logger.info("✅ Server stopped")

# =============================
# MQTT CLIENT (Data Publisher)
# =============================
class SolarDataPublisher:
    """Publishes sensor data to AWS IoT Core"""
    
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
            
            # Configure TLS
            self.client.tls_set(
                ca_certs=str(Config.CLIENT_ROOT_CA),
                certfile=str(Config.CLIENT_CERT),
                keyfile=str(Config.CLIENT_KEY),
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLSv1_2
            )
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            
            # Connect
            logger.info("📡 Connecting publisher to AWS IoT Core...")
            self.client.connect(Config.AWS_IOT_ENDPOINT, 8883, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info("✅ Publisher connected to AWS IoT Core")
                return True
            else:
                logger.error("❌ Publisher connection timeout")
                return False
            
        except Exception as e:
            logger.error(f"❌ Publisher connection failed: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Connection callback"""
        if rc == 0:
            self.connected = True
            logger.info("✅ Publisher connected successfully")
        else:
            logger.error(f"❌ Publisher connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Disconnection callback"""
        self.connected = False
        logger.warning("⚠️ Publisher disconnected")
    
    def _on_publish(self, client, userdata, mid, rc, properties=None):
        """Publish callback"""
        logger.debug(f"📤 Message {mid} published")
    
    def publish_sensor_data(self, sensor_data: List[List[float]], device_id: str = "Raspberry") -> bool:
        """Publish sensor data to AWS IoT Core"""
        if not self.connected:
            logger.error("❌ Publisher not connected")
            return False
        
        try:
            payload = {
                "deviceId": device_id,
                "timestamp": time.time(),
                "data": sensor_data
            }
            
            result = self.client.publish(
                Config.CLIENT_PUBLISH_TOPIC,
                json.dumps(payload),
                qos=1
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"📤 Sensor data published to {Config.CLIENT_PUBLISH_TOPIC}")
                return True
            else:
                logger.error(f"❌ Publish failed with code: {result.rc}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to publish sensor data: {e}")
            return False
    
    def disconnect(self):
        """Disconnect publisher"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("✅ Publisher disconnected")

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
        logger.info("📊 Performance monitor started")
        
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.metrics_history:
            metrics_file = Path(f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"📊 Performance metrics saved to {metrics_file}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self.model_manager.get_performance_metrics()
                metrics['timestamp'] = time.time()
                metrics['timestamp_str'] = datetime.now().isoformat()
                
                self.metrics_history.append(metrics)
                
                logger.info("📊 Performance Metrics:")
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
                logger.error(f"❌ Performance monitor error: {e}")
                time.sleep(60)

# =============================
# MAIN APPLICATION
# =============================
class SolarPredictionApp:
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
        logger.info("SOLAR PREDICTION SYSTEM - INITIALIZING")
        logger.info("=" * 60)
        
        # Load model
        if not self.model_manager.load_models():
            return False
        
        # Connect to Arduino
        if self.arduino_collector.connect():
            logger.info("✅ Arduino detected - using real sensor data")
        else:
            logger.warning("⚠️ No Arduino detected - running in simulation mode")
        
        # Start Arduino data collection
        self.arduino_collector.start_collection()
        
        # Initialize server with Arduino collector
        self.server = SolarPredictionServer(self.model_manager, self.arduino_collector)
        
        # Initialize publisher
        self.publisher = SolarDataPublisher()
        
        # Initialize performance monitor
        if Config.ENABLE_PERFORMANCE_MONITORING:
            self.monitor = PerformanceMonitor(self.model_manager)
        
        logger.info("=" * 60)
        logger.info("✅ SYSTEM INITIALIZED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
    
    def start(self):
        """Start all components"""
        logger.info("🚀 STARTING SOLAR PREDICTION SYSTEM")
        
        # Start server
        if not self.server.start():
            logger.error("❌ Failed to start server")
            return
        
        # Connect publisher
        if not self.publisher.connect():
            logger.warning("⚠️ Publisher not connected - continuing without publishing")
        
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
        logger.info("🛑 SHUTTING DOWN SOLAR PREDICTION SYSTEM")
        
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
        
        logger.info("👋 SYSTEM SHUTDOWN COMPLETE")
    
    def _print_system_info(self):
        """Print system information"""
        buffer_status = self.arduino_collector.get_buffer_status()
        
        logger.info("=" * 60)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 60)
        logger.info(f"📡 Server listening on: {Config.REQUEST_TOPIC}")
        logger.info(f"📤 Publishing results to: {Config.RESPONSE_TOPIC}")
        logger.info(f"📤 Also publishing to: {Config.CLIENT_PUBLISH_TOPIC}")
        logger.info(f"🧵 Worker threads: {Config.NUM_WORKER_THREADS}")
        logger.info(f"📊 Performance monitoring: {'Enabled' if Config.ENABLE_PERFORMANCE_MONITORING else 'Disabled'}")
        logger.info(f"💾 DynamoDB logging: {'Enabled' if Config.ENABLE_DYNAMODB else 'Disabled'}")
        logger.info(f"🔌 Arduino: {buffer_status['source'].upper()}")
        logger.info(f"📊 Data buffer: {buffer_status['size']}/24 points")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to shutdown")
        logger.info("=" * 60)

# =============================
# UTILITY FUNCTIONS
# =============================
def generate_test_sensor_data() -> List[List[float]]:
    """Generate test sensor data for 24 hours"""
    hours = 24
    data = []
    
    for hour in range(hours):
        if 6 <= hour <= 18:
            irradiance = np.sin((hour - 6) * np.pi / 12) ** 2
        else:
            irradiance = 0
        
        temperature = 20 + 10 * irradiance + np.random.normal(0, 1)
        humidity = 60 - 30 * irradiance + np.random.normal(0, 5)
        
        data.append([
            float(irradiance),
            float(temperature),
            float(max(0, min(100, humidity)))
        ])
    
    return data

def send_test_prediction_request():
    """Send a test prediction request"""
    publisher = SolarDataPublisher()
    if publisher.connect():
        sensor_data = generate_test_sensor_data()
        request_id = f"test_{int(time.time() * 1000)}"
        
        request = {
            "requestId": request_id,
            "deviceId": "Raspberry_Test",
            "sensor_data": sensor_data
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
        
        print(f"\n📤 Sending test request with requestId: {request_id}")
        result = client.publish(Config.REQUEST_TOPIC, json.dumps(request), qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"✅ Test prediction request sent with requestId: {request_id}")
            print(f"✅ Request sent. Check response on topic: {Config.RESPONSE_TOPIC}")
        else:
            logger.error("❌ Failed to send test request")
        
        time.sleep(2)
        client.loop_stop()
        client.disconnect()
        publisher.disconnect()

def test_prediction_with_your_data():
    """Test the prediction logic with sample data"""
    print("\n" + "="*60)
    print("TESTING PREDICTION WITH SAMPLE DATA")
    print("="*60)
    
    model_manager = ModelManager()
    if not model_manager.load_models():
        print("❌ Failed to load models")
        return
    
    recent_data = np.array([
        [0.00, 0.55, 0.85], [0.00, 0.54, 0.86], [0.00, 0.53, 0.87],
        [0.00, 0.52, 0.88], [0.00, 0.51, 0.89], [0.00, 0.50, 0.90],
        [0.15, 0.55, 0.85], [0.30, 0.60, 0.80], [0.50, 0.65, 0.75],
        [0.70, 0.70, 0.70], [0.90, 0.75, 0.65], [1.00, 0.78, 0.60],
        [0.95, 0.77, 0.62], [0.80, 0.74, 0.65], [0.60, 0.70, 0.70],
        [0.40, 0.65, 0.75], [0.20, 0.60, 0.80], [0.05, 0.58, 0.82],
        [0.00, 0.55, 0.85], [0.00, 0.54, 0.86], [0.00, 0.53, 0.87],
        [0.00, 0.52, 0.88], [0.00, 0.51, 0.89], [0.00, 0.50, 0.90],
    ], dtype=np.float32)
    
    print("\n📊 Input data shape:", recent_data.shape)
    
    results = []
    for i in range(10):
        result = model_manager.predict(recent_data)
        results.append(result)
        
        if i == 0:
            print("\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            print(f"\n📈 Total 24h Energy: {result['total_energy_kwh']:.3f} kWh")
            print("\nHour | Energy (kWh) | Irradiance")
            print("-" * 35)
            for hour in range(24):
                print(f"{hour+1:02d}   | {result['hourly_energy'][hour]:.3f}      | {result['hourly_irradiance'][hour]:.3f}")
            
            print("\n" + "="*60)
            print("PERFORMANCE METRICS")
            print("="*60)
            print(f"⏱️  Inference Time: {result['performance']['inference_time_ms']:.2f} ms")
            print(f"💾 RAM Usage: {result['performance']['ram_usage_mb']:.1f} MB")
            print(f"⚡ CPU Usage: {result['performance']['cpu_percent']:.1f}%")
    
    avg_times = [r['performance']['inference_time_ms'] for r in results]
    print(f"\n📊 Average inference time (10 runs): {np.mean(avg_times):.2f} ms")
    print(f"📊 Min inference time: {np.min(avg_times):.2f} ms")
    print(f"📊 Max inference time: {np.max(avg_times):.2f} ms")
    
    summary = model_manager.get_performance_summary()
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"📊 Total Predictions: {summary['total_predictions']}")
    print(f"⏱️  Mean Inference Time: {summary['inference_time_ms']['mean']:.2f} ms")
    print(f"📈 Inference Time Std: {summary['inference_time_ms']['std']:.2f} ms")
    print(f"💾 Mean RAM Usage: {summary['ram_usage_mb']['mean']:.1f} MB")
    print(f"⚡ Mean CPU Usage: {summary['cpu_percent']['mean']:.1f}%")

# =============================
# MAIN ENTRY POINT
# =============================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Solar Prediction System')
    parser.add_argument('--test', action='store_true', help='Send test prediction request')
    parser.add_argument('--test-local', action='store_true', help='Run local test with sample data')
    parser.add_argument('--publish-test-data', action='store_true', help='Publish test sensor data')
    parser.add_argument('--no-dynamodb', action='store_true', help='Disable DynamoDB logging')
    parser.add_argument('--simulate-arduino', action='store_true', help='Force simulation mode even if Arduino connected')
    args = parser.parse_args()
    
    if args.no_dynamodb:
        Config.ENABLE_DYNAMODB = False
        logger.info("ℹ️ DynamoDB logging disabled by command line")
    
    if args.test_local:
        test_prediction_with_your_data()
        return
    
    if args.test:
        send_test_prediction_request()
        return
    
    if args.publish_test_data:
        publisher = SolarDataPublisher()
        if publisher.connect():
            test_data = generate_test_sensor_data()
            publisher.publish_sensor_data(test_data, "Raspberry_Test")
            publisher.disconnect()
        return
    
    # Run main application
    app = SolarPredictionApp()
    
    if app.initialize():
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            app.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        app.start()
    else:
        logger.error("❌ Failed to initialize application")
        sys.exit(1)

if __name__ == "__main__":
    main()
