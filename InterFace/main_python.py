#!/usr/bin/env python3
"""
Main IoT Core Application - Reads Arduino Serial data and sends for solar prediction
"""

import json
import logging
import os
import sys
import threading
import queue
import time
from datetime import datetime
from typing import Dict, Any, Optional
import signal
from pathlib import Path
import serial
import serial.tools.list_ports

# AWS IoT Core imports
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iot_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArduinoReader:
    """Reads data from Arduino via Serial"""
    
    def __init__(self, port=None, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.running = False
        self.latest_data = {}
        self.data_history = []  # Store last 24 readings for prediction
        self.lock = threading.Lock()
        
    def find_arduino_port(self):
        """Auto-detect Arduino port"""
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            # Arduino typically identifies as such
            if 'Arduino' in port.description or 'USB Serial' in port.description:
                logger.info(f"Found Arduino on {port.device}")
                return port.device
        return None
    
    def connect(self):
        """Connect to Arduino"""
        try:
            # Auto-detect port if not specified
            if not self.port:
                self.port = self.find_arduino_port()
            
            if not self.port:
                logger.error("Could not find Arduino port")
                return False
            
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1
            )
            logger.info(f"✅ Connected to Arduino on {self.port}")
            
            # Clear buffer
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def parse_arduino_data(self, line: str) -> Dict[str, float]:
        """
        Parse Arduino data format:
        SERVO9:20,SERVO10:40,SERVO11:40,SERVO12:30,TEMP:25.9,HUM:51.0,LUX:464.17,SOLAR:4.13,BATT:3.68
        """
        try:
            data = {}
            # Clean the line
            line = line.strip()
            
            # Split by comma
            pairs = line.split(',')
            
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    try:
                        # Convert to float if possible
                        data[key] = float(value)
                    except ValueError:
                        # Keep as string if not numeric
                        data[key] = value
            
            return data
            
        except Exception as e:
            logger.error(f"Error parsing Arduino data: {e}")
            return {}
    
    def read_loop(self):
        """Continuous reading from Arduino"""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    # Read line from Arduino
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    if line:
                        logger.debug(f"Arduino: {line}")
                        
                        # Parse the data
                        data = self.parse_arduino_data(line)
                        
                        if data:
                            with self.lock:
                                self.latest_data = data
                                self.data_history.append(data)
                                
                                # Keep only last 100 readings
                                if len(self.data_history) > 100:
                                    self.data_history = self.data_history[-100:]
                            
                            # Log important values
                            if 'SOLAR' in data:
                                logger.info(f"📊 SOLAR: {data['SOLAR']}V, BATT: {data.get('BATT', 'N/A')}V")
                            
                            # If we have a callback, trigger it
                            if hasattr(self, 'on_new_data') and self.on_new_data:
                                self.on_new_data(data)
                    
                time.sleep(0.1)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                logger.error(f"Error reading from Arduino: {e}")
                time.sleep(1)
    
    def start(self, on_data_callback=None):
        """Start reading Arduino data"""
        self.on_new_data = on_data_callback
        self.running = True
        thread = threading.Thread(target=self.read_loop, daemon=True)
        thread.start()
        logger.info("📡 Arduino reader started")
        
    def stop(self):
        """Stop reading and close connection"""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
            logger.info("🔌 Arduino disconnected")
    
    def get_sensor_data_for_prediction(self) -> Optional[list]:
        """
        Format sensor data for solar prediction model
        Returns 24x3 array: [irradiance, temperature, humidity] for 24 hours
        """
        with self.lock:
            if len(self.data_history) < 24:
                logger.warning(f"Not enough data for prediction: {len(self.data_history)}/24 readings")
                return None
            
            # Get last 24 readings
            last_24 = self.data_history[-24:]
            
            # Format for model: 24 rows of [LUX/1000, TEMP, HUM]
            # LUX is divided by 1000 to normalize (typical range 0-1000+)
            sensor_array = []
            for reading in last_24:
                # Get values with defaults if missing
                lux = reading.get('LUX', 0) / 1000.0  # Normalize
                temp = reading.get('TEMP', 25.0)
                hum = reading.get('HUM', 50.0)
                
                sensor_array.append([lux, temp, hum])
            
            return sensor_array


class SolarPredictionHandler:
    """
    Handler for solar prediction results
    """
    
    def __init__(self):
        self.predictions = {}
        self.prediction_callbacks = {}
        self.lock = threading.Lock()
        
    def handle_prediction_result(self, payload: Dict[str, Any]):
        """Handle incoming prediction result"""
        try:
            request_id = payload.get('requestId')
            if not request_id:
                request_id = payload.get('request_id')
            
            if not request_id:
                logger.error("No requestId in prediction result")
                return
            
            # Extract prediction data
            total_energy_formatted = payload.get('total_energy', payload.get('total_energy_formatted', 'N/A'))
            total_energy_kwh = payload.get('total_energy_kwh', 0)
            device_id = payload.get('deviceId', 'unknown')
            timestamp = payload.get('timestamp', time.time())
            
            # Store the prediction
            with self.lock:
                self.predictions[request_id] = {
                    'request_id': request_id,
                    'device_id': device_id,
                    'total_energy_formatted': total_energy_formatted,
                    'total_energy_kwh': total_energy_kwh,
                    'timestamp': timestamp,
                    'received_at': time.time(),
                    'full_response': payload
                }
                
                # If there's a callback waiting, execute it
                if request_id in self.prediction_callbacks:
                    callback = self.prediction_callbacks.pop(request_id)
                    callback(self.predictions[request_id])
            
            # Log the prediction (this is the 24h KWH result you need)
            logger.info("=" * 60)
            logger.info(f"✅ SOLAR PREDICTION RESULT RECEIVED")
            logger.info(f"📊 Request ID: {request_id}")
            logger.info(f"📱 Device: {device_id}")
            logger.info(f"⚡ {total_energy_formatted}")
            logger.info(f"🔢 Raw Value: {total_energy_kwh} kWh")
            logger.info("=" * 60)
            
            # Save to file
            self._save_prediction(self.predictions[request_id])
            
        except Exception as e:
            logger.error(f"Error handling prediction result: {e}")
    
    def _save_prediction(self, prediction: Dict[str, Any]):
        """Save prediction to file"""
        try:
            filename = f"predictions/prediction_{prediction['request_id']}.json"
            os.makedirs("predictions", exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(prediction, f, indent=2)
            
            logger.info(f"💾 Prediction saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
    
    def get_prediction(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored prediction"""
        with self.lock:
            return self.predictions.get(request_id)
    
    def wait_for_prediction(self, request_id: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for a specific prediction"""
        result_queue = queue.Queue()
        
        def callback(prediction):
            result_queue.put(prediction)
        
        with self.lock:
            if request_id in self.predictions:
                return self.predictions[request_id]
            self.prediction_callbacks[request_id] = callback
        
        try:
            return result_queue.get(timeout=timeout)
        except queue.Empty:
            with self.lock:
                self.prediction_callbacks.pop(request_id, None)
            logger.error(f"Timeout waiting for prediction {request_id}")
            return None


class IoTModelProcessor:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the IoT Model Processor
        """
        self.config = self.load_config(config_path)
        self.models = {}
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.processing_threads = []
        
        # AWS IoT Core client
        self.mqtt_client = None
        
        # Arduino reader
        self.arduino = ArduinoReader(
            port=self.config.get("arduino_port", None),
            baud_rate=self.config.get("arduino_baud", 9600)
        )
        
        # Solar prediction handler
        self.solar_handler = SolarPredictionHandler()
        
        # Mobile request handler (new)
        self.mobile_request_topic = "raspberrypi/request"
        self.mobile_response_topic = "raspberrypi/userappvisits/data"
        
        # Auto-prediction timer
        self.prediction_timer = None
        self.prediction_interval = self.config.get("prediction_interval", 3600)  # Default: 1 hour
        
        # Setup paths
        self.models_path = Path(self.config.get("models_path", "./models"))
        self.data_path = Path(self.config.get("data_path", "./data"))
        
        # Create directories
        self.models_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        Path("predictions").mkdir(exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        
        # Certificate paths (relative to script directory)
        default_config = {
            "aws_iot": {
                "endpoint": "ajmja1mzmi1j4-ats.iot.eu-north-1.amazonaws.com",
                "port": 8883,
                "root_ca": str(script_dir / "AmazonRootCA1"),  # Direct file in script directory
                "private_key": str(script_dir / "private.pem.key"),  # Direct file in script directory
                "certificate": str(script_dir / "certificate.pem.crt"),  # Direct file in script directory
                "client_id": "iot_model_processor",
                "subscribe_topics": [
                    "solar/prediction/result",
                    "solar/prediction/response",
                    "raspberrypi/request"  # Add mobile request topic
                ],
                "publish_topic": "solar/prediction/request",
                "mobile_response_topic": "raspberrypi/userappvisits/data"
            },
            "arduino_port": '/dev/ttyACM0',
            "arduino_baud": 9600,
            "models_path": "./models",
            "data_path": "./data",
            "num_processing_threads": 4,
            "prediction_interval": 3600,
            "auto_predict": False,  # Disable auto-predict by default
            "min_readings_for_prediction": 24
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Loaded configuration from {config_path}")
            else:
                # Save default config with absolute paths
                with open(config_path, 'w') as f:
                    # Convert Path objects to strings for JSON serialization
                    serializable_config = {}
                    for key, value in default_config.items():
                        if key == "aws_iot":
                            serializable_config[key] = {}
                            for k, v in value.items():
                                serializable_config[key][k] = str(v) if isinstance(v, Path) else v
                        else:
                            serializable_config[key] = value
                    json.dump(serializable_config, f, indent=2)
                logger.info(f"Created default configuration at {config_path}")
                logger.info(f"Certificate paths set to: {default_config['aws_iot']['root_ca']}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def verify_certificates(self):
        """Verify that certificate files exist"""
        aws_config = self.config.get("aws_iot", {})
        
        cert_files = [
            ("Root CA", aws_config.get("root_ca")),
            ("Private Key", aws_config.get("private_key")),
            ("Certificate", aws_config.get("certificate"))
        ]
        
        all_exist = True
        for cert_name, cert_path in cert_files:
            if cert_path and os.path.exists(cert_path):
                size = os.path.getsize(cert_path)
                logger.info(f"✅ {cert_name} found: {cert_path} ({size} bytes)")
            else:
                logger.error(f"❌ {cert_name} NOT found at: {cert_path}")
                all_exist = False
        
        return all_exist
    
    def setup_aws_iot(self):
        """Setup AWS IoT Core MQTT client"""
        aws_config = self.config.get("aws_iot", {})
        
        # Verify certificates first
        if not self.verify_certificates():
            error_msg = "Certificate files missing. Please check the paths."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Create MQTT client
            self.mqtt_client = AWSIoTMQTTClient(aws_config.get("client_id"))
            self.mqtt_client.configureEndpoint(
                aws_config.get("endpoint"),
                aws_config.get("port", 8883)
            )
            self.mqtt_client.configureCredentials(
                aws_config.get("root_ca"),
                aws_config.get("private_key"),
                aws_config.get("certificate")
            )
            
            # Configure MQTT client
            self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
            self.mqtt_client.configureOfflinePublishQueueing(-1)
            self.mqtt_client.configureDrainingFrequency(2)
            self.mqtt_client.configureConnectDisconnectTimeout(10)
            self.mqtt_client.configureMQTTOperationTimeout(5)
            
            # Connect
            logger.info("📡 Connecting to AWS IoT Core...")
            self.mqtt_client.connect()
            logger.info("✅ Connected to AWS IoT Core")
            
            # Subscribe to topics
            subscribe_topics = aws_config.get("subscribe_topics", [])
            for topic in subscribe_topics:
                self.mqtt_client.subscribe(topic, 1, self.mqtt_message_callback)
                logger.info(f"📡 Subscribed to topic: {topic}")
            
        except Exception as e:
            logger.error(f"Error setting up AWS IoT: {e}")
            raise
    
    def mqtt_message_callback(self, client, userdata, message):
        """Callback for incoming MQTT messages"""
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            topic = message.topic
            logger.info(f"Received message on topic {topic}")
            
            # Handle solar prediction results
            if topic in ["solar/prediction/result", "solar/prediction/response"]:
                self.solar_handler.handle_prediction_result(payload)
            
            # Handle mobile requests
            elif topic == "raspberrypi/request":
                self.handle_mobile_request(payload)
            
            else:
                # Handle other requests
                self.request_queue.put({
                    'topic': topic,
                    'payload': payload,
                    'timestamp': time.time()
                })
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON payload: {message.payload}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def handle_mobile_request(self, payload: Dict[str, Any]):
        """Handle mobile request - trigger prediction and send response"""
        try:
            request_id = payload.get('requestId') or payload.get('request_id')
            
            if not request_id:
                logger.error("No requestId in mobile request")
                return
            
            logger.info(f"📱 Mobile request received with ID: {request_id}")
            
            # Get sensor data for prediction
            sensor_data = self.arduino.get_sensor_data_for_prediction()
            
            if not sensor_data:
                # If not enough historical data, use latest reading
                latest = self.arduino.latest_data
                if latest:
                    logger.warning("Not enough historical data, using latest reading repeated")
                    lux = latest.get('LUX', 0) / 1000.0
                    temp = latest.get('TEMP', 25.0)
                    hum = latest.get('HUM', 50.0)
                    sensor_data = [[lux, temp, hum]] * 24
                else:
                    # No data at all
                    error_response = {
                        'requestId': request_id,
                        'status': 'error',
                        'message': 'No sensor data available from Arduino',
                        'timestamp': time.time()
                    }
                    self.mqtt_client.publish(
                        self.config["aws_iot"].get("mobile_response_topic", "raspberrypi/userappvisits/data"),
                        json.dumps(error_response),
                        1
                    )
                    return
            
            # Send prediction request
            pred_request_id = self.send_prediction_request(sensor_data, f"Mobile_{request_id}")
            
            if not pred_request_id:
                error_response = {
                    'requestId': request_id,
                    'status': 'error',
                    'message': 'Failed to send prediction request',
                    'timestamp': time.time()
                }
                self.mqtt_client.publish(
                    self.config["aws_iot"].get("mobile_response_topic", "raspberrypi/userappvisits/data"),
                    json.dumps(error_response),
                    1
                )
                return
            
            # Wait for prediction result
            logger.info(f"⏳ Waiting for prediction result for mobile request {request_id}")
            prediction = self.solar_handler.wait_for_prediction(pred_request_id, timeout=60)
            
            if prediction:
                # Format response for mobile
                response = {
                    'requestId': request_id,
                    'status': 'success',
                    'data': {
                        'total_energy_kwh': prediction['total_energy_kwh'],
                        'total_energy_formatted': prediction['total_energy_formatted'],
                        'timestamp': time.time(),
                        'prediction_time': prediction.get('timestamp', time.time())
                    }
                }
                logger.info(f"✅ Mobile response prepared: {prediction['total_energy_formatted']}")
            else:
                response = {
                    'requestId': request_id,
                    'status': 'error',
                    'message': 'Prediction timeout or failed',
                    'timestamp': time.time()
                }
                logger.warning(f"⚠️ Prediction failed for mobile request {request_id}")
            
            # Publish response
            self.mqtt_client.publish(
                self.config["aws_iot"].get("mobile_response_topic", "raspberrypi/userappvisits/data"),
                json.dumps(response),
                1
            )
            logger.info(f"📤 Mobile response sent to {self.config['aws_iot'].get('mobile_response_topic')}")
            
        except Exception as e:
            logger.error(f"Error handling mobile request: {e}")
    
    def send_prediction_request(self, sensor_data: list, device_id: str = "Arduino_RPi") -> str:
        """
        Send a prediction request to the solar predictor
        """
        try:
            # Generate unique request ID
            request_id = f"pred_{int(time.time() * 1000)}_{os.getpid()}"
            
            # Prepare request payload
            request = {
                "requestId": request_id,
                "deviceId": device_id,
                "sensor_data": sensor_data
            }
            
            # Publish to request topic
            publish_topic = self.config["aws_iot"]["publish_topic"]
            self.mqtt_client.publish(
                publish_topic,
                json.dumps(request),
                1
            )
            
            logger.info(f"📤 Sent solar prediction request {request_id}")
            logger.info(f"   Sensor data: {len(sensor_data)} readings prepared")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error sending prediction request: {e}")
            return None
    
    def on_arduino_data(self, data: Dict[str, float]):
        """
        Callback when new Arduino data arrives
        """
        # Just log, no auto-prediction
        pass
    
    def auto_predict(self):
        """
        Automatically send prediction when enough data is collected
        """
        while self.running:
            try:
                # Check if we have enough readings
                sensor_data = self.arduino.get_sensor_data_for_prediction()
                
                if sensor_data:
                    logger.info(f"📊 Collected {len(sensor_data)} sensor readings, sending prediction...")
                    
                    # Send prediction request
                    request_id = self.send_prediction_request(sensor_data, "Arduino_Auto")
                    
                    if request_id:
                        # Wait for prediction (optional)
                        prediction = self.solar_handler.wait_for_prediction(request_id, timeout=60)
                        
                        if prediction:
                            logger.info(f"✅ Auto-prediction complete: {prediction['total_energy_formatted']}")
                        else:
                            logger.warning("⏰ Auto-prediction timeout - result may arrive later")
                
                # Wait for next interval
                for _ in range(self.prediction_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in auto-predict: {e}")
                time.sleep(60)
    
    def load_all_models(self):
        """Load all models from the models directory"""
        logger.info("Starting to load all models...")
        logger.info("Model loading complete (using cloud models via AWS IoT)")
    
    def start(self):
        """Start the IoT Model Processor"""
        logger.info("=" * 60)
        logger.info("Starting IoT Model Processor with Arduino Integration")
        logger.info("=" * 60)
        
        # Load models (optional - if you have local models)
        self.load_all_models()
        
        # Connect to Arduino
        if self.arduino.connect():
            self.arduino.start(on_data_callback=self.on_arduino_data)
        else:
            logger.error("Failed to connect to Arduino")
            return
        
        # Setup AWS IoT
        try:
            self.setup_aws_iot()
        except Exception as e:
            logger.error(f"Failed to setup AWS IoT: {e}")
            logger.warning("Continuing without AWS IoT (local mode)")
            # Don't return - we can still run without IoT for testing
        
        # Start auto-prediction if enabled
        if self.config.get("auto_predict", False):
            self.prediction_timer = threading.Thread(target=self.auto_predict, daemon=True)
            self.prediction_timer.start()
            logger.info(f"⏰ Auto-prediction enabled (every {self.prediction_interval} seconds)")
        else:
            logger.info("⏸️ Auto-prediction disabled - will predict only on mobile requests")
        
        # Start processing threads
        self.running = True
        num_threads = self.config.get("num_processing_threads", 4)
        
        for i in range(num_threads):
            thread = threading.Thread(target=self.worker_thread, name=f"Worker-{i}")
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
            logger.info(f"Started worker thread: Worker-{i}")
        
        logger.info("✅ IoT Model Processor started successfully")
        logger.info("📡 Listening for:")
        logger.info("   - Arduino data (continuous)")
        logger.info("   - Solar prediction results (solar/prediction/result)")
        logger.info("   - Mobile requests (raspberrypi/request)")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop")
        
        # Keep main thread alive
        try:
            while self.running:
                # Display latest Arduino data periodically
                if self.arduino.latest_data:
                    latest = self.arduino.latest_data
                    readings_count = len(self.arduino.data_history)
                    sys.stdout.write(f"\r📊 Latest: SOLAR={latest.get('SOLAR', 'N/A')}V, "
                                   f"TEMP={latest.get('TEMP', 'N/A')}°C, "
                                   f"LUX={latest.get('LUX', 'N/A')} | "
                                   f"Readings: {readings_count}/24  ")
                    sys.stdout.flush()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def worker_thread(self):
        """Worker thread for processing requests"""
        while self.running:
            try:
                request = self.request_queue.get(timeout=1)
                # Process other model requests if needed
                self.request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    def stop(self):
        """Stop the IoT Model Processor"""
        logger.info("\n" + "=" * 60)
        logger.info("Stopping IoT Model Processor...")
        
        self.running = False
        
        # Stop Arduino reader
        self.arduino.stop()
        
        # Wait for threads
        for thread in self.processing_threads:
            thread.join(timeout=5)
        
        # Disconnect MQTT
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
                logger.info("Disconnected from AWS IoT Core")
            except:
                pass
        
        logger.info("✅ IoT Model Processor stopped")
        logger.info("=" * 60)


def manual_prediction(processor):
    """Manually trigger a prediction with current Arduino data"""
    print("\n" + "=" * 60)
    print("MANUAL PREDICTION REQUEST")
    print("=" * 60)
    
    sensor_data = processor.arduino.get_sensor_data_for_prediction()
    
    if sensor_data:
        print(f"✅ Collected {len(sensor_data)} sensor readings")
        print("Sending prediction request...")
        
        request_id = processor.send_prediction_request(sensor_data, "Manual_Trigger")
        
        if request_id:
            print(f"⏳ Waiting for prediction (request ID: {request_id})...")
            prediction = processor.solar_handler.wait_for_prediction(request_id, timeout=30)
            
            if prediction:
                print(f"\n✅ PREDICTION RECEIVED:")
                print(f"   {prediction['total_energy_formatted']}")
                print(f"   Raw: {prediction['total_energy_kwh']} kWh")
            else:
                print("\n❌ Timeout waiting for prediction")
    else:
        print(f"❌ Not enough data. Need 24 readings, have {len(processor.arduino.data_history)}")


def test_mobile_request(processor=None):
    """Test function to simulate a mobile request"""
    print("\n" + "=" * 60)
    print("TESTING MOBILE REQUEST HANDLER")
    print("=" * 60)
    
    # Create a test processor if not provided
    if not processor:
        processor = IoTModelProcessor()
        try:
            processor.setup_aws_iot()
        except Exception as e:
            print(f"❌ Failed to connect to AWS IoT: {e}")
            return
    
    # Simulate a mobile request
    test_request = {
        "requestId": "EN123456789",
        "deviceId": "test_mobile",
        "timestamp": time.time(),
        "command": "get_prediction"
    }
    
    print(f"\n📱 Sending test mobile request: {json.dumps(test_request, indent=2)}")
    
    # Publish to request topic
    processor.mqtt_client.publish(
        "raspberrypi/request",
        json.dumps(test_request),
        1
    )
    
    print("✅ Test request sent. Check logs for response.")
    print("\n⏳ Waiting 10 seconds for response...")
    
    # Wait and then check if we got a response
    time.sleep(10)
    
    # Check if any prediction was received
    if processor.solar_handler.predictions:
        print("\n✅ Predictions received:")
        for req_id, pred in processor.solar_handler.predictions.items():
            print(f"   - {req_id}: {pred.get('total_energy_formatted', 'N/A')}")
    else:
        print("\n⚠️ No predictions received yet")
    
    processor.mqtt_client.disconnect()


def setup_signal_handlers(processor):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        processor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     IoT Core Model Processor - Arduino + Mobile Integration      ║
    ║                                                                    ║
    ║     Reading: TEMP, HUM, LUX, SOLAR, BATT from Arduino            ║
    ║     Predicts 24h solar energy when mobile requests arrive        ║
    ║     Responds with prediction to raspberrypi/userappvisits/data   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    import argparse
    parser = argparse.ArgumentParser(description='IoT Model Processor with Arduino')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--predict-now', action='store_true', help='Trigger manual prediction')
    parser.add_argument('--test-mobile', action='store_true', help='Test mobile request handler')
    parser.add_argument('--auto-predict', action='store_true', help='Enable auto-prediction')
    parser.add_argument('--check-certs', action='store_true', help='Check certificate files')
    args = parser.parse_args()
    
    # Get config path
    config_path = os.environ.get('CONFIG_PATH', args.config)
    
    # Create processor instance
    processor = IoTModelProcessor(config_path)
    
    # Check certificates if requested
    if args.check_certs:
        processor.verify_certificates()
        return
    
    if args.test_mobile:
        # For test mobile, we need AWS IoT connection
        try:
            processor.setup_aws_iot()
            test_mobile_request(processor)
        except Exception as e:
            logger.error(f"Failed to setup AWS IoT for test: {e}")
            processor.verify_certificates()
        return
    
    # Override auto_predict if specified
    if args.auto_predict:
        processor.config['auto_predict'] = True
        logger.info("Auto-prediction enabled via command line")
    
    # Connect to Arduino first (needed for manual prediction)
    if not processor.arduino.connect():
        logger.error("Failed to connect to Arduino")
        if not args.predict_now:
            return
    
    if args.predict_now:
        # Just do a manual prediction and exit
        processor.arduino.start()  # Start reading
        time.sleep(2)  # Let it collect some data
        manual_prediction(processor)
        processor.arduino.stop()
        return
    
    # Setup signal handlers
    setup_signal_handlers(processor)
    
    # Start processor
    try:
        processor.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        processor.verify_certificates()
        sys.exit(1)

if __name__ == "__main__":
    main()