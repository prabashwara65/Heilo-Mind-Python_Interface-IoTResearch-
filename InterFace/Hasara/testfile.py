import time
import logging
import json
import sys
import datetime
import os
from pathlib import Path
import serial
import serial.tools.list_ports


import numpy as np
import joblib
import pandas as pd

# Try to import tflite_runtime (for Raspberry Pi)
# Fall back to full tensorflow if not available (for development)
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
    print("Using tflite_runtime (optimized for Raspberry Pi)")
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TFLITE_AVAILABLE = True
        print("Using TensorFlow Lite from full TensorFlow")
    except ImportError:
        TFLITE_AVAILABLE = False
        print("ERROR: Neither tflite_runtime nor tensorflow is installed!")
        print("Please install tflite_runtime for Raspberry Pi:")
        print("  pip install tflite-runtime")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.queue_manager import QueueManager
from src.retry_manager import RetryManager
from src.aws_sender import publish_data


# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'controller.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class TFLiteSyncScheduler:
    """Wrapper for TensorFlow Lite sync scheduler model"""
    
    def __init__(self, model_path='models/sync_scheduler/'):
        self.model_path = Path(model_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.feature_names = None
        self.scaler = None
        self.input_size = None
        self.is_loaded = False
        
    def load_model(self, filename='sync_scheduler_model_float16.tflite'):
        """Load TFLite model"""
        if not TFLITE_AVAILABLE:
            logging.error("TFLite not available. Cannot load model.")
            return False
            
        model_file = self.model_path / filename
        
        if not model_file.exists():
            # Try other formats
            alternatives = ['sync_scheduler_model_dynamic.tflite', 
                           'sync_scheduler_model_none.tflite',
                           'sync_scheduler_model.tflite']
            for alt in alternatives:
                if (self.model_path / alt).exists():
                    model_file = self.model_path / alt
                    logging.info(f"Using alternative model: {alt}")
                    break
            else:
                logging.error(f"No TFLite model found in {self.model_path}")
                return False
        
        try:
            # Load TFLite model
            self.interpreter = Interpreter(model_path=str(model_file))
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_size = self.input_details[0]['shape'][1]
            
            logging.info(f"Loaded TFLite model from {model_file}")
            logging.info(f"Input shape: {self.input_details[0]['shape']}, dtype: {self.input_details[0]['dtype']}")
            logging.info(f"Output shape: {self.output_details[0]['shape']}, dtype: {self.output_details[0]['dtype']}")
            
            # Load feature names
            feat_file = self.model_path / 'feature_names.txt'
            if feat_file.exists():
                with open(feat_file, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                logging.info(f"Loaded {len(self.feature_names)} features")
            
            # Load scaler
            scaler_file = self.model_path / 'feature_scaler.pkl'
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logging.info("Loaded feature scaler")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logging.error(f"Failed to load TFLite model: {e}")
            return False
    
    def predict(self, X):
        """Run inference with TFLite model"""
        if not self.is_loaded or self.interpreter is None:
            raise ValueError("Model not loaded")
        
        # Ensure input is correct type and shape
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Check input size
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {X.shape[1]}")
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output


class TFLiteAnomalyDetector:
    """Wrapper for TensorFlow Lite anomaly detector models"""
    
    def __init__(self, model_path='models/anomaly_detector/'):
        self.model_path = Path(model_path)
        self.ae_interpreter = None
        self.enc_interpreter = None
        self.threshold = None
        self.feature_names = None
        self.scaler = None
        self.input_size = None
        self.is_loaded = False
        
    def load_autoencoder(self, filename='autoencoder_float16.tflite'):
        """Load autoencoder TFLite model"""
        if not TFLITE_AVAILABLE:
            logging.error("TFLite not available. Cannot load model.")
            return False
            
        model_file = self.model_path / filename
        
        if not model_file.exists():
            # Try other formats
            alternatives = ['autoencoder_dynamic.tflite', 
                           'autoencoder_none.tflite',
                           'anomaly_autoencoder.tflite']
            for alt in alternatives:
                if (self.model_path / alt).exists():
                    model_file = self.model_path / alt
                    logging.info(f"Using alternative autoencoder: {alt}")
                    break
            else:
                logging.warning(f"No autoencoder TFLite model found in {self.model_path}")
                return False
        
        try:
            # Load TFLite model
            self.ae_interpreter = Interpreter(model_path=str(model_file))
            self.ae_interpreter.allocate_tensors()
            
            input_details = self.ae_interpreter.get_input_details()
            self.input_size = input_details[0]['shape'][1]
            
            logging.info(f"Loaded autoencoder TFLite model from {model_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load autoencoder: {e}")
            return False
    
    def load_encoder(self, filename='encoder_float16.tflite'):
        """Load encoder TFLite model"""
        if not TFLITE_AVAILABLE:
            return False
            
        model_file = self.model_path / filename
        
        if not model_file.exists():
            # Try other formats
            alternatives = ['encoder_dynamic.tflite', 
                           'encoder_none.tflite',
                           'anomaly_encoder.tflite']
            for alt in alternatives:
                if (self.model_path / alt).exists():
                    model_file = self.model_path / alt
                    logging.info(f"Using alternative encoder: {alt}")
                    break
            else:
                logging.debug(f"No encoder TFLite model found in {self.model_path}")
                return False
        
        try:
            # Load TFLite model
            self.enc_interpreter = Interpreter(model_path=str(model_file))
            self.enc_interpreter.allocate_tensors()
            
            logging.info(f"Loaded encoder TFLite model from {model_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load encoder: {e}")
            return False
    
    def load_model(self):
        """Load both autoencoder and encoder if available"""
        # Load autoencoder
        ae_loaded = self.load_autoencoder()
        
        # Load encoder (optional)
        self.load_encoder()
        
        # Load threshold
        threshold_file = self.model_path / 'threshold.txt'
        if threshold_file.exists():
            try:
                with open(threshold_file, 'r') as f:
                    self.threshold = float(f.read().strip())
                logging.info(f"Loaded anomaly threshold: {self.threshold:.6f}")
            except:
                logging.warning("Could not load threshold file")
        
        # Load feature names
        feat_file = self.model_path / 'feature_names.txt'
        if feat_file.exists():
            try:
                with open(feat_file, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                logging.info(f"Loaded {len(self.feature_names)} anomaly features")
            except:
                logging.warning("Could not load feature names")
        
        # Load scaler
        scaler_file = self.model_path / 'scaler.pkl'
        if scaler_file.exists():
            try:
                self.scaler = joblib.load(scaler_file)
                logging.info("Loaded anomaly scaler")
            except:
                logging.warning("Could not load anomaly scaler")
        
        self.is_loaded = ae_loaded
        return ae_loaded
    
    def check(self, X):
        """
        Check if samples are anomalies
        Returns: (is_anomaly, reconstruction_error)
        """
        if self.ae_interpreter is None:
            raise ValueError("Autoencoder not loaded")
        
        # Ensure input is correct type and shape
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Check input size
        if X.shape[1] != self.input_size:
            logging.warning(f"Expected {self.input_size} features, got {X.shape[1]}")
            # Pad or truncate if needed
            if X.shape[1] < self.input_size:
                # Pad with zeros
                X_padded = np.zeros((1, self.input_size))
                X_padded[0, :X.shape[1]] = X[0]
                X = X_padded
            else:
                # Truncate
                X = X[:, :self.input_size]
        
        # Get input/output details
        input_details = self.ae_interpreter.get_input_details()
        output_details = self.ae_interpreter.get_output_details()
        
        # Run inference
        self.ae_interpreter.set_tensor(input_details[0]['index'], X)
        self.ae_interpreter.invoke()
        reconstruction = self.ae_interpreter.get_tensor(output_details[0]['index'])
        
        # Calculate MSE
        mse = float(np.mean(np.square(X - reconstruction), axis=1)[0])
        
        # Check against threshold
        is_anomaly = mse > self.threshold if self.threshold is not None else False
        
        return is_anomaly, mse


class main_controller:
    def __init__(self, config_path='config.json'):
        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize components
        self.queue = QueueManager()
        self.retry = RetryManager(
            max_retries=self.config.get('max_retries', 5),
            base_delay=self.config.get('base_delay', 1),
            max_delay=self.config.get('max_delay', 60)
        )

        # --- Serial Communication Setup (for Arduino) ---
        self.serial_port = None
        self.setup_serial_connection()
        
        # --- Sync Scheduler (TFLite version) ---
        self.sync_scheduler = TFLiteSyncScheduler(model_path='models/sync_scheduler/')
        self.sync_features = []
        self.scaler = None
        
        try:
            if self.sync_scheduler.load_model():
                self.sync_features = self.sync_scheduler.feature_names or []
                self.scaler = self.sync_scheduler.scaler
                logging.info("Sync scheduler: TFLite model loaded successfully")
                logging.info(f"Sync features: {self.sync_features}")
            else:
                logging.error("Failed to load sync scheduler model")
                logging.warning("Continuing without sync scheduler")
        except Exception as e:
            logging.error(f"Failed to load sync scheduler: {e}")
            logging.warning("Continuing without sync scheduler")

        # --- Anomaly Detector (TFLite version) ---
        self.anomaly_detector = TFLiteAnomalyDetector(model_path='models/anomaly_detector/')
        self.anomaly_features = []
        self.anomaly_scaler = None
        
        try:
            if self.anomaly_detector.load_model():
                self.anomaly_features = self.anomaly_detector.feature_names or []
                self.anomaly_scaler = self.anomaly_detector.scaler
                logging.info("Anomaly detector: TFLite model loaded successfully")
                logging.info(f"Anomaly features: {self.anomaly_features}")
            else:
                logging.warning("Failed to load anomaly detector model")
        except Exception as e:
            logging.warning(f"Failed to load anomaly detector: {e}")

        # User behaviour simulation parameter
        self.user_class = self.config.get('user_class', 3)
        self.running = False
        
        # For tracking last valid readings
        self.last_valid_sensor_data = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Statistics
        self.total_readings = 0
        self.arduino_readings = 0
        self.simulated_readings = 0
        
        # Transmission disabled flag (to avoid errors when no modem)
        self.transmission_enabled = self.config.get('transmission_enabled', False)

        # NB-IoT and Firebase (disabled)
        self.nbiot = None
        self.firebase = None

        # Model info
        self._log_model_info()

    def _log_model_info(self):
        """Log information about loaded models"""
        logging.info("="*60)
        logging.info("MODELS LOADED")
        logging.info("="*60)
        
        # Sync scheduler info
        logging.info("Sync Scheduler:")
        if self.sync_features:
            logging.info(f"  - Type: TensorFlow Lite")
            logging.info(f"  - Features: {len(self.sync_features)}")
            logging.info(f"  - Features list: {self.sync_features}")
            logging.info(f"  - Scaler: {'Loaded' if self.scaler else 'Not loaded'}")
        else:
            logging.info("  - Not loaded")
        
        # Anomaly detector info
        logging.info("Anomaly Detector:")
        if self.anomaly_features:
            logging.info(f"  - Type: TensorFlow Lite")
            logging.info(f"  - Features: {len(self.anomaly_features)}")
            logging.info(f"  - Features list: {self.anomaly_features}")
            logging.info(f"  - Threshold: {self.anomaly_detector.threshold:.6f}" if self.anomaly_detector.threshold else "  - Threshold: Not set")
            logging.info(f"  - Autoencoder: {'Loaded' if self.anomaly_detector.ae_interpreter else 'Not loaded'}")
        else:
            logging.info("  - Not loaded")
        
        logging.info("="*60)

    def setup_serial_connection(self):
        """Setup serial connection with Arduino with enhanced debugging"""
        serial_config = self.config.get('serial', {})
        port = serial_config.get('port', 'auto')
        baudrate = serial_config.get('baudrate', 9600)
        timeout = serial_config.get('timeout', 1)
        
        logging.info("="*60)
        logging.info("SERIAL PORT DETECTION")
        logging.info("="*60)
        
        # List all available ports
        ports = list(serial.tools.list_ports.comports())
        logging.info(f"Found {len(ports)} serial ports:")
        for p in ports:
            logging.info(f"  - {p.device}: {p.description} (VID:{p.vid}, PID:{p.pid})")
        
        if port == 'auto':
            # Auto-detect Arduino port
            arduino_ports = []
            for p in ports:
                desc_lower = p.description.lower()
                if any(keyword in desc_lower for keyword in ['arduino', 'ch340', 'ch341', 'usb serial', 'ftdi', 'cp210x']):
                    arduino_ports.append(p.device)
                    logging.info(f"  ✅ Arduino candidate: {p.device} - {p.description}")
            
            if arduino_ports:
                port = arduino_ports[0]
                logging.info(f"Auto-detected Arduino on {port}")
            else:
                logging.warning("No Arduino-specific ports found. Trying common ports...")
                # Fallback to common ports for Raspberry Pi and Windows
                common_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyAMA0', 
                               'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9']
                
                for p in common_ports:
                    try:
                        # Test if port exists
                        test_ser = serial.Serial(p, baudrate, timeout=1)
                        test_ser.close()
                        port = p
                        logging.info(f"Found working port: {port}")
                        break
                    except Exception as e:
                        logging.debug(f"Port {p} not available: {e}")
                        continue
        
        try:
            logging.info(f"Attempting to connect to {port} at {baudrate} baud...")
            self.serial_port = serial.Serial(
                port=port, 
                baudrate=baudrate, 
                timeout=timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            time.sleep(2)  # Wait for Arduino reset
            
            # Clear any initial garbage
            self.serial_port.reset_input_buffer()
            
            # Test read to verify connection
            logging.info("Testing serial connection...")
            test_start = time.time()
            while time.time() - test_start < 3:  # Try for 3 seconds
                if self.serial_port.in_waiting:
                    test_line = self.serial_port.readline().decode().strip()
                    if test_line:
                        logging.info(f"✅ Received test data: {test_line[:50]}...")
                        logging.info(f"Connected to Arduino on {port}")
                        self.consecutive_errors = 0
                        return
                time.sleep(0.1)
            
            logging.warning("Connected but no data received yet - this might be normal")
            logging.info(f"Connected to Arduino on {port}")
            self.consecutive_errors = 0
            
        except Exception as e:
            logging.error(f"❌ Failed to connect to Arduino on {port}: {e}")
            self.serial_port = None

    def test_arduino_communication(self):
        """Test Arduino communication directly"""
        if not self.serial_port:
            logging.error("No serial port connected")
            return False
        
        logging.info("="*60)
        logging.info("TESTING ARDUINO COMMUNICATION")
        logging.info("="*60)
        logging.info("Reading for 10 seconds... Press Ctrl+C to stop early")
        logging.info("="*60)
        
        try:
            start_time = time.time()
            lines_received = 0
            
            while time.time() - start_time < 10:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode().strip()
                    if line:
                        lines_received += 1
                        print(f"\n[{lines_received}] {line}")
                        
                        # Parse and display values
                        data_pairs = line.split(',')
                        parsed = {}
                        for pair in data_pairs:
                            if ':' in pair:
                                key, value = pair.split(':', 1)
                                parsed[key] = value
                        
                        # Show key values
                        temp = parsed.get('TEMP', 'N/A')
                        hum = parsed.get('HUM', 'N/A')
                        lux = parsed.get('LUX', 'N/A')
                        solar = parsed.get('SOLAR', 'N/A')
                        batt = parsed.get('BATT', 'N/A')
                        
                        print(f"   TEMP={temp}°C, HUM={hum}%, LUX={lux}, SOLAR={solar}V, BATT={batt}V")
                
                time.sleep(0.01)
            
            logging.info(f"\n✅ Received {lines_received} lines in 10 seconds")
            return lines_received > 0
            
        except KeyboardInterrupt:
            logging.info("\nTest stopped by user")
            return lines_received > 0
        except Exception as e:
            logging.error(f"Error testing Arduino: {e}")
            return False

    def read_sensors(self):
        """
        Read sensor data from Arduino via serial
        Format from Arduino: SERVO9:20,SERVO10:40,SERVO11:40,SERVO12:30,TEMP:25.9,HUM:51.1,LUX:496.67,SOLAR:4.10,BATT:3.59
        Returns a dictionary of current sensor readings.
        """
        self.total_readings += 1
        
        if not self.serial_port:
            logging.error("No serial connection - cannot read sensors")
            # Don't use simulated data, return None to indicate failure
            return None
        
        try:
            # Read line from Arduino with timeout
            if self.serial_port.in_waiting:
                line = self.serial_port.readline().decode().strip()
                
                if line:
                    logging.debug(f"Raw Arduino data: {line}")
                    
                    # Parse the Arduino format: KEY:VALUE pairs separated by commas
                    data_pairs = line.split(',')
                    parsed_data = {}
                    
                    for pair in data_pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            try:
                                # Try to convert to float if possible
                                parsed_data[key] = float(value)
                            except ValueError:
                                # Keep as string if not numeric
                                parsed_data[key] = value
                    
                    # Check if we have the expected sensor data
                    required_keys = ['TEMP', 'HUM', 'LUX', 'SOLAR']
                    
                    if all(key in parsed_data for key in required_keys):
                        temp = parsed_data['TEMP']
                        hum = parsed_data['HUM']
                        lux = parsed_data['LUX']
                        solar_volt = parsed_data['SOLAR']
                        battery_volt = parsed_data.get('BATT', 3.7)  # Default if missing
                        
                        # Log the extracted values
                        logging.info(f"✅ REAL Arduino data: TEMP={temp}C, HUM={hum}%, LUX={lux}, SOLAR={solar_volt}V, BATT={battery_volt}V")
                        
                        # Validate readings (basic sanity checks)
                        if not (-40 <= temp <= 80):
                            logging.warning(f"Invalid temperature reading: {temp} C")
                            return None
                        
                        if not (0 <= hum <= 100):
                            logging.warning(f"Invalid humidity reading: {hum} %")
                            return None
                        
                        if not (0 <= lux <= 200000):
                            logging.warning(f"Invalid lux reading: {lux} lx")
                            return None
                        
                        if not (0 <= solar_volt <= 30):
                            logging.warning(f"Invalid solar voltage reading: {solar_volt} V")
                            return None
                        
                        if not (0 <= battery_volt <= 30):
                            logging.warning(f"Invalid battery voltage reading: {battery_volt} V")
                            return None
                        
                        # Calculate derived values
                        # Solar Current estimation based on lux
                        if lux > 10:  # Only calculate if there's meaningful light
                            # Rough estimation: at 100,000 lux (full sun), panel produces ~5.5A
                            solar_current = (lux / 100000.0) * 5.5
                            
                            # Adjust based on voltage (if voltage is low, current might be limited)
                            if solar_volt < 12 and solar_current > 2:
                                solar_current = min(solar_current, 2.0)  # Limit current at low voltage
                        else:
                            solar_current = 0.0
                        
                        solar_current = min(solar_current, 10.0)  # Absolute cap
                        
                        # Calculate panel power
                        panel_power = solar_volt * solar_current
                        
                        # Estimate battery SoC from voltage
                        battery_soc = self._voltage_to_soc(battery_volt)
                        
                        # Get system time features
                        now = datetime.datetime.now()
                        
                        # RSSI placeholder
                        rssi = -85
                        
                        # Create sensor data dictionary
                        sensor_data = {
                            # FROM ARDUINO (DIRECT READINGS)
                            'arduino_temperature': round(temp, 2),
                            'arduino_humidity': round(hum, 2),
                            'arduino_irradiance_lux': int(lux),
                            'arduino_panel_voltage': round(solar_volt, 3),
                            'arduino_battery_voltage': round(battery_volt, 3),
                            
                            # SERVO values (if needed)
                            'servo9': parsed_data.get('SERVO9', 0),
                            'servo10': parsed_data.get('SERVO10', 0),
                            'servo11': parsed_data.get('SERVO11', 0),
                            'servo12': parsed_data.get('SERVO12', 0),
                            
                            # CALCULATED VALUES
                            'calculated_battery_soc': battery_soc,
                            'calculated_panel_current': round(solar_current, 3),
                            'calculated_panel_power': round(panel_power, 3),
                            
                            # SYSTEM TIME FEATURES
                            'system_hour': now.hour,
                            'system_month': now.month,
                            'system_is_daytime': 1 if 6 <= now.hour < 18 else 0,
                            
                            # EXTERNAL/PLACEHOLDER
                            'external_rssi': rssi,
                            'external_app_usage': self._estimate_user_usage(now.hour),
                            
                            # METADATA
                            'timestamp': time.time(),
                            'data_source': 'arduino',
                            'reading_id': self.arduino_readings + 1,
                            'raw_line': line,  # Store raw line for debugging
                            
                            # MAPPED NAMES (for ML models)
                            'temperature': round(temp, 2),
                            'humidity': round(hum, 2),
                            'irradiance': int(lux),
                            'panel_voltage': round(solar_volt, 3),
                            'battery_voltage': round(battery_volt, 3),
                            'battery_soc': battery_soc,
                            'panel_current': round(solar_current, 3),
                            'panel_power': round(panel_power, 3),
                            'hour': now.hour,
                            'month': now.month,
                            'is_daytime': 1 if 6 <= now.hour < 18 else 0,
                            'rssi': rssi,
                            'app_usage_hourly': self._estimate_user_usage(now.hour),
                            
                            # Placeholder for missing features
                            'panel_temperature': round(temp, 2),  # Use ambient as proxy
                            'battery_current': round(solar_current * 0.5, 3),  # Rough estimate
                            'battery_temperature': round(temp, 2),  # Use ambient as proxy
                            'snr': 25,  # Default SNR value
                        }
                        
                        # Store as last valid reading
                        self.last_valid_sensor_data = sensor_data
                        self.arduino_readings += 1
                        self.consecutive_errors = 0
                        
                        # Display all values
                        self._display_sensor_values(sensor_data)
                        
                        return sensor_data
                    else:
                        missing = [key for key in required_keys if key not in parsed_data]
                        logging.warning(f"Missing required keys: {missing}. Raw data: {line}")
                        self.consecutive_errors += 1
                        
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            logging.error(f"Too many consecutive errors ({self.consecutive_errors}), reconnecting...")
                            self.reconnect_arduino()
                        
                        return None
                else:
                    # Empty line
                    return None
            else:
                # No data waiting
                return None
            
        except Exception as e:
            logging.error(f"Error reading from Arduino: {e}")
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= self.max_consecutive_errors:
                logging.error(f"Too many consecutive errors ({self.consecutive_errors}), reconnecting...")
                self.reconnect_arduino()
            
            return None
    
    def _display_sensor_values(self, sensor_data):
        """Display all sensor values in a nice formatted way"""
        print("\n" + "="*60)
        print("✅ REAL ARDUINO SENSOR READINGS")
        print("="*60)
        
        # SERVO values
        print("\nSERVO POSITIONS:")
        print(f"   SERVO9:  {sensor_data.get('servo9', 0):>3d}")
        print(f"   SERVO10: {sensor_data.get('servo10', 0):>3d}")
        print(f"   SERVO11: {sensor_data.get('servo11', 0):>3d}")
        print(f"   SERVO12: {sensor_data.get('servo12', 0):>3d}")
        
        # Arduino Direct Readings
        print("\nFROM ARDUINO:")
        print(f"   Temperature:     {sensor_data.get('arduino_temperature', 0):>8.2f} C")
        print(f"   Humidity:        {sensor_data.get('arduino_humidity', 0):>8.2f} %")
        print(f"   Irradiance:      {sensor_data.get('arduino_irradiance_lux', 0):>8d} lx")
        print(f"   Panel Voltage:   {sensor_data.get('arduino_panel_voltage', 0):>8.3f} V")
        print(f"   Battery Voltage: {sensor_data.get('arduino_battery_voltage', 0):>8.3f} V")
        
        # Calculated Values
        print("\nCALCULATED:")
        panel_current = sensor_data.get('calculated_panel_current', 0)
        panel_power = sensor_data.get('calculated_panel_power', 0)
        battery_soc = sensor_data.get('calculated_battery_soc', 0)
        
        print(f"   Panel Current:    {panel_current:>8.3f} A")
        print(f"   Panel Power:      {panel_power:>8.3f} W")
        print(f"   Battery SoC:      {battery_soc:>8.1f} %")
        
        # System Time
        print("\nSYSTEM TIME:")
        hour = sensor_data.get('system_hour', 0)
        time_str = f"{hour:02d}:00"
        day_str = "Daytime" if sensor_data.get('system_is_daytime', 0) else "Night"
        print(f"   Time:             {time_str:>8} ({day_str})")
        
        # Metadata
        print("\nMETADATA:")
        print(f"   Reading #:        {sensor_data.get('reading_id', 0):>8d}")
        print(f"   Timestamp:        {time.strftime('%H:%M:%S', time.localtime(sensor_data.get('timestamp', time.time())))}")
        
        # Raw data
        if 'raw_line' in sensor_data:
            print(f"   Raw:              {sensor_data['raw_line']}")
        
        print("="*60 + "\n")
    
    def _get_last_valid_or_simulated(self):
        """Return last valid reading - NO SIMULATED DATA"""
        if self.last_valid_sensor_data:
            logging.info("Using last valid sensor data (Arduino disconnected temporarily)")
            data = self.last_valid_sensor_data.copy()
            data['data_source'] = 'cached'
            data['timestamp'] = time.time()
            data['reading_id'] = self.total_readings
            self._display_sensor_values(data)
            return data
        else:
            logging.error("No Arduino data available and no cached data - skipping reading")
            return None

    def _voltage_to_soc(self, voltage):
        """Convert battery voltage to State of Charge"""
        # Li-ion discharge curve
        if voltage >= 4.2:
            return 100.0
        elif voltage >= 4.0:
            return 80.0 + (voltage - 4.0) * (20.0 / 0.2)
        elif voltage >= 3.8:
            return 50.0 + (voltage - 3.8) * (30.0 / 0.2)
        elif voltage >= 3.6:
            return 20.0 + (voltage - 3.6) * (30.0 / 0.2)
        elif voltage >= 3.3:
            return 5.0 + (voltage - 3.3) * (15.0 / 0.3)
        elif voltage >= 3.0:
            return (voltage - 3.0) * (5.0 / 0.3)
        else:
            return 0.0

    def _simulate_sensor_data(self):
        """REMOVED - No simulation, only real data"""
        logging.error("Simulation is disabled - only real Arduino data is used")
        return None

    def reconnect_arduino(self):
        """Attempt to reconnect to Arduino"""
        if self.serial_port:
            try:
                self.serial_port.close()
            except:
                pass
            self.serial_port = None
        
        logging.info("Attempting to reconnect to Arduino...")
        time.sleep(2)
        self.setup_serial_connection()
        self.consecutive_errors = 0

    def _estimate_user_usage(self, hour):
        """Simulate user app usage (minutes per hour)"""
        if 7 <= hour <= 9:
            return 15.0
        elif 12 <= hour <= 13:
            return 10.0
        elif 18 <= hour <= 22:
            return 25.0
        elif 23 <= hour or hour <= 5:
            return 2.0
        else:
            return 5.0

    def _build_sync_feature_vector(self, sensor_data):
        """Construct scaled feature vector for the sync scheduler"""
        if not self.sync_features or not self.scaler:
            logging.warning("Sync scheduler not properly loaded")
            return None
            
        feat_dict = {}
        missing = []
        
        for feat in self.sync_features:
            if feat in sensor_data:
                feat_dict[feat] = sensor_data[feat]
            else:
                missing.append(feat)
        
        if missing:
            logging.warning(f"Missing sync features: {missing}")
            return None
        
        try:
            df = pd.DataFrame([feat_dict])[self.sync_features]
            X_scaled = self.scaler.transform(df)
            return X_scaled
        except Exception as e:
            logging.error(f"Error building feature vector: {e}")
            return None

    def _check_anomaly(self, sensor_data):
        """Run anomaly detection using TFLite model"""
        if not self.anomaly_features or self.anomaly_detector.ae_interpreter is None:
            return False, 0.0

        input_dict = {}
        missing = []
        
        for feat in self.anomaly_features:
            if feat in sensor_data:
                input_dict[feat] = sensor_data[feat]
            else:
                missing.append(feat)
        
        if missing:
            logging.warning(f"Anomaly detection skipped – missing features: {missing}")
            return False, 0.0

        try:
            df = pd.DataFrame([input_dict])[self.anomaly_features]
            X_scaled = self.anomaly_scaler.transform(df)
            is_anomaly, error = self.anomaly_detector.check(X_scaled)
            return is_anomaly, error
        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            return False, 0.0

    def decide_sync_action(self, prob, hour):
        """Return (action, delay_minutes) based on probability and time"""
        if prob > 0.6:
            return 'sync_now', 0
        elif prob >= 0.2:
            if 6 <= hour <= 18:
                delay = max(5, int(20 * (1 - prob)))
            else:
                delay = 60
            return 'schedule', delay
        else:
            return 'skip', None


    def transmit_data(self, data):
        """Send data to AWS IoT Core"""
        try:
            publish_data(data)
            logging.info(f" Data sent to AWS IoT: reading_id={data.get('reading_id')}")
            return True
        except Exception as e:
            logging.error(f" Failed to send data to AWS IoT: {e}")
            return False

    def process_pending_queue(self, limit=5):
        """
        Process pending queue items (locally only, no transmission)
        """
        pending = self.queue.get_pending(limit)
        if pending:
            logging.info(f"📤 Would send {len(pending)} pending items (simulated)")
            for item in pending:
                logging.info(f"   - Item {item['id']} from {time.ctime(item['timestamp'])}")
                # Mark as sent locally (no actual transmission)
                self.queue.mark_sent(item['id'])
        return len(pending) if pending else 0

    def run_cycle(self):
        """One iteration of the main loop - WITH QUEUE DISPLAY"""
        sensor_data = self.read_sensors()
        
        # If no sensor data, skip this cycle
        if sensor_data is None:
            logging.warning("No sensor data available, skipping cycle")
            return
            
        hour = sensor_data['system_hour']

        # Get queue size using get_stats()
        queue_stats = self.queue.get_stats()
        queue_size = queue_stats.get('pending', 0)
        
        # Print statistics with queue size
        print(f"\n📊 Statistics: Arduino: {self.arduino_readings} | Total: {self.total_readings} | Queue: {queue_size} items")

        # 1. Anomaly detection
        is_anomaly, error = self._check_anomaly(sensor_data)
        if is_anomaly:
            logging.warning(f"🚨 ANOMALY DETECTED! Error={error:.4f}")
            print("\n" + "!"*60)
            print("🚨 ANOMALY DETECTED! 🚨")
            print(f"   Reconstruction Error: {error:.4f}")
            if self.anomaly_detector.threshold:
                print(f"   Threshold: {self.anomaly_detector.threshold:.4f}")
            print("!"*60 + "\n")
            
            # Add anomaly flag and transmit
            sensor_data['anomaly'] = True
            sensor_data['reconstruction_error'] = error
            self.transmit_data(sensor_data)
            return

        # 2. Build features and run sync scheduler
        try:
            X = self._build_sync_feature_vector(sensor_data)
            if X is not None:
                prob = float(self.sync_scheduler.predict(X)[0, 0])
                
                # DISPLAY SYNC PROBABILITY
                print(f"\n📈 Sync Probability: {prob:.3f}")
                
                action, delay = self.decide_sync_action(prob, hour)

                if action == 'sync_now':
                    logging.info(f"⚡ SYNC NOW (prob={prob:.3f})")
                    print(f"   ✅ Action: Send data immediately")
                    self.transmit_data(sensor_data)
                    
                elif action == 'schedule':
                    logging.info(f"⏰ SCHEDULE in {delay} min (prob={prob:.3f})")
                    print(f"   ⏱️  Action: Queue data for later (will send in {delay} minutes)")
                    print(f"   📦 Current queue size: {queue_size + 1} items")
                    self.queue.add(sensor_data)
                    
                else:  # skip
                    logging.info(f"⏸️ SKIP sync (prob={prob:.3f})")
                    print(f"   ⏸️  Action: Skip transmission")
                    print(f"   📦 Queue size remains: {queue_size} items")

                # 3. If conditions good, process pending queue
                if prob > 0.5 and queue_size > 0:
                    print(f"   🔄 Good conditions - processing queue...")
                    sent = self.process_pending_queue(limit=3)
                    if sent > 0:
                        print(f"   ✅ Processed {sent} items from queue")
                        
            else:
                logging.warning("Could not build sync feature vector")
                
        except Exception as e:
            logging.error(f"Sync feature construction failed: {e}")

    def run(self, cycle_seconds=300):
        """Main loop"""
        self.running = True
        
        logging.info("="*60)
        logging.info("MAIN CONTROLLER STARTED")
        logging.info("="*60)
        logging.info(f"Cycle interval: {cycle_seconds} seconds")
        logging.info(f"Features: {len(self.sync_features) if self.sync_features else 0} total")
        logging.info("  - From Arduino: temperature, humidity, irradiance, panel_voltage, battery_voltage")
        logging.info("  - Calculated: panel_current, panel_power, battery_soc")
        logging.info("  - System: hour, month, is_daytime")
        logging.info("  - External: rssi, app_usage_hourly")
        logging.info(f"Transmission enabled: {self.transmission_enabled}")
        logging.info("="*60)
        logging.info("⚠️  ONLY REAL ARDUINO DATA - NO SIMULATION ⚠️")
        logging.info("="*60)
        
        while self.running:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.exception("Unhandled exception in cycle")
            time.sleep(cycle_seconds)

    def stop(self):
        """Stop the controller"""
        self.running = False
        if self.serial_port:
            try:
                self.serial_port.close()
            except:
                pass
        
        queue_stats = self.queue.get_stats()
        final_queue_size = queue_stats.get('pending', 0)
        
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"   Total readings:      {self.total_readings}")
        print(f"   Arduino readings:    {self.arduino_readings}")
        print(f"   Final queue size:    {final_queue_size} items")
        print("="*60)
        logging.info("Main controller stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Main Controller - Only real Arduino data')
    parser.add_argument('--test-arduino', action='store_true', help='Test Arduino connection')
    parser.add_argument('--cycle-seconds', type=int, default=60, help='Cycle interval in seconds')
    args = parser.parse_args()
    
    controller = main_controller('config.json')
    
    if args.test_arduino:
        # Test mode - just check Arduino
        if controller.serial_port:
            controller.test_arduino_communication()
        else:
            print("No Arduino connection. Check logs for details.")
    else:
        # Normal run mode - only real Arduino data
        try:
            controller.run(cycle_seconds=args.cycle_seconds)
        except KeyboardInterrupt:
            controller.stop()