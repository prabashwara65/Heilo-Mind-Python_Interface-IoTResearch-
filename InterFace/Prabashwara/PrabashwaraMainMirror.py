#!/usr/bin/env python3
"""
Mirror Control Main Program - Waits for MQTT requests
Runs redirect_algorithm.py ONLY when run_prediction command is received
"""

import os
import json
import time
import threading
import queue
import numpy as np
import cv2
import serial
import serial.tools.list_ports
import boto3
import ssl
import signal
import sys
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
from decimal import Decimal
from collections import deque
from threading import Thread
from queue import Queue, Empty

# AWS IoT Core SDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================
# CONFIGURATION
# =============================
class Config:
    """Central configuration class"""
    
    # AWS IoT Core Settings
    SERVER_CLIENT_ID = "mirror_control_server"
    AWS_IOT_ENDPOINT = "ajmja1mzmi1j4-ats.iot.eu-north-1.amazonaws.com"
    REQUEST_TOPIC = "mirrorAngle/prediction/request"  # Only listening to this topic
    RESPONSE_TOPIC = "mirror/prediction/result"
    
    # DynamoDB
    TABLE_NAME = "PrabhashwaraMirrorResults"
    REGION = "eu-north-1"
    ENABLE_DYNAMODB = True
    
    # Paths
    ROOT_DIR = Path(__file__).resolve().parent
    CERTS_DIR = ROOT_DIR / "certs"
    
    # Redirect algorithm script
    REDIRECT_SCRIPT = ROOT_DIR / "redirect_algorithm.py"
    
    # Certificate paths
    SERVER_ROOT_CA = CERTS_DIR / "AmazonRootCA1.pem"
    SERVER_CERT = CERTS_DIR / "certificate.pem.crt"
    SERVER_PRIVATE_KEY = CERTS_DIR / "private.pem.key"
    
    # Threading
    NUM_WORKER_THREADS = 2
    REQUEST_QUEUE_SIZE = 50
    
    DEBUG = True

# =============================
# LOGGING SETUP
#=============================
logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mirror_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================
# DYNAMODB HELPER
# =============================
def convert_floats_to_decimal(obj):
    """Recursively convert floats to Decimal for DynamoDB compatibility"""
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
# REDIRECT ALGORITHM RUNNER
# =============================
class RedirectRunner:
    """Runs the redirect_algorithm.py script when requested"""
    
    def __init__(self):
        self.process = None
        self.running = False
        self.process_lock = threading.Lock()
        
    def check_script_exists(self) -> bool:
        """Check if the redirect script exists"""
        exists = Config.REDIRECT_SCRIPT.exists()
        logger.debug(f"Redirect script exists: {exists} at {Config.REDIRECT_SCRIPT}")
        return exists
    
    def get_script_path(self) -> str:
        """Get the script path as string"""
        return str(Config.REDIRECT_SCRIPT)
    
    def run_redirect(self) -> Dict[str, Any]:
        """
        Run the redirect_algorithm.py script
        Returns dictionary with results
        """
        logger.info("=" * 50)
        logger.info("☀️ RUNNING REDIRECT ALGORITHM")
        logger.info("=" * 50)
        
        # Check if already running
        with self.process_lock:
            if self.running and self.process:
                logger.warning("⚠️ Redirect already running!")
                return {
                    "success": False,
                    "error": "Redirect already in progress",
                    "already_running": True
                }
        
        if not self.check_script_exists():
            error_msg = f"Redirect script not found at: {self.get_script_path()}"
            logger.error(f"❌ {error_msg}")
            
            # List all Python files in directory for debugging
            try:
                py_files = list(Config.ROOT_DIR.glob("*.py"))
                logger.info(f"📂 Available Python files: {[f.name for f in py_files]}")
            except:
                pass
                
            return {
                "success": False,
                "error": error_msg,
                "script_path": self.get_script_path()
            }
        
        logger.info(f"📄 Script path: {self.get_script_path()}")
        
        try:
            # Make the script executable
            os.chmod(self.get_script_path(), 0o755)
            logger.debug("Set script executable permissions")
            
            # Run the script and capture output
            start_time = time.time()
            logger.info("🚀 Launching redirect algorithm...")
            
            with self.process_lock:
                self.process = subprocess.Popen(
                    [sys.executable, self.get_script_path()],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(Config.ROOT_DIR)
                )
                self.running = True
                pid = self.process.pid
            
            logger.info(f"Process started with PID: {pid}")
            
            # Wait for completion
            stdout, stderr = self.process.communicate(timeout=300)  # 5 minute timeout
            execution_time = time.time() - start_time
            
            with self.process_lock:
                self.running = False
                self.process = None
            
            # Log the output
            logger.info(f"📤 Script stdout: {stdout[:500]}..." if len(stdout) > 500 else f"📤 Script stdout: {stdout}")
            if stderr:
                logger.warning(f"⚠️ Script stderr: {stderr}")
            
            if self.process.returncode == 0:
                logger.info(f"✅ Redirect completed successfully in {execution_time:.2f} seconds")
                
                # Parse output for key information
                lines = stdout.strip().split('\n')
                
                return {
                    "success": True,
                    "execution_time_seconds": round(execution_time, 2),
                    "return_code": self.process.returncode,
                    "output": lines[-10:],  # Last 10 lines
                    "full_output": stdout,
                    "script_path": self.get_script_path()
                }
            else:
                logger.error(f"❌ Redirect failed with code {self.process.returncode}")
                return {
                    "success": False,
                    "error": stderr or "Unknown error",
                    "return_code": self.process.returncode,
                    "execution_time_seconds": round(execution_time, 2),
                    "script_path": self.get_script_path()
                }
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Redirect timed out after 5 minutes")
            with self.process_lock:
                if self.process:
                    self.process.kill()
                    self.process = None
                self.running = False
            return {
                "success": False,
                "error": "Timeout after 5 minutes",
                "timeout": True,
                "script_path": self.get_script_path()
            }
            
        except Exception as e:
            logger.error(f"❌ Error running redirect: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            with self.process_lock:
                self.running = False
                self.process = None
            
            return {
                "success": False,
                "error": str(e),
                "script_path": self.get_script_path()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current redirect status"""
        with self.process_lock:
            status = {
                "running": self.running,
                "process_active": self.process is not None,
                "pid": self.process.pid if self.process else None
            }
        return status

# =============================
# MQTT SERVER - WAITS FOR REQUESTS
# =============================
class MirrorControlServer:
    """Handles incoming mirror control requests via AWS IoT Core"""
    
    def __init__(self, redirect_runner: RedirectRunner):
        self.redirect_runner = redirect_runner
        self.mqtt_client = None
        self.running = False
        self.request_queue = queue.Queue(maxsize=Config.REQUEST_QUEUE_SIZE)
        self.worker_threads = []
        self.dynamodb = None
        self.table = None
        
        # Request tracking
        self.request_count = 0
        self.request_lock = threading.Lock()
        
        # Initialize DynamoDB
        if Config.ENABLE_DYNAMODB:
            try:
                self.dynamodb = boto3.resource("dynamodb", region_name=Config.REGION)
                self._ensure_table_exists()
            except Exception as e:
                logger.warning(f"⚠️ DynamoDB initialization failed: {e}")
                self.dynamodb = None
        else:
            logger.info("ℹ️ DynamoDB logging is disabled")
    
    def _ensure_table_exists(self):
        """Ensure DynamoDB table exists"""
        if not self.dynamodb:
            return False
        
        try:
            existing_tables = self.dynamodb.meta.client.list_tables()['TableNames']
            if Config.TABLE_NAME not in existing_tables:
                logger.info(f"Creating DynamoDB table: {Config.TABLE_NAME}")
                
                table = self.dynamodb.create_table(
                    TableName=Config.TABLE_NAME,
                    KeySchema=[{'AttributeName': 'requestId', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': 'requestId', 'AttributeType': 'S'}],
                    BillingMode='PAY_PER_REQUEST'
                )
                
                table.wait_until_exists()
                logger.info(f"✅ Table {Config.TABLE_NAME} created successfully")
            
            self.table = self.dynamodb.Table(Config.TABLE_NAME)
            logger.info("✅ DynamoDB initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ DynamoDB setup failed: {e}")
            return False
    
    def connect(self):
        """Connect to AWS IoT Core"""
        try:
            logger.info("=" * 50)
            logger.info("📡 CONNECTING TO AWS IOT CORE")
            logger.info("=" * 50)
            
            # Verify certificate files exist
            logger.info("🔍 Checking certificate files...")
            cert_files = [
                ("Root CA", Config.SERVER_ROOT_CA),
                ("Certificate", Config.SERVER_CERT),
                ("Private Key", Config.SERVER_PRIVATE_KEY)
            ]
            
            for name, path in cert_files:
                if not path.exists():
                    logger.error(f"❌ {name} file NOT FOUND: {path}")
                    return False
                else:
                    logger.info(f"✅ {name} found: {path}")
            
            logger.info(f"Creating MQTT client with ID: {Config.SERVER_CLIENT_ID}")
            self.mqtt_client = AWSIoTMQTTClient(Config.SERVER_CLIENT_ID)
            
            logger.info(f"Configuring endpoint: {Config.AWS_IOT_ENDPOINT}:8883")
            self.mqtt_client.configureEndpoint(Config.AWS_IOT_ENDPOINT, 8883)
            
            logger.info("Configuring credentials...")
            self.mqtt_client.configureCredentials(
                str(Config.SERVER_ROOT_CA),
                str(Config.SERVER_PRIVATE_KEY),
                str(Config.SERVER_CERT)
            )
            
            # Configure connection settings
            logger.info("Configuring connection settings...")
            
            try:
                self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
                self.mqtt_client.configureOfflinePublishQueueing(-1)
                self.mqtt_client.configureDrainingFrequency(2)
                self.mqtt_client.configureConnectDisconnectTimeout(10)
                self.mqtt_client.configureMQTTOperationTimeout(5)
            except AttributeError:
                logger.debug("Some configuration methods not available - continuing...")
            
            logger.info("Connecting to AWS IoT Core...")
            connection_result = self.mqtt_client.connect()
            
            if connection_result:
                logger.info("✅ SUCCESS: Connected to AWS IoT Core")
                
                logger.info(f"Subscribing to topic: {Config.REQUEST_TOPIC}")
                subscribe_result = self.mqtt_client.subscribe(Config.REQUEST_TOPIC, 1, self._message_callback)
                
                if subscribe_result:
                    logger.info(f"✅ SUCCESS: Subscribed to topic: {Config.REQUEST_TOPIC}")
                    logger.info("📡 Waiting for run_prediction commands...")
                else:
                    logger.error(f"❌ FAILED: Could not subscribe to topic: {Config.REQUEST_TOPIC}")
                    return False
                
                return True
            else:
                logger.error("❌ FAILED: Could not connect to AWS IoT Core")
                return False
            
        except Exception as e:
            logger.error(f"❌ Exception during connection: {e}")
            return False
    
    def _message_callback(self, client, userdata, message):
        """Callback for incoming MQTT messages - THIS IS TRIGGERED ONLY WHEN A MESSAGE IS RECEIVED"""
        try:
            print("\n" + "="*70)
            print("🔔 MESSAGE RECEIVED! 🔔")
            print("="*70)
            
            with self.request_lock:
                self.request_count += 1
                current_count = self.request_count
            
            logger.info("=" * 70)
            logger.info(f"📩 REQUEST RECEIVED #{current_count}")
            logger.info("=" * 70)
            logger.info(f"📨 Topic: {message.topic}")
            logger.info(f"📏 Payload size: {len(message.payload)} bytes")
            
            print(f"📨 Topic: {message.topic}")
            
            # Parse payload
            try:
                payload_str = message.payload.decode('utf-8')
                logger.info(f"📄 Payload: {payload_str}")
                print(f"📄 Payload: {payload_str}")
                
                payload = json.loads(payload_str)
                
                request_id = payload.get('requestId', f"auto_{int(time.time()*1000)}")
                device_id = payload.get('deviceId', 'unknown')
                command = payload.get('command', '').lower()
                
                logger.info(f"🔑 Request ID: {request_id}")
                logger.info(f"📱 Device ID: {device_id}")
                logger.info(f"🎮 Command: {command}")
                
                print(f"🔑 Request ID: {request_id}")
                print(f"🎮 Command: {command}")
                
            except Exception as e:
                logger.error(f"❌ Error parsing payload: {e}")
                return
            
            # Add to queue for processing
            try:
                queue_item = {
                    'topic': message.topic,
                    'payload': payload,
                    'request_id': request_id,
                    'device_id': device_id,
                    'command': command,
                    'timestamp': time.time()
                }
                
                self.request_queue.put(queue_item, block=False)
                logger.info(f"📥 Added to queue (size: {self.request_queue.qsize()})")
                
            except queue.Full:
                logger.error("❌ Queue full - dropping request")
            
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Error in callback: {e}")
    
    def _process_request(self, request: Dict[str, Any]):
        """Process a single request - runs ONLY when a message is received"""
        start_time = time.time()
        
        try:
            request_id = request['request_id']
            device_id = request['device_id']
            command = request['command']
            topic = request['topic']
            
            logger.info("=" * 50)
            logger.info(f"🔄 PROCESSING: {request_id}")
            logger.info("=" * 50)
            logger.info(f"📱 Device: {device_id}")
            logger.info(f"🎮 Command: {command}")
            
            print(f"\n🔄 Processing: {command}")
            
            result = {}
            
            # ONLY RUN WHEN COMMAND IS run_prediction
            if command == 'run_prediction':
                logger.info("✅ run_prediction received - running redirect_algorithm.py")
                print("☀️ Running redirect algorithm...")
                
                # Run the redirect algorithm
                redirect_result = self.redirect_runner.run_redirect()
                
                result = {
                    "command": "run_prediction",
                    "action": "redirect_algorithm_executed",
                    "redirect_result": redirect_result,
                    "script_path": self.redirect_runner.get_script_path(),
                    "script_exists": self.redirect_runner.check_script_exists()
                }
                
            else:
                # Ignore other commands
                logger.info(f"⏭️ Ignoring command: {command} (only run_prediction is processed)")
                print(f"⏭️ Ignoring {command} - waiting for run_prediction")
                
                result = {
                    "command": command,
                    "status": "ignored",
                    "message": "Only run_prediction commands are processed"
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "requestId": request_id,
                "deviceId": device_id,
                "topic": topic,
                "timestamp": time.time(),
                "processing_time_seconds": round(processing_time, 3),
                "status": "success",
                "result": result
            }
            
            logger.info(f"⏱️ Processing time: {processing_time:.3f}s")
            print(f"✅ Request completed in {processing_time:.3f}s")
            
            # Publish response
            response_json = json.dumps(response)
            
            logger.info(f"Publishing to {Config.RESPONSE_TOPIC}...")
            self.mqtt_client.publish(Config.RESPONSE_TOPIC, response_json, 1)
            
            # Log to DynamoDB
            if self.dynamodb and self.table:
                try:
                    response['timestamp_str'] = datetime.fromtimestamp(response['timestamp']).isoformat()
                    response['datetime'] = datetime.fromtimestamp(response['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    
                    dynamodb_item = convert_floats_to_decimal(response)
                    self.table.put_item(Item=dynamodb_item)
                    logger.info("✅ Logged to DynamoDB")
                    
                except Exception as e:
                    logger.error(f"❌ DynamoDB logging failed: {e}")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _worker_thread(self, thread_id: int):
        """Worker thread for processing requests"""
        logger.info(f"🧵 Worker {thread_id} started")
        
        while self.running:
            try:
                # Wait for requests from queue
                request = self.request_queue.get(timeout=1)
                logger.info(f"🧵 Worker {thread_id} processing {request.get('request_id', 'unknown')}")
                self._process_request(request)
                self.request_queue.task_done()
                
            except queue.Empty:
                # No requests - just continue waiting
                continue
            except Exception as e:
                logger.error(f"❌ Worker {thread_id} error: {e}")
        
        logger.info(f"🧵 Worker {thread_id} stopped")
    
    def start(self):
        """Start the server"""
        logger.info("=" * 60)
        logger.info("🚀 STARTING MIRROR CONTROL SERVER")
        logger.info("=" * 60)
        
        if not self.connect():
            logger.error("❌ Failed to start server")
            return False
        
        self.running = True
        
        # Start worker threads
        for i in range(Config.NUM_WORKER_THREADS):
            thread = Thread(target=self._worker_thread, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
            logger.info(f"✅ Started worker {i}")
        
        logger.info(f"📡 Listening on: {Config.REQUEST_TOPIC}")
        logger.info(f"📤 Publishing to: {Config.RESPONSE_TOPIC}")
        logger.info("=" * 60)
        logger.info("⏳ System is waiting for MQTT requests...")
        logger.info("   Will run redirect_algorithm.py when 'run_prediction' is received")
        logger.info("=" * 60)
        
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
        
        logger.info(f"📊 Total requests processed: {self.request_count}")
        logger.info("✅ Server stopped")

# =============================
# MAIN APPLICATION
# =============================
class MirrorControlApp:
    """Main application - waits for MQTT requests"""
    
    def __init__(self):
        self.redirect_runner = RedirectRunner()
        self.server = None
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("🪞 MIRROR CONTROL SYSTEM - INITIALIZING")
        logger.info("=" * 60)
        
        # Check if redirect script exists
        if self.redirect_runner.check_script_exists():
            logger.info(f"✅ Redirect script found: {self.redirect_runner.get_script_path()}")
        else:
            logger.warning(f"⚠️ Redirect script NOT found: {self.redirect_runner.get_script_path()}")
            # List Python files for debugging
            try:
                py_files = list(Config.ROOT_DIR.glob("*.py"))
                logger.info(f"📂 Available: {[f.name for f in py_files]}")
            except:
                pass
        
        # Initialize server
        self.server = MirrorControlServer(self.redirect_runner)
        
        logger.info("=" * 60)
        logger.info("✅ SYSTEM INITIALIZED")
        logger.info("=" * 60)
        
        return True
    
    def start(self):
        """Start all components"""
        logger.info("🚀 STARTING SYSTEM")
        
        if not self.server.start():
            logger.error("❌ Failed to start server")
            return
        
        self.running = True
        
        # Print info
        logger.info("=" * 60)
        logger.info("📊 SYSTEM READY - WAITING FOR REQUESTS")
        logger.info("=" * 60)
        logger.info(f"📡 Listening on: {Config.REQUEST_TOPIC}")
        logger.info(f"🔍 Will run redirect_algorithm.py when 'run_prediction' is received")
        logger.info(f"📤 Responses on: {Config.RESPONSE_TOPIC}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to shutdown")
        logger.info("=" * 60)
        
        # Main loop - just keep the program alive while waiting for requests
        try:
            while self.running:
                time.sleep(1)  # Sleep, don't do anything - just wait for requests
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()
    
    def stop(self):
        """Stop all components"""
        logger.info("🛑 SHUTTING DOWN")
        
        self.running = False
        
        if self.server:
            self.server.stop()
        
        logger.info("👋 SHUTDOWN COMPLETE")

# =============================
# MAIN ENTRY POINT
# =============================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mirror Control System - Waits for MQTT requests')
    parser.add_argument('--no-dynamodb', action='store_true', help='Disable DynamoDB logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    if args.no_dynamodb:
        Config.ENABLE_DYNAMODB = False
        logger.info("ℹ️ DynamoDB logging disabled")
    
    if args.debug:
        Config.DEBUG = True
        logger.setLevel(logging.DEBUG)
        logger.info("🔧 Debug mode enabled")
    
    # Run main application
    app = MirrorControlApp()
    
    if app.initialize():
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            app.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        app.start()
    else:
        logger.error("❌ Failed to initialize")
        sys.exit(1)

if __name__ == "__main__":
    main()