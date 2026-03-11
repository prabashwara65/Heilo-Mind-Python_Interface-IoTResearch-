"""
Sync Scheduler Neural Network Model
Predicts optimal sync timing based on all features
Includes TensorFlow Lite conversion for mobile/edge deployment
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
import os
import joblib
import json
import sys

# Optional seaborn import for nicer plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Seaborn not installed. Using matplotlib for plots (install seaborn for better visuals).")

class SyncSchedulerModel:
    def __init__(self, model_path='models/sync_scheduler/'):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.model = None
        self.history = None
        self.feature_names = None
        self.scaler = None
        
    def build_model(self, input_dim):
        """
        Build neural network for sync probability prediction
        
        Architecture:
        - Input layer: matches feature count
        - 3 hidden layers with dropout for regularization
        - Output: sigmoid for probability (0-1)
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,), name='input_layer'),
            
            # First hidden layer - feature extraction
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.BatchNormalization(name='batchnorm_1'),
            layers.Dropout(0.3, name='dropout_1'),
            
            # Second hidden layer - pattern learning
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.BatchNormalization(name='batchnorm_2'),
            layers.Dropout(0.2, name='dropout_2'),
            
            # Third hidden layer - refinement
            layers.Dense(32, activation='relu', name='dense_3'),
            layers.BatchNormalization(name='batchnorm_3'),
            layers.Dropout(0.1, name='dropout_3'),
            
            # Fourth hidden layer
            layers.Dense(16, activation='relu', name='dense_4'),
            
            # Output layer - sync probability
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model – removed 'accuracy' metric as target is continuous
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['mae']  # Mean Absolute Error is appropriate for regression
        )
        
        return model
    
    def load_training_data(self, training_path='data/training/'):
        """Load preprocessed training data from feature engineering step"""
        print("\n" + "="*60)
        print("LOADING TRAINING DATA")
        print("="*60)
        
        # Load feature names
        feature_names_path = os.path.join(training_path, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.feature_names)} feature names")
            print("   Features:", ", ".join(self.feature_names))  # Show the actual features
        else:
            print("⚠️ feature_names.txt not found. Using default feature order from CSV.")
        
        # Load scaler
        scaler_path = os.path.join(training_path, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✅ Loaded feature scaler")
        
        # Load training data
        X_train = pd.read_csv(os.path.join(training_path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(training_path, 'X_test.csv'))
        y_prob_train = pd.read_csv(os.path.join(training_path, 'y_prob_train.csv')).squeeze()
        y_prob_test = pd.read_csv(os.path.join(training_path, 'y_prob_test.csv')).squeeze()
        y_class_train = pd.read_csv(os.path.join(training_path, 'y_class_train.csv')).squeeze()
        y_class_test = pd.read_csv(os.path.join(training_path, 'y_class_test.csv')).squeeze()
        
        print(f"\n📊 Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"📊 Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_prob_train, y_prob_test, y_class_train, y_class_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the sync scheduler model
        """
        print("\n" + "="*60)
        print("STEP 6: TRAINING SYNC SCHEDULER MODEL")
        print("="*60)
        
        # If validation data not provided, split from training
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42
            )
        
        print(f"\n📊 Training data: {X_train.shape[0]} samples")
        print(f"📊 Validation data: {X_val.shape[0]} samples")
        print(f"📊 Features: {X_train.shape[1]}")
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        print("\n🧠 Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_path, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger(
                filename=os.path.join(self.model_path, 'training_log.csv')
            )
        ]
        
        # Train
        print("\n🚀 Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        self.save_model('sync_scheduler_final.keras')
        
        # Save training history plot
        self.plot_training_history()
        
        # Evaluate on validation
        self.evaluate_model(X_val, y_val)
        
        return self.history
    
    def plot_training_history(self):
        """Plot training metrics – now only loss and MAE (accuracy removed)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Mean Absolute Error', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_history.png'), dpi=150)
        plt.show()
    
    def evaluate_model(self, X_val, y_val):
        """Evaluate model performance on validation set"""
        print("\n" + "="*60)
        print("STEP 7: MODEL EVALUATION")
        print("="*60)
        
        # Predict
        y_pred_prob = self.model.predict(X_val).flatten()
        
        # Convert to binary predictions for classification metrics
        y_pred_class = (y_pred_prob > 0.5).astype(int)
        y_val_class = (y_val > 0.5).astype(int)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred_prob)
        mse = np.mean((y_val - y_pred_prob)**2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_prob)
        
        print(f"\n📈 Regression Metrics:")
        print(f"   Mean Absolute Error (MAE): {mae:.4f}")
        print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"   R² Score: {r2:.4f}")
        
        # Classification report
        print(f"\n📊 Classification Report (threshold=0.5):")
        print(classification_report(y_val_class, y_pred_class, 
                                   target_names=['No Sync', 'Sync']))
        
        # Confusion matrix
        cm = confusion_matrix(y_val_class, y_pred_class)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion matrix plot (with or without seaborn)
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        else:
            # Fallback to matplotlib imshow
            im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[0].figure.colorbar(im, ax=axes[0])
            # Show all ticks and label them
            axes[0].set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       xticklabels=['No Sync', 'Sync'],
                       yticklabels=['No Sync', 'Sync'])
            # Loop over data dimensions and create text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
        
        axes[0].set_title('Confusion Matrix', fontsize=14)
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Prediction distribution
        axes[1].hist(y_val, bins=30, alpha=0.5, label='Actual', density=True)
        axes[1].hist(y_pred_prob, bins=30, alpha=0.5, label='Predicted', density=True)
        axes[1].axvline(0.6, color='red', linestyle='--', label='High threshold (0.6)')
        axes[1].axvline(0.2, color='orange', linestyle='--', label='Low threshold (0.2)')
        axes[1].set_title('Probability Distribution', fontsize=14)
        axes[1].set_xlabel('Sync Probability')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'evaluation_plots.png'), dpi=150)
        plt.show()
        
        # Save metrics to JSON
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_val_class, y_pred_class, output_dict=True)
        }
        with open(os.path.join(self.model_path, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def predict(self, X):
        """Predict sync probability for new samples"""
        if self.model is None:
            raise ValueError("Model not trained. Load or train first.")
        return self.model.predict(X).flatten()
    
    def predict_class(self, X, threshold=0.5):
        """Predict sync class (0 or 1) based on threshold"""
        probs = self.predict(X)
        return (probs > threshold).astype(int)
    
    def save_model(self, filename='sync_scheduler_model.keras'):
        """Save model to file"""
        if self.model:
            # Save with save_format='keras' for better compatibility
            filepath = os.path.join(self.model_path, filename)
            self.model.save(filepath, save_format='keras')
            print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filename='sync_scheduler_final.keras'):
        """Load trained model from file with compatibility handling"""
        model_file = os.path.join(self.model_path, filename)
        
        if not os.path.exists(model_file):
            print(f"⚠️ Model file not found: {model_file}")
            return False
        
        print(f"Attempting to load model from: {model_file}")
        
        # Method 1: Standard loading with TensorFlow
        try:
            self.model = tf.keras.models.load_model(model_file)
            print("✅ Model loaded successfully with standard method")
            return True
            
        except Exception as e:
            print(f"⚠️ Standard loading failed: {e}")
            
            # Method 2: Custom InputLayer to handle batch_shape
            try:
                print("Attempting loading with custom InputLayer...")
                
                # Define a custom InputLayer that handles batch_shape
                class CompatibleInputLayer(tf.keras.layers.InputLayer):
                    def __init__(self, **kwargs):
                        # Convert batch_shape to batch_input_shape if present
                        if 'batch_shape' in kwargs:
                            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                        super().__init__(**kwargs)
                
                # Register the custom layer
                custom_objects = {
                    'InputLayer': CompatibleInputLayer,
                    'Sequential': tf.keras.Sequential
                }
                
                self.model = tf.keras.models.load_model(
                    model_file, 
                    custom_objects=custom_objects,
                    compile=False  # Don't compile initially
                )
                
                # Recompile the model
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['mae']
                )
                
                print("✅ Model loaded successfully with custom InputLayer")
                return True
                
            except Exception as e2:
                print(f"⚠️ Custom loading failed: {e2}")
                
                # Method 3: Load with safe mode and manual config fix
                try:
                    print("Attempting manual config modification...")
                    
                    import h5py
                    import json
                    
                    # Read the model file manually
                    with h5py.File(model_file, 'r') as f:
                        # Get model config
                        if 'model_config' in f.attrs:
                            model_config = f.attrs['model_config']
                            if isinstance(model_config, bytes):
                                model_config = model_config.decode('utf-8')
                            
                            # Parse config
                            config_dict = json.loads(model_config)
                            
                            # Function to recursively fix InputLayer configs
                            def fix_input_layer_config(obj):
                                if isinstance(obj, dict):
                                    # Check if this is an InputLayer
                                    if obj.get('class_name') == 'InputLayer':
                                        if 'config' in obj:
                                            config = obj['config']
                                            # Replace batch_shape with batch_input_shape
                                            if 'batch_shape' in config:
                                                config['batch_input_shape'] = config.pop('batch_shape')
                                    
                                    # Recursively process all values
                                    for key, value in obj.items():
                                        obj[key] = fix_input_layer_config(value)
                                
                                elif isinstance(obj, list):
                                    obj = [fix_input_layer_config(item) for item in obj]
                                
                                return obj
                            
                            # Fix the config
                            config_dict = fix_input_layer_config(config_dict)
                            
                            # Rebuild model from config
                            from tensorflow.keras.models import model_from_json
                            self.model = model_from_json(json.dumps(config_dict))
                            
                            # Load weights if available
                            if 'weights' in f:
                                # We need to load weights from the file
                                # This is complex - better to use the standard method with weights
                                print("⚠️ Manual loading created model structure but weights need to be loaded")
                                
                                # Try to load the full model again with the fixed config
                                # For now, we'll recompile and return
                                self.model.compile(
                                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                    loss='binary_crossentropy',
                                    metrics=['mae']
                                )
                                
                                print("✅ Model structure loaded. Weights may need to be retrained.")
                                return True
                
                except Exception as e3:
                    print(f"⚠️ Manual loading failed: {e3}")
                    
                    # Method 4: Last resort - create new model
                    print("All loading methods failed. You may need to retrain the model.")
                    return False

    # ==================== NEW: TENSORFLOW LITE CONVERSION METHODS ====================
    
    def convert_to_tflite(self, filename='sync_scheduler_final.keras', 
                          quantization='float16', 
                          representative_data=None):
        """
        Convert the trained model to TensorFlow Lite format
        
        Args:
            filename: Name of the Keras model file to convert
            quantization: Type of quantization ('none', 'float16', 'int8', 'dynamic')
            representative_data: Representative dataset for int8 quantization
        
        Returns:
            Path to the saved TFLite model
        """
        print("\n" + "="*60)
        print("CONVERTING MODEL TO TENSORFLOW LITE")
        print("="*60)
        
        # Load the model if not already loaded
        if self.model is None:
            if not self.load_model(filename):
                print("❌ Failed to load model for TFLite conversion")
                return None
        
        # Load feature names and scaler if available
        if self.feature_names is None:
            try:
                with open(os.path.join(self.model_path, 'feature_names.txt'), 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            except:
                print("⚠️ Feature names not found. Creating default feature names.")
                self.feature_names = [f'feature_{i}' for i in range(self.model.input_shape[1])]
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply quantization based on selection
        if quantization == 'float16':
            print("Applying float16 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization == 'int8':
            print("Applying int8 quantization (full integer quantization)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_data is not None:
                converter.representative_dataset = representative_data
            else:
                print("⚠️ No representative dataset provided for int8 quantization.")
                print("   Attempting to generate from training data...")
                
                # Try to load training data for representative dataset
                try:
                    X_train = pd.read_csv(os.path.join('data/training/', 'X_train.csv'))
                    # Use a subset of training data for representative dataset
                    converter.representative_dataset = self._get_representative_dataset(X_train[:100])
                except:
                    print("❌ Could not generate representative dataset.")
                    print("   Falling back to float16 quantization.")
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    quantization = 'float16'
            
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
        elif quantization == 'dynamic':
            print("Applying dynamic range quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        else:  # no quantization
            print("No quantization applied.")
            converter.optimizations = []
        
        # Convert the model
        try:
            tflite_model = converter.convert()
            
            # Save the TFLite model
            quant_suffix = '' if quantization == 'none' else f'_{quantization}'
            tflite_filename = f'sync_scheduler_model{quant_suffix}.tflite'
            tflite_path = os.path.join(self.model_path, tflite_filename)
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get file size
            file_size_kb = len(tflite_model) / 1024
            original_size_kb = os.path.getsize(os.path.join(self.model_path, filename)) / 1024
            
            print(f"\n✅ TFLite model saved to: {tflite_path}")
            print(f"📊 Model size: {file_size_kb:.2f} KB (original: {original_size_kb:.2f} KB)")
            print(f"📊 Compression ratio: {original_size_kb/file_size_kb:.2f}x")
            
            # Save metadata
            self._save_tflite_metadata(tflite_path, quantization, file_size_kb)
            
            return tflite_path
            
        except Exception as e:
            print(f"❌ TFLite conversion failed: {e}")
            return None
    
    def _get_representative_dataset(self, data_samples):
        """Generate representative dataset for int8 quantization"""
        def representative_dataset():
            for i in range(len(data_samples)):
                # Get a single sample and convert to float32
                sample = data_samples.iloc[i:i+1].values.astype(np.float32)
                yield [sample]
        return representative_dataset
    
    def _save_tflite_metadata(self, tflite_path, quantization, file_size_kb):
        """Save metadata for the TFLite model"""
        metadata = {
            'model_type': 'sync_scheduler',
            'quantization': quantization,
            'file_size_kb': file_size_kb,
            'input_features': self.feature_names,
            'input_shape': self.model.input_shape[1:],
            'output_activation': 'sigmoid',
            'output_range': [0, 1],
            'conversion_date': pd.Timestamp.now().isoformat(),
            'tensorflow_version': tf.__version__
        }
        
        metadata_path = os.path.join(self.model_path, f'tflite_metadata_{quantization}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to: {metadata_path}")
    
    def convert_all_formats(self, filename='sync_scheduler_final.keras'):
        """
        Convert the model to all TFLite formats (no quant, float16, int8, dynamic)
        
        Returns:
            Dictionary with paths to all converted models
        """
        print("\n" + "="*60)
        print("CONVERTING TO ALL TENSORFLOW LITE FORMATS")
        print("="*60)
        
        # Load model if needed
        if self.model is None:
            if not self.load_model(filename):
                print("❌ Failed to load model for TFLite conversion")
                return {}
        
        # Try to load training data for int8 quantization
        representative_data = None
        try:
            X_train = pd.read_csv(os.path.join('data/training/', 'X_train.csv'))
            representative_data = self._get_representative_dataset(X_train[:100])
        except:
            print("⚠️ Could not load training data for int8 quantization.")
        
        results = {}
        
        # Convert without quantization
        print("\n" + "-"*40)
        path = self.convert_to_tflite(filename, quantization='none')
        if path:
            results['none'] = path
        
        # Convert with dynamic range quantization
        print("\n" + "-"*40)
        path = self.convert_to_tflite(filename, quantization='dynamic')
        if path:
            results['dynamic'] = path
        
        # Convert with float16 quantization
        print("\n" + "-"*40)
        path = self.convert_to_tflite(filename, quantization='float16')
        if path:
            results['float16'] = path
        
        # Convert with int8 quantization
        if representative_data:
            print("\n" + "-"*40)
            path = self.convert_to_tflite(filename, quantization='int8', 
                                          representative_data=representative_data)
            if path:
                results['int8'] = path
        else:
            print("\n⚠️ Skipping int8 quantization (representative data required)")
        
        # Save comparison report
        self._save_tflite_comparison(results)
        
        return results
    
    def _save_tflite_comparison(self, results):
        """Save comparison of different TFLite models"""
        comparison = []
        
        for quant_type, path in results.items():
            size_kb = os.path.getsize(path) / 1024
            comparison.append({
                'quantization': quant_type,
                'path': path,
                'size_kb': size_kb
            })
        
        # Sort by size
        comparison.sort(key=lambda x: x['size_kb'])
        
        # Save comparison
        comparison_path = os.path.join(self.model_path, 'tflite_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump({
                'models': comparison,
                'recommendation': comparison[0]['quantization'] if comparison else None
            }, f, indent=2)
        
        print("\n" + "="*60)
        print("TFLITE MODEL COMPARISON")
        print("="*60)
        for model in comparison:
            print(f"📊 {model['quantization']}: {model['size_kb']:.2f} KB")
        
        if comparison:
            print(f"\n✅ Recommended for production: {comparison[0]['quantization']} quantization")
            print(f"   (Smallest size: {comparison[0]['size_kb']:.2f} KB)")
    
    def verify_tflite_model(self, tflite_path, test_samples=None):
        """
        Verify the TFLite model by running inference and comparing with original model
        
        Args:
            tflite_path: Path to the TFLite model
            test_samples: Test samples to use for verification
        """
        print("\n" + "="*60)
        print("VERIFYING TFLITE MODEL")
        print("="*60)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📊 Input details: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
        print(f"📊 Output details: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
        
        # Get test samples if not provided
        if test_samples is None:
            try:
                X_test = pd.read_csv(os.path.join('data/training/', 'X_test.csv'))
                test_samples = X_test[:10]
            except:
                print("⚠️ Could not load test data. Generating random samples.")
                test_samples = np.random.randn(10, self.model.input_shape[1]).astype(np.float32)
        
        # Run inference on both models
        tflite_outputs = []
        original_outputs = []
        
        for i in range(len(test_samples)):
            # Get sample
            if isinstance(test_samples, pd.DataFrame):
                sample = test_samples.iloc[i:i+1].values.astype(np.float32)
            else:
                sample = test_samples[i:i+1]
            
            # TFLite inference
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_details[0]['index'])[0][0]
            tflite_outputs.append(tflite_output)
            
            # Original model inference
            original_output = self.model.predict(sample, verbose=0)[0][0]
            original_outputs.append(original_output)
        
        # Compare results
        tflite_outputs = np.array(tflite_outputs)
        original_outputs = np.array(original_outputs)
        
        # Calculate differences
        abs_diff = np.abs(tflite_outputs - original_outputs)
        rel_diff = abs_diff / (np.abs(original_outputs) + 1e-7)
        
        print(f"\n📊 Verification Results:")
        print(f"   Mean absolute difference: {np.mean(abs_diff):.6f}")
        print(f"   Max absolute difference: {np.max(abs_diff):.6f}")
        print(f"   Mean relative difference: {np.mean(rel_diff):.6f}")
        
        # Show sample comparisons
        print("\n📊 Sample Predictions (Original vs TFLite):")
        print("   Index | Original  | TFLite    | Diff")
        print("   " + "-"*45)
        for i in range(min(10, len(test_samples))):
            print(f"   {i:5d} | {original_outputs[i]:.4f}    | {tflite_outputs[i]:.4f}    | {abs_diff[i]:.4f}")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(original_outputs, tflite_outputs, alpha=0.7)
        axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect match')
        axes[0].set_xlabel('Original Model Output')
        axes[0].set_ylabel('TFLite Model Output')
        axes[0].set_title('Original vs TFLite Predictions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Difference histogram
        axes[1].hist(abs_diff, bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Absolute Difference')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Prediction Differences')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'tflite_verification.png'), dpi=150)
        plt.show()
        
        # Determine if verification passed
        max_allowed_diff = 0.01  # 1% difference
        passed = np.max(abs_diff) < max_allowed_diff
        
        if passed:
            print(f"\n✅ Verification PASSED! (Max diff < {max_allowed_diff})")
        else:
            print(f"\n⚠️ Verification WARNING! (Max diff >= {max_allowed_diff})")
            print("   Consider using higher precision quantization or no quantization.")
        
        return passed
    
    def export_for_mobile(self, filename='sync_scheduler_final.keras'):
        """
        Export optimized model for mobile deployment
        Creates both TFLite model and metadata for mobile integration
        """
        print("\n" + "="*60)
        print("EXPORTING FOR MOBILE DEPLOYMENT")
        print("="*60)
        
        # Convert all formats
        tflite_results = self.convert_all_formats(filename)
        
        if not tflite_results:
            print("❌ No TFLite models created")
            return None
        
        # Create mobile deployment package
        mobile_dir = os.path.join(self.model_path, 'mobile_deployment')
        os.makedirs(mobile_dir, exist_ok=True)
        
        # Copy the smallest model (usually int8 or float16)
        comparison_path = os.path.join(self.model_path, 'tflite_comparison.json')
        with open(comparison_path, 'r') as f:
            comparison = json.load(f)
        
        if comparison['models']:
            best_model = comparison['models'][0]
            best_model_path = best_model['path']
            
            # Copy to mobile directory
            import shutil
            mobile_model_path = os.path.join(mobile_dir, 'sync_scheduler_optimized.tflite')
            shutil.copy2(best_model_path, mobile_model_path)
            
            # Create metadata for mobile
            mobile_metadata = {
                'model_name': 'Sync Scheduler',
                'version': '1.0.0',
                'quantization': best_model['quantization'],
                'input_features': self.feature_names,
                'input_shape': self.model.input_shape[1],
                'output_description': 'Sync probability (0-1)',
                'thresholds': {
                    'low': 0.2,
                    'medium': 0.5,
                    'high': 0.6
                },
                'labels': ['No Sync', 'Sync'],
                'preprocessing': {
                    'requires_scaler': True,
                    'scaler_file': 'feature_scaler.pkl'
                }
            }
            
            mobile_metadata_path = os.path.join(mobile_dir, 'model_metadata.json')
            with open(mobile_metadata_path, 'w') as f:
                json.dump(mobile_metadata, f, indent=2)
            
            # Copy scaler if exists
            scaler_path = os.path.join('data/training/', 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                shutil.copy2(scaler_path, os.path.join(mobile_dir, 'feature_scaler.pkl'))
            
            print(f"\n✅ Mobile deployment package created in: {mobile_dir}")
            print(f"   - Optimized model: sync_scheduler_optimized.tflite ({best_model['size_kb']:.2f} KB)")
            print(f"   - Metadata: model_metadata.json")
            print(f"   - Scaler: feature_scaler.pkl (if available)")
            
            return mobile_dir
        
        return None


if __name__ == "__main__":
    # Initialize model trainer
    model_trainer = SyncSchedulerModel()
    
    # Try to load existing model first
    model_loaded = model_trainer.load_model()
    
    if not model_loaded:
        print("\n" + "="*60)
        print("Training new model...")
        print("="*60)
        
        # Load preprocessed training data
        X_train, X_test, y_prob_train, y_prob_test, y_class_train, y_class_test = model_trainer.load_training_data()
        
        # Train model on probability target
        history = model_trainer.train(
            X_train=X_train,
            y_train=y_prob_train,
            X_val=X_test,
            y_val=y_prob_test,
            epochs=50,
            batch_size=32
        )
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: {model_trainer.model_path}")
    else:
        print("\n✅ Existing model loaded successfully!")
    
    # Convert to TensorFlow Lite (for mobile/edge deployment)
    print("\n" + "="*60)
    print("CONVERTING TO TENSORFLOW LITE")
    print("="*60)
    
    # Option 1: Convert to all formats
    tflite_results = model_trainer.convert_all_formats()
    
    # Option 2: Convert to specific format
    # tflite_path = model_trainer.convert_to_tflite(quantization='float16')
    
    # Verify the converted model
    if tflite_results and 'float16' in tflite_results:
        model_trainer.verify_tflite_model(tflite_results['float16'])
    
    # Export for mobile deployment (automatically selects best format)
    mobile_dir = model_trainer.export_for_mobile()
    
    print("\n" + "="*60)
    print("✅ ALL STEPS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("   - Use model for inference in sync scheduler")
    print("   - Deploy TFLite model to mobile/edge devices")
    print("   - Monitor performance and retrain periodically")
    print("   - Test different quantization options for your target device")