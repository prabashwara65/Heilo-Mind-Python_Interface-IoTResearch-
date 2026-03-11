"""
Anomaly Detector using Autoencoder
Detects unusual patterns in solar panel, battery, and signal data
Includes TensorFlow Lite conversion for mobile/edge deployment
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib
import json

# Optional seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class AnomalyDetector:
    def __init__(self, model_path='models/anomaly_detector/'):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.autoencoder = None
        self.encoder = None
        self.scaler = None
        self.threshold = None
        self.feature_names = None

    def build_autoencoder(self, input_dim, encoding_dim=8):
        """
        Build autoencoder: input -> encoder -> latent -> decoder -> output
        """
        input_layer = layers.Input(shape=(input_dim,))

        # Encoder
        encoded = layers.Dense(16, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = models.Model(input_layer, decoded)
        encoder = models.Model(input_layer, encoded)

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.autoencoder = autoencoder
        self.encoder = encoder
        return autoencoder

    def prepare_anomaly_data(self, combined_df, feature_columns, normal_ratio=0.7):
        """
        Prepare data for anomaly detection.
        Assumes first `normal_ratio` of data is normal (no anomalies).
        Splits into train (normal) and test (rest, possibly containing anomalies).
        """
        print("\n" + "="*60)
        print("PREPARING ANOMALY DETECTION DATA")
        print("="*60)

        X = combined_df[feature_columns].copy()
        X = X.fillna(X.mean())
        self.feature_names = feature_columns

        # Standardise (important for autoencoder)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Assume first `normal_ratio` is normal for training
        split_idx = int(normal_ratio * len(X_scaled))
        X_train = X_scaled[:split_idx]   # normal data only
        X_test = X_scaled[split_idx:]    # may contain anomalies

        print(f"✅ Training data (normal only): {X_train.shape[0]} samples")
        print(f"✅ Test data (may contain anomalies): {X_test.shape[0]} samples")
        print(f"✅ Features: {X_train.shape[1]}")

        return X_train, X_test, X_scaled

    def train(self, X_train, X_val=None, epochs=100, batch_size=32, validation_split=0.1):
        """
        Train autoencoder on normal data.
        """
        if X_val is None:
            # Use part of training as validation
            split = int((1 - validation_split) * len(X_train))
            X_train, X_val = X_train[:split], X_train[split:]

        self.build_autoencoder(X_train.shape[1])

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
                filepath=os.path.join(self.model_path, 'best_autoencoder.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        history = self.autoencoder.fit(
            X_train, X_train,  # autoencoder: input = output
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        # Calculate reconstruction error on training data to set threshold
        train_pred = self.autoencoder.predict(X_train)
        train_errors = np.mean(np.square(X_train - train_pred), axis=1)

        # Set threshold at 95th percentile (5% false positive rate)
        self.threshold = np.percentile(train_errors, 95)
        print(f"\n✅ Anomaly threshold set to: {self.threshold:.4f} (95th percentile)")

        self.plot_training_history(history)
        self.plot_error_distribution(train_errors)
        
        # Save final model
        self.save_model()

        return history

    def plot_training_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Autoencoder Loss')
        axes[0].legend()

        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_title('Autoencoder MAE')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_history.png'))
        plt.show()

    def plot_error_distribution(self, errors):
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=50, alpha=0.7, label='Reconstruction Errors')
        plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold:.4f})')
        plt.title('Reconstruction Error Distribution (Training Data)')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.model_path, 'error_distribution.png'))
        plt.show()

    def predict_anomaly(self, X):
        """
        Returns:
            anomalies: boolean array (True = anomaly)
            mse: reconstruction error per sample
        """
        if self.autoencoder is None:
            raise ValueError("Model not trained. Call train() first.")

        reconstructions = self.autoencoder.predict(X)
        mse = np.mean(np.square(X - reconstructions), axis=1)
        anomalies = mse > self.threshold
        return anomalies, mse

    def evaluate(self, X_test, y_true_anomaly=None):
        """
        Evaluate on test data.
        If y_true_anomaly is provided (ground truth), compute classification metrics.
        Otherwise, just show anomaly statistics.
        """
        anomalies, errors = self.predict_anomaly(X_test)
        print(f"\n📊 Detected {np.sum(anomalies)} anomalies out of {len(X_test)} test samples ({np.mean(anomalies)*100:.1f}%)")

        if y_true_anomaly is not None:
            # Compute metrics
            cm = confusion_matrix(y_true_anomaly, anomalies)
            print("\nClassification Report:")
            print(classification_report(y_true_anomaly, anomalies, target_names=['Normal', 'Anomaly']))

            # Plot confusion matrix
            plt.figure(figsize=(6,5))
            if HAS_SEABORN:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            else:
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.colorbar()
                plt.xticks([0,1], ['Normal', 'Anomaly'])
                plt.yticks([0,1], ['Normal', 'Anomaly'])
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, cm[i, j], ha='center', va='center',
                                 color='white' if cm[i, j] > cm.max()/2 else 'black')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_path, 'confusion_matrix.png'))
            plt.show()

        return anomalies, errors

    def save_model(self):
        """Save autoencoder, scaler, threshold, and feature names."""
        self.autoencoder.save(os.path.join(self.model_path, 'autoencoder_final.keras'))
        joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))
        with open(os.path.join(self.model_path, 'threshold.txt'), 'w') as f:
            f.write(str(self.threshold))
        if self.feature_names:
            with open(os.path.join(self.model_path, 'feature_names.txt'), 'w') as f:
                f.write('\n'.join(self.feature_names))
        print(f"✅ Anomaly detector saved to {self.model_path}")

    def load_model(self):
        """Load trained autoencoder and associated artifacts."""
        self.autoencoder = keras.models.load_model(os.path.join(self.model_path, 'autoencoder_final.keras'))
        self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.pkl'))
        with open(os.path.join(self.model_path, 'threshold.txt'), 'r') as f:
            self.threshold = float(f.read().strip())
        feat_file = os.path.join(self.model_path, 'feature_names.txt')
        if os.path.exists(feat_file):
            with open(feat_file, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ Anomaly detector loaded from {self.model_path}")

    # ==================== NEW: TENSORFLOW LITE CONVERSION METHODS ====================
    
    def convert_to_tflite(self, model_part='autoencoder', quantization='float16', 
                          representative_data=None):
        """
        Convert the trained autoencoder or encoder to TensorFlow Lite format
        
        Args:
            model_part: Which part to convert ('autoencoder' or 'encoder')
            quantization: Type of quantization ('none', 'float16', 'int8', 'dynamic')
            representative_data: Representative dataset for int8 quantization
        
        Returns:
            Path to the saved TFLite model
        """
        print("\n" + "="*60)
        print(f"CONVERTING {model_part.upper()} TO TENSORFLOW LITE")
        print("="*60)
        
        # Check if model is loaded
        if self.autoencoder is None:
            print("❌ No model loaded. Please train or load a model first.")
            return None
        
        # Select which model to convert
        if model_part == 'autoencoder':
            model_to_convert = self.autoencoder
            model_name = 'autoencoder'
        elif model_part == 'encoder':
            if self.encoder is None:
                print("❌ Encoder not available. Building from autoencoder...")
                # Recreate encoder from autoencoder
                self.encoder = models.Model(
                    inputs=self.autoencoder.input,
                    outputs=self.autoencoder.layers[4].output  # Adjust based on your architecture
                )
            model_to_convert = self.encoder
            model_name = 'encoder'
        else:
            print(f"❌ Invalid model_part: {model_part}. Choose 'autoencoder' or 'encoder'")
            return None
        
        # Load feature names if available
        if self.feature_names is None:
            try:
                with open(os.path.join(self.model_path, 'feature_names.txt'), 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            except:
                print("⚠️ Feature names not found.")
                self.feature_names = [f'feature_{i}' for i in range(model_to_convert.input.shape[1])]
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
        
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
                    # Load scaled data (assuming it was saved)
                    X_train_scaled = np.load(os.path.join('data/training/', 'X_train_scaled.npy'))
                    converter.representative_dataset = self._get_representative_dataset(X_train_scaled[:100])
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
            tflite_filename = f'{model_name}{quant_suffix}.tflite'
            tflite_path = os.path.join(self.model_path, tflite_filename)
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get file size
            file_size_kb = len(tflite_model) / 1024
            
            print(f"\n✅ TFLite model saved to: {tflite_path}")
            print(f"📊 Model size: {file_size_kb:.2f} KB")
            
            # Save metadata
            self._save_tflite_metadata(tflite_path, model_name, quantization, file_size_kb)
            
            return tflite_path
            
        except Exception as e:
            print(f"❌ TFLite conversion failed: {e}")
            return None
    
    def _get_representative_dataset(self, data_samples):
        """Generate representative dataset for int8 quantization"""
        def representative_dataset():
            for i in range(min(len(data_samples), 100)):
                # Get a single sample and ensure correct shape and type
                sample = data_samples[i:i+1].astype(np.float32)
                yield [sample]
        return representative_dataset
    
    def _save_tflite_metadata(self, tflite_path, model_name, quantization, file_size_kb):
        """Save metadata for the TFLite model"""
        metadata = {
            'model_type': f'anomaly_detector_{model_name}',
            'quantization': quantization,
            'file_size_kb': file_size_kb,
            'input_features': self.feature_names,
            'input_shape': list(self.autoencoder.input_shape[1:]),
            'threshold': float(self.threshold) if self.threshold else None,
            'conversion_date': pd.Timestamp.now().isoformat(),
            'tensorflow_version': tf.__version__
        }
        
        metadata_path = os.path.join(self.model_path, f'tflite_metadata_{model_name}_{quantization}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to: {metadata_path}")
    
    def convert_all_formats(self):
        """
        Convert both autoencoder and encoder to all TFLite formats
        """
        print("\n" + "="*60)
        print("CONVERTING TO ALL TENSORFLOW LITE FORMATS")
        print("="*60)
        
        if self.autoencoder is None:
            print("❌ No model loaded. Please train or load a model first.")
            return {}
        
        # Try to load training data for int8 quantization
        representative_data = None
        try:
            # Try to load scaled training data if available
            X_train_scaled = np.load(os.path.join('data/training/', 'X_train_scaled.npy'))
            representative_data = self._get_representative_dataset(X_train_scaled[:100])
        except:
            print("⚠️ Could not load training data for int8 quantization.")
        
        results = {
            'autoencoder': {},
            'encoder': {}
        }
        
        # Convert autoencoder in different formats
        print("\n📦 Converting AUTOENCODER...")
        for quant in ['none', 'dynamic', 'float16']:
            print(f"\n{'-'*40}")
            path = self.convert_to_tflite(model_part='autoencoder', quantization=quant)
            if path:
                results['autoencoder'][quant] = path
        
        # Try int8 if representative data available
        if representative_data:
            print(f"\n{'-'*40}")
            path = self.convert_to_tflite(model_part='autoencoder', quantization='int8', 
                                          representative_data=representative_data)
            if path:
                results['autoencoder']['int8'] = path
        
        # Convert encoder in different formats
        print("\n📦 Converting ENCODER...")
        for quant in ['none', 'dynamic', 'float16']:
            print(f"\n{'-'*40}")
            path = self.convert_to_tflite(model_part='encoder', quantization=quant)
            if path:
                results['encoder'][quant] = path
        
        # Save comparison report
        self._save_tflite_comparison(results)
        
        return results
    
    def _save_tflite_comparison(self, results):
        """Save comparison of different TFLite models"""
        comparison = {
            'autoencoder': [],
            'encoder': []
        }
        
        for model_type in ['autoencoder', 'encoder']:
            for quant_type, path in results[model_type].items():
                if os.path.exists(path):
                    size_kb = os.path.getsize(path) / 1024
                    comparison[model_type].append({
                        'quantization': quant_type,
                        'path': path,
                        'size_kb': size_kb
                    })
            
            # Sort by size
            comparison[model_type].sort(key=lambda x: x['size_kb'])
        
        # Save comparison
        comparison_path = os.path.join(self.model_path, 'tflite_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("\n" + "="*60)
        print("TFLITE MODEL COMPARISON")
        print("="*60)
        
        for model_type in ['autoencoder', 'encoder']:
            print(f"\n📊 {model_type.upper()}:")
            if comparison[model_type]:
                for model in comparison[model_type]:
                    print(f"   {model['quantization']:8s}: {model['size_kb']:.2f} KB")
                print(f"   ✅ Recommended: {comparison[model_type][0]['quantization']} ({comparison[model_type][0]['size_kb']:.2f} KB)")
            else:
                print("   No models converted")
    
    def verify_tflite_model(self, tflite_path, test_samples=None):
        """
        Verify the TFLite model by running inference and comparing with original
        
        For autoencoder: compares reconstruction
        For encoder: compares latent representation
        """
        print("\n" + "="*60)
        print(f"VERIFYING TFLITE MODEL: {os.path.basename(tflite_path)}")
        print("="*60)
        
        # Determine if this is autoencoder or encoder
        is_autoencoder = 'autoencoder' in tflite_path
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📊 Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
        print(f"📊 Output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
        
        # Get test samples if not provided
        if test_samples is None:
            try:
                # Try to load test data
                X_test_scaled = np.load(os.path.join('data/training/', 'X_test_scaled.npy'))
                test_samples = X_test_scaled[:20]
            except:
                print("⚠️ Could not load test data. Generating random samples.")
                test_samples = np.random.randn(20, self.autoencoder.input_shape[1]).astype(np.float32)
        
        # Run inference on both models
        tflite_outputs = []
        original_outputs = []
        
        for i in range(len(test_samples)):
            # Get sample and ensure correct shape
            sample = test_samples[i:i+1].astype(np.float32)
            
            # TFLite inference
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]
            tflite_outputs.append(tflite_output)
            
            # Original model inference
            if is_autoencoder:
                original_output = self.autoencoder.predict(sample, verbose=0)[0]
            else:
                if self.encoder is None:
                    # Recreate encoder
                    self.encoder = models.Model(
                        inputs=self.autoencoder.input,
                        outputs=self.autoencoder.layers[4].output
                    )
                original_output = self.encoder.predict(sample, verbose=0)[0]
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
        
        # For autoencoder, also check anomaly detection capability
        if is_autoencoder:
            # Calculate MSE for each sample
            tflite_mse = np.mean(np.square(test_samples - tflite_outputs), axis=1)
            original_mse = np.mean(np.square(test_samples - original_outputs), axis=1)
            
            mse_diff = np.abs(tflite_mse - original_mse)
            print(f"\n📊 Anomaly Detection Metrics:")
            print(f"   MSE mean difference: {np.mean(mse_diff):.6f}")
            print(f"   MSE max difference: {np.max(mse_diff):.6f}")
            
            # Check if anomaly decisions would be the same
            if self.threshold:
                tflite_anomalies = tflite_mse > self.threshold
                original_anomalies = original_mse > self.threshold
                agreement = np.mean(tflite_anomalies == original_anomalies) * 100
                print(f"   Anomaly decision agreement: {agreement:.1f}%")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot for first feature dimension
        axes[0].scatter(original_outputs[:, 0], tflite_outputs[:, 0], alpha=0.7)
        min_val = min(original_outputs[:, 0].min(), tflite_outputs[:, 0].min())
        max_val = max(original_outputs[:, 0].max(), tflite_outputs[:, 0].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match')
        axes[0].set_xlabel('Original Model Output (dim 0)')
        axes[0].set_ylabel('TFLite Model Output (dim 0)')
        axes[0].set_title('Original vs TFLite Output Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Difference histogram (all dimensions)
        all_diffs = abs_diff.flatten()
        axes[1].hist(all_diffs, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Absolute Difference')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Prediction Differences (All Dimensions)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'tflite_verification.png'), dpi=150)
        plt.show()
        
        # Determine if verification passed
        max_allowed_diff = 0.01 if is_autoencoder else 0.1  # Autoencoder needs higher precision
        passed = np.max(abs_diff) < max_allowed_diff
        
        if passed:
            print(f"\n✅ Verification PASSED! (Max diff < {max_allowed_diff})")
        else:
            print(f"\n⚠️ Verification WARNING! (Max diff >= {max_allowed_diff})")
            print("   Consider using higher precision quantization or no quantization.")
        
        return passed
    
    def export_for_mobile(self):
        """
        Export optimized models for mobile deployment
        Creates TFLite models and metadata for mobile integration
        """
        print("\n" + "="*60)
        print("EXPORTING FOR MOBILE DEPLOYMENT")
        print("="*60)
        
        # Convert all formats
        tflite_results = self.convert_all_formats()
        
        if not tflite_results['autoencoder'] and not tflite_results['encoder']:
            print("❌ No TFLite models created")
            return None
        
        # Create mobile deployment package
        mobile_dir = os.path.join(self.model_path, 'mobile_deployment')
        os.makedirs(mobile_dir, exist_ok=True)
        
        import shutil
        
        # Copy the smallest autoencoder model
        comparison_path = os.path.join(self.model_path, 'tflite_comparison.json')
        with open(comparison_path, 'r') as f:
            comparison = json.load(f)
        
        # Copy autoencoder
        if comparison['autoencoder']:
            best_ae = comparison['autoencoder'][0]
            shutil.copy2(best_ae['path'], os.path.join(mobile_dir, 'anomaly_autoencoder.tflite'))
        
        # Copy encoder if available
        if comparison['encoder']:
            best_enc = comparison['encoder'][0]
            shutil.copy2(best_enc['path'], os.path.join(mobile_dir, 'anomaly_encoder.tflite'))
        
        # Copy scaler
        scaler_path = os.path.join(self.model_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            shutil.copy2(scaler_path, os.path.join(mobile_dir, 'scaler.pkl'))
        
        # Create metadata for mobile
        mobile_metadata = {
            'model_name': 'Anomaly Detector',
            'version': '1.0.0',
            'description': 'Autoencoder-based anomaly detection for solar/battery/signal data',
            'models': {
                'autoencoder': {
                    'file': 'anomaly_autoencoder.tflite',
                    'purpose': 'Reconstruct input to detect anomalies',
                    'quantization': comparison['autoencoder'][0]['quantization'] if comparison['autoencoder'] else 'unknown',
                    'size_kb': comparison['autoencoder'][0]['size_kb'] if comparison['autoencoder'] else 0
                },
                'encoder': {
                    'file': 'anomaly_encoder.tflite',
                    'purpose': 'Generate latent representation (8-dim)',
                    'quantization': comparison['encoder'][0]['quantization'] if comparison['encoder'] else 'unknown',
                    'size_kb': comparison['encoder'][0]['size_kb'] if comparison['encoder'] else 0
                }
            },
            'input_features': self.feature_names,
            'input_shape': self.autoencoder.input_shape[1] if self.autoencoder else None,
            'latent_dim': 8,
            'anomaly_threshold': float(self.threshold) if self.threshold else None,
            'threshold_percentile': 95,
            'preprocessing': {
                'requires_scaler': True,
                'scaler_file': 'scaler.pkl',
                'scale_type': 'standardization'
            }
        }
        
        mobile_metadata_path = os.path.join(mobile_dir, 'model_metadata.json')
        with open(mobile_metadata_path, 'w') as f:
            json.dump(mobile_metadata, f, indent=2)
        
        print(f"\n✅ Mobile deployment package created in: {mobile_dir}")
        print(f"   - Autoencoder: anomaly_autoencoder.tflite")
        print(f"   - Encoder: anomaly_encoder.tflite")
        print(f"   - Scaler: scaler.pkl")
        print(f"   - Metadata: model_metadata.json")
        
        return mobile_dir


if __name__ == "__main__":
    # Initialize anomaly detector
    detector = AnomalyDetector()
    
    # Try to load existing model first
    try:
        detector.load_model()
        print("\n✅ Existing anomaly detector loaded successfully!")
    except:
        print("\n" + "="*60)
        print("Training new anomaly detector...")
        print("="*60)
        
        # Example: Load your combined data
        # combined_df = pd.read_csv('data/combined_sensor_data.csv')
        # feature_cols = ['solar_voltage', 'solar_current', 'battery_voltage', 
        #                'battery_current', 'signal_strength', 'temperature']
        
        # For demonstration, create synthetic data
        print("\n📊 Creating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 10000
        n_features = 6
        
        # Normal data
        normal_data = np.random.randn(n_samples, n_features) * 0.5 + 1.0
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=500, replace=False)
        normal_data[anomaly_indices] += np.random.randn(500, n_features) * 3
        
        # Create DataFrame
        combined_df = pd.DataFrame(normal_data, 
                                   columns=['solar_voltage', 'solar_current', 'battery_voltage',
                                           'battery_current', 'signal_strength', 'temperature'])
        feature_cols = combined_df.columns.tolist()
        
        # Prepare data
        X_train, X_test, X_scaled = detector.prepare_anomaly_data(
            combined_df, 
            feature_columns=feature_cols,
            normal_ratio=0.7
        )
        
        # Save scaled data for later use
        os.makedirs('data/training/', exist_ok=True)
        np.save('data/training/X_train_scaled.npy', X_train)
        np.save('data/training/X_test_scaled.npy', X_test)
        
        # Train model
        history = detector.train(X_train, epochs=50)
        
        # Evaluate
        # Create ground truth (first 70% normal, rest unknown)
        y_true = np.zeros(len(X_test))
        # Simulate some known anomalies in test set
        y_true[200:250] = 1  # Mark some as anomalies for demonstration
        detector.evaluate(X_test, y_true)
        
        # Save model
        detector.save_model()
    
    # Convert to TensorFlow Lite
    print("\n" + "="*60)
    print("CONVERTING TO TENSORFLOW LITE")
    print("="*60)
    
    # Convert all formats
    tflite_results = detector.convert_all_formats()
    
    # Verify one of the converted models
    if tflite_results['autoencoder'] and 'float16' in tflite_results['autoencoder']:
        detector.verify_tflite_model(tflite_results['autoencoder']['float16'])
    
    # Export for mobile deployment
    mobile_dir = detector.export_for_mobile()
    
    print("\n" + "="*60)
    print("✅ ALL STEPS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("   - Use autoencoder for anomaly detection on edge devices")
    print("   - Use encoder for feature extraction (8-dim latent space)")
    print("   - Deploy TFLite models to mobile/IoT devices")
    print("   - Monitor reconstruction errors and adjust threshold if needed")