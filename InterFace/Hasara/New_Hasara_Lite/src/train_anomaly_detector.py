"""
Training script for Anomaly Detector
Loads processed data, prepares features, trains autoencoder, and saves model.
Includes TensorFlow Lite conversion for mobile/edge deployment
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anomaly_detector import AnomalyDetector
from feature_engineering import FeatureEngineer  # Reuse your feature engineering class

def inject_anomalies_for_testing(X_test, anomaly_ratio=0.1, random_seed=42):
    """
    Inject synthetic anomalies into test data for evaluation
    Returns modified X_test and ground truth labels
    """
    np.random.seed(random_seed)
    n_test = X_test.shape[0]
    n_anomalies = int(anomaly_ratio * n_test)
    anomaly_indices = np.random.choice(n_test, n_anomalies, replace=False)
    
    y_true = np.zeros(n_test)
    X_test_modified = X_test.copy()
    
    for idx in anomaly_indices:
        # Create different types of anomalies
        anomaly_type = np.random.choice(['scale', 'shift', 'noise', 'spike'])
        
        if anomaly_type == 'scale':
            # Scale values drastically
            X_test_modified[idx] = X_test[idx] * np.random.uniform(2, 5)
        elif anomaly_type == 'shift':
            # Add large constant shift
            X_test_modified[idx] = X_test[idx] + np.random.uniform(3, 6)
        elif anomaly_type == 'noise':
            # Add high magnitude noise
            noise = np.random.randn(*X_test[idx].shape) * 3
            X_test_modified[idx] = X_test[idx] + noise
        else:  # spike
            # Spike in random feature
            spike_feature = np.random.randint(0, X_test.shape[1])
            X_test_modified[idx, spike_feature] *= np.random.uniform(5, 10)
        
        y_true[idx] = 1
    
    print(f"🔧 Injected {n_anomalies} synthetic anomalies ({anomaly_ratio*100:.0f}% of test set)")
    print(f"   Anomaly types: scale, shift, noise, spike")
    
    return X_test_modified, y_true

def save_training_artifacts(detector, X_train, X_test, feature_names, output_dir='data/training/'):
    """Save training data for later use in TFLite conversion"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scaled data for representative dataset
    np.save(os.path.join(output_dir, 'X_train_scaled.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test_scaled.npy'), X_test)
    
    # Save feature names
    with open(os.path.join(output_dir, 'anomaly_feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Save training summary
    summary = {
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
        'anomaly_threshold': float(detector.threshold) if detector.threshold else None,
        'training_complete': True
    }
    
    with open(os.path.join(output_dir, 'anomaly_training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Training artifacts saved to {output_dir}")

def plot_anomaly_detection_results(detector, X_test, y_true, anomalies, errors, save_path='models/anomaly_detector/'):
    """Plot anomaly detection results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error distribution
    axes[0, 0].hist(errors[y_true==0], bins=30, alpha=0.7, label='Normal', density=True)
    axes[0, 0].hist(errors[y_true==1], bins=30, alpha=0.7, label='Anomaly', density=True)
    axes[0, 0].axvline(detector.threshold, color='r', linestyle='--', label=f'Threshold ({detector.threshold:.4f})')
    axes[0, 0].set_xlabel('Reconstruction Error (MSE)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Error Distribution by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error vs sample index
    axes[0, 1].scatter(range(len(errors)), errors, c=y_true, cmap='coolwarm', alpha=0.6, s=10)
    axes[0, 1].axhline(detector.threshold, color='r', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Reconstruction Error')
    axes[0, 1].set_title('Anomaly Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, anomalies)
    
    axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Normal', 'Anomaly'])
    axes[1, 0].set_yticklabels(['Normal', 'Anomaly'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    # 4. ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    
    axes[1, 1].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Random')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'anomaly_detection_results.png'), dpi=150)
    plt.show()
    
    print(f"📊 ROC AUC: {roc_auc:.4f}")

def main():
    print("="*70)
    print("ANOMALY DETECTOR TRAINING PIPELINE")
    print("="*70)

    # Step 1: Load processed data using your existing FeatureEngineer
    engineer = FeatureEngineer()
    engineer.load_processed_data()

    # Step 2: Create features (use all provinces, maybe fewer samples for anomaly detection)
    # We'll create a combined dataset with all provinces, e.g., 10k samples each
    features_df = engineer.create_features_for_all_provinces(n_samples_per_province=10000)

    if features_df is None or len(features_df) == 0:
        print("❌ Feature creation failed.")
        return

    # Step 3: Define features for anomaly detection
    anomaly_features = [
        'panel_voltage', 'panel_current', 'panel_power', 'panel_temperature',
        'battery_voltage', 'battery_current', 'battery_temperature', 'battery_soc',
        'rssi', 'snr', 'irradiance', 'temperature'
    ]

    # Check which are available
    available = [f for f in anomaly_features if f in features_df.columns]
    missing = [f for f in anomaly_features if f not in features_df.columns]
    if missing:
        print(f"⚠️ Missing features: {missing}. Using only available: {available}")
    if not available:
        print("❌ No anomaly features available. Exiting.")
        return

    # Step 4: Prepare data for anomaly detection
    detector = AnomalyDetector(model_path='models/anomaly_detector/')

    # We assume first 70% of data (by index) is normal. If you have labelled anomalies, adjust.
    X_train, X_test, X_scaled = detector.prepare_anomaly_data(
        combined_df=features_df,
        feature_columns=available,
        normal_ratio=0.7
    )

    # Step 5: Train autoencoder on normal data
    print("\n" + "="*70)
    print("TRAINING AUTOENCODER")
    print("="*70)
    detector.train(X_train, epochs=50, batch_size=32)

    # Step 6: Evaluate on test set with synthetic anomalies
    print("\n" + "="*70)
    print("EVALUATING WITH SYNTHETIC ANOMALIES")
    print("="*70)
    
    X_test_modified, y_true = inject_anomalies_for_testing(X_test, anomaly_ratio=0.1)
    anomalies, errors = detector.evaluate(X_test_modified, y_true_anomaly=y_true)
    
    # Plot detailed results
    plot_anomaly_detection_results(detector, X_test_modified, y_true, anomalies, errors)

    # Step 7: Save model and training artifacts
    print("\n" + "="*70)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*70)
    detector.save_model()
    save_training_artifacts(detector, X_train, X_test, available)

    # ==================== NEW: TENSORFLOW LITE CONVERSION ====================
    print("\n" + "="*70)
    print("CONVERTING TO TENSORFLOW LITE")
    print("="*70)
    
    # Convert to all TFLite formats
    tflite_results = detector.convert_all_formats()
    
    # Verify the converted models
    if tflite_results['autoencoder'] and 'float16' in tflite_results['autoencoder']:
        print("\n🔍 Verifying float16 autoencoder...")
        detector.verify_tflite_model(tflite_results['autoencoder']['float16'], 
                                     test_samples=X_test[:20])
    
    if tflite_results['encoder'] and 'float16' in tflite_results['encoder']:
        print("\n🔍 Verifying float16 encoder...")
        detector.verify_tflite_model(tflite_results['encoder']['float16'],
                                     test_samples=X_test[:20])
    
    # Export for mobile deployment
    print("\n" + "="*70)
    print("EXPORTING FOR MOBILE DEPLOYMENT")
    print("="*70)
    mobile_dir = detector.export_for_mobile()
    
    # Generate deployment instructions
    if mobile_dir:
        print("\n📱 MOBILE DEPLOYMENT INSTRUCTIONS")
        print("="*70)
        print(f"Model files are ready in: {mobile_dir}")
        print("\nFor Android (Java/Kotlin):")
        print("  1. Copy .tflite files to app/src/main/assets/")
        print("  2. Copy scaler.pkl to app/src/main/assets/")
        print("  3. Use TensorFlow Lite Interpreter for inference")
        print("\nFor iOS (Swift):")
        print("  1. Add .tflite files to your Xcode project")
        print("  2. Copy scaler.pkl to your app bundle")
        print("  3. Use TensorFlowLiteSwift pod")
        print("\nExample Android code snippet:")
        print("""
        // Load model
        Interpreter tflite = new Interpreter(loadModelFile());
        
        // Preprocess with scaler (implement in Java)
        float[] input = preprocessWithScaler(rawData, scaler);
        
        // Run inference
        float[][] output = new float[1][inputFeatures];
        tflite.run(input, output);
        
        // Calculate reconstruction error
        float mse = 0;
        for (int i = 0; i < input.length; i++) {
            mse += Math.pow(input[i] - output[0][i], 2);
        }
        mse /= input.length;
        
        // Detect anomaly
        boolean isAnomaly = mse > anomalyThreshold;
        """)

    print("\n" + "="*70)
    print("✅ ANOMALY DETECTOR TRAINING COMPLETE!")
    print("="*70)
    
    # Print summary
    print("\n📊 TRAINING SUMMARY")
    print("-"*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(available)}")
    print(f"Anomaly threshold: {detector.threshold:.6f}")
    print(f"Autoencoder saved to: models/anomaly_detector/autoencoder_final.keras")
    print(f"TFLite models created: {len(tflite_results['autoencoder'])} formats")
    print(f"Mobile package created: {mobile_dir}")
    print("\nNext steps:")
    print("   - Use autoencoder for real-time anomaly detection on edge devices")
    print("   - Use encoder for feature extraction (8-dim latent space)")
    print("   - Monitor reconstruction errors and adjust threshold if needed")
    print("   - Deploy to production with the mobile deployment package")

if __name__ == "__main__":
    main()