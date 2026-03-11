"""
Feature Engineering Module for Sync Scheduler
Combines all 5 datasets into training features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib
import glob

class FeatureEngineer:
    def __init__(self, processed_path='data/processed/', training_path='data/training/'):
        self.processed_path = processed_path
        self.training_path = training_path
        os.makedirs(training_path, exist_ok=True)
        
        # Load processed data
        self.weather_data = {}  # Dictionary for province-wise weather
        self.battery = None
        self.signal = None
        self.panel = None
        self.user = None
        
    def load_processed_data(self):
        """Load the cleaned datasets from data_preparator output"""
        print("\n" + "="*60)
        print("STEP 2: LOADING PROCESSED DATASETS")
        print("="*60)
        
        # Load province-wise weather data
        province_dir = os.path.join(self.processed_path, 'provinces')
        if os.path.exists(province_dir):
            province_files = glob.glob(os.path.join(province_dir, '*_cleaned.csv'))
            
            for file_path in province_files:
                province_name = os.path.basename(file_path).replace('_cleaned.csv', '')
                df = pd.read_csv(file_path)
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                self.weather_data[province_name] = df
                print(f"   ✅ Weather - {province_name.title()}: {len(df)} records")
        
        # Load other datasets
        battery_path = os.path.join(self.processed_path, 'battery_cleaned.csv')
        if os.path.exists(battery_path):
            self.battery = pd.read_csv(battery_path)
            print(f"   ✅ Battery: {len(self.battery)} records")
        
        signal_path = os.path.join(self.processed_path, 'signal_cleaned.csv')
        if os.path.exists(signal_path):
            self.signal = pd.read_csv(signal_path)
            print(f"   ✅ Signal: {len(self.signal)} records")
        
        panel_path = os.path.join(self.processed_path, 'panel_cleaned.csv')
        if os.path.exists(panel_path):
            self.panel = pd.read_csv(panel_path)
            # Convert datetime
            if 'DateTime' in self.panel.columns:
                self.panel['DateTime'] = pd.to_datetime(self.panel['DateTime'])
            elif 'Date Time' in self.panel.columns:
                self.panel['DateTime'] = pd.to_datetime(self.panel['Date Time'])
            print(f"   ✅ Panel: {len(self.panel)} records")
        
        user_path = os.path.join(self.processed_path, 'user_cleaned.csv')
        if os.path.exists(user_path):
            self.user = pd.read_csv(user_path)
            print(f"   ✅ User: {len(self.user)} records")
        
        print(f"\n📊 Summary:")
        print(f"   - Weather: {len(self.weather_data)} provinces loaded")
        print(f"   - Battery: {len(self.battery) if self.battery is not None else 0} records")
        print(f"   - Signal: {len(self.signal) if self.signal is not None else 0} records")
        print(f"   - Panel: {len(self.panel) if self.panel is not None else 0} records")
        print(f"   - User: {len(self.user) if self.user is not None else 0} records")
        
    def create_features(self, province='western', n_samples=None):
        """
        Create feature matrix for sync scheduler
        Combines all datasets into one feature set for a specific province
        """
        print("\n" + "="*60)
        print(f"STEP 3: CREATING FEATURES FOR SYNC SCHEDULER - {province.title()} Province")
        print("="*60)
        
        if province not in self.weather_data:
            available = list(self.weather_data.keys())
            print(f"⚠️ Province '{province}' not found. Available: {available}")
            if not available:
                raise ValueError("No weather data loaded for any province.")
            province = available[0]
            print(f"   Using '{province.title()}' instead.")
        
        weather_df = self.weather_data[province].copy()
        print(f"   Weather data shape: {weather_df.shape}")
        
        # Limit samples if specified
        if n_samples and n_samples < len(weather_df):
            weather_df = weather_df.sample(n=n_samples, random_state=42).sort_values('TIMESTAMP')
            print(f"   Sampled {len(weather_df)} records")
        else:
            print(f"   Using all {len(weather_df)} records")
        
        # Add time features to weather
        weather_df['hour'] = weather_df['TIMESTAMP'].dt.hour
        weather_df['day'] = weather_df['TIMESTAMP'].dt.day
        weather_df['month'] = weather_df['TIMESTAMP'].dt.month
        weather_df['day_of_week'] = weather_df['TIMESTAMP'].dt.dayofweek
        weather_df['is_weekend'] = (weather_df['day_of_week'] >= 5).astype(int)
        weather_df['is_daytime'] = ((weather_df['hour'] >= 6) & (weather_df['hour'] <= 18)).astype(int)
        
        # ==================== PANEL HOURLY STATISTICS ====================
        if self.panel is not None:
            self.panel['hour'] = self.panel['DateTime'].dt.hour
            # Check which columns exist to avoid KeyError
            panel_cols = self.panel.columns
            agg_dict = {}
            if 'Bus Voltage(V)' in panel_cols:
                agg_dict['Bus Voltage(V)'] = 'mean'
            if 'Current(mA)' in panel_cols:
                agg_dict['Current(mA)'] = 'mean'
            if 'Power(mW)' in panel_cols:
                agg_dict['Power(mW)'] = 'mean'
            if 'Temperature(oC)' in panel_cols:
                agg_dict['Temperature(oC)'] = 'mean'
            if 'Humidity(%)' in panel_cols:
                agg_dict['Humidity(%)'] = 'mean'
            
            if agg_dict:
                panel_hourly = self.panel.groupby('hour').agg(agg_dict).rename(columns={
                    'Bus Voltage(V)': 'panel_voltage_mean',
                    'Current(mA)': 'panel_current_mean',
                    'Power(mW)': 'panel_power_mean',
                    'Temperature(oC)': 'panel_temp_mean',
                    'Humidity(%)': 'panel_humidity_mean'
                })
            else:
                print("⚠️ No expected panel columns found. Using synthetic panel data.")
                panel_hourly = self._create_synthetic_panel_hourly()
        else:
            print("⚠️ No panel data loaded. Using synthetic panel data.")
            panel_hourly = self._create_synthetic_panel_hourly()
        
        # ==================== USER STATISTICS BY CLASS ====================
        if self.user is not None:
            user_cols = self.user.columns
            agg_dict = {}
            if 'App Usage Time (min/day)' in user_cols:
                agg_dict['App Usage Time (min/day)'] = 'mean'
            if 'Screen On Time (hours/day)' in user_cols:
                agg_dict['Screen On Time (hours/day)'] = 'mean'
            if 'Battery Drain (mAh/day)' in user_cols:
                agg_dict['Battery Drain (mAh/day)'] = 'mean'
            if 'Data Usage (MB/day)' in user_cols:
                agg_dict['Data Usage (MB/day)'] = 'mean'
            if 'Number of Apps Installed' in user_cols:
                agg_dict['Number of Apps Installed'] = 'mean'
            
            if 'User Behavior Class' in user_cols and agg_dict:
                user_by_class = self.user.groupby('User Behavior Class').agg(agg_dict).rename(columns={
                    'App Usage Time (min/day)': 'avg_app_usage_min',
                    'Screen On Time (hours/day)': 'avg_screen_on_hours',
                    'Battery Drain (mAh/day)': 'avg_battery_drain_mah',
                    'Data Usage (MB/day)': 'avg_data_usage_mb',
                    'Number of Apps Installed': 'avg_apps_installed'
                })
            else:
                print("⚠️ User data missing required columns. Using synthetic user stats.")
                user_by_class = self._create_synthetic_user_stats()
        else:
            print("⚠️ No user data loaded. Using synthetic user stats.")
            user_by_class = self._create_synthetic_user_stats()
        
        # ==================== CREATE TRAINING SAMPLES ====================
        print(f"\n🔄 Creating {len(weather_df)} training samples...")
        
        training_samples = []
        errors = 0
        
        for idx, weather_row in weather_df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processed {idx}/{len(weather_df)} samples...")
            
            try:
                hour = weather_row['hour']
                month = weather_row['month']
                
                # ----- 1. Panel features for this hour -----
                if hour in panel_hourly.index:
                    panel_features = panel_hourly.loc[hour].to_dict()
                else:
                    panel_features = {
                        'panel_voltage_mean': panel_hourly['panel_voltage_mean'].mean(),
                        'panel_current_mean': panel_hourly['panel_current_mean'].mean(),
                        'panel_power_mean': panel_hourly['panel_power_mean'].mean(),
                        'panel_temp_mean': panel_hourly['panel_temp_mean'].mean(),
                        'panel_humidity_mean': panel_hourly['panel_humidity_mean'].mean()
                    }
                
                # ----- 2. Battery sample -----
                if self.battery is not None:
                    # Safely access columns
                    if 'Voltage_measured' in self.battery.columns:
                        if weather_row['is_daytime']:
                            candidates = self.battery[self.battery['Voltage_measured'] > 3.8]
                        else:
                            candidates = self.battery[self.battery['Voltage_measured'] <= 3.8]
                        
                        if len(candidates) > 0:
                            battery_sample = candidates.sample(1).iloc[0]
                        else:
                            battery_sample = self.battery.sample(1).iloc[0]
                    else:
                        # Fallback if column missing
                        battery_sample = self.battery.sample(1).iloc[0]
                    
                    # Get SOC (might be 'soc' or 'SOC')
                    battery_soc = battery_sample.get('soc', battery_sample.get('SOC', 50))
                    battery_voltage = battery_sample.get('Voltage_measured', 3.7)
                    battery_current = battery_sample.get('Current_measured', 0.0)
                    battery_temp = battery_sample.get('Temperature_measured', 25.0)
                else:
                    # Synthetic battery
                    battery_voltage = np.random.normal(3.7, 0.3)
                    battery_current = np.random.normal(0, 1)
                    battery_temp = np.random.normal(25, 5)
                    battery_soc = ((battery_voltage - 3.0) / (4.2 - 3.0) * 100)
                
                # ----- 3. Signal sample -----
                if self.signal is not None:
                    # Filter by weather if possible
                    if 'Weather_Condition' in self.signal.columns:
                        if weather_row['RH2M'] > 80:
                            candidates = self.signal[self.signal['Weather_Condition'].str.contains('Rain|Fog', na=False)]
                        else:
                            candidates = self.signal[self.signal['Weather_Condition'].str.contains('Clear', na=False)]
                        
                        if len(candidates) > 0:
                            signal_sample = candidates.sample(1).iloc[0]
                        else:
                            signal_sample = self.signal.sample(1).iloc[0]
                    else:
                        signal_sample = self.signal.sample(1).iloc[0]
                    
                    rssi = signal_sample.get('RSSI', -80)
                    snr = signal_sample.get('SNR', 10)
                    distance = signal_sample.get('Distance', 500)
                    tx_power = signal_sample.get('Transmission_Power', 10)
                else:
                    rssi = np.random.normal(-80, 15)
                    snr = np.random.normal(10, 8)
                    distance = np.random.uniform(10, 1000)
                    tx_power = np.random.choice([5, 10, 15, 20])
                
                # ----- 4. User class based on time -----
                if hour >= 18 or hour <= 22:  # Evening peak
                    user_class_weights = {5: 0.4, 4: 0.3, 3: 0.2, 2: 0.07, 1: 0.03}
                elif hour >= 7 and hour <= 9:  # Morning peak
                    user_class_weights = {3: 0.3, 4: 0.25, 2: 0.2, 5: 0.15, 1: 0.1}
                else:  # Off-peak
                    user_class_weights = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
                
                # Normalize weights
                classes = list(user_class_weights.keys())
                weights = np.array(list(user_class_weights.values()))
                weights = weights / weights.sum()
                
                selected_class = np.random.choice(classes, p=weights)
                
                # Get user features for this class
                if selected_class in user_by_class.index:
                    user_features = user_by_class.loc[selected_class].to_dict()
                else:
                    user_features = {
                        'avg_app_usage_min': 200,
                        'avg_screen_on_hours': 5,
                        'avg_battery_drain_mah': 1500,
                        'avg_data_usage_mb': 1000,
                        'avg_apps_installed': 50
                    }
                
                # ----- Assemble feature row -----
                feature_row = {
                    # Temporal features
                    'hour': hour,
                    'month': month,
                    'day_of_week': weather_row['day_of_week'],
                    'is_weekend': weather_row['is_weekend'],
                    'is_daytime': weather_row['is_daytime'],
                    
                    # Weather features
                    'temperature': weather_row['T2M'],
                    'humidity': weather_row['RH2M'],
                    'irradiance': weather_row['ALLSKY_SFC_SW_DWN'],
                    'clear_sky_irradiance': weather_row['CLRSKY_SFC_SW_DWN'],
                    'wind_speed': weather_row['WS10M'],
                    
                    # Panel features (SCALED to your 10W system)
                    'panel_voltage': panel_features['panel_voltage_mean'] * 5,      # Scale to 5V
                    'panel_current': panel_features['panel_current_mean'] * 20 / 1000,  # mA to A, scale to 2A
                    'panel_power': panel_features['panel_power_mean'] * 100 / 1000, # mW to W, scale to 10W
                    'panel_temperature': panel_features['panel_temp_mean'],
                    
                    # Battery features
                    'battery_voltage': battery_voltage,
                    'battery_current': battery_current,
                    'battery_temperature': battery_temp,
                    'battery_soc': battery_soc,
                    
                    # Signal features
                    'rssi': rssi,
                    'snr': snr,
                    'distance': distance,
                    'transmission_power': tx_power,
                    
                    # User features (scaled to per-hour values)
                    'user_class': selected_class,
                    'app_usage_hourly': user_features.get('avg_app_usage_min', 200) / 24,
                    'screen_on_hourly': user_features.get('avg_screen_on_hours', 5) / 24,
                    'data_usage_hourly': user_features.get('avg_data_usage_mb', 1000) / 24,
                    'battery_drain_hourly': user_features.get('avg_battery_drain_mah', 1500) / 24,
                    'apps_installed': user_features.get('avg_apps_installed', 50),
                    
                    # Province metadata
                    'province': province
                }
                
                training_samples.append(feature_row)
                
            except Exception as e:
                errors += 1
                if errors <= 5:  # Print first few errors
                    print(f"   ⚠️ Error at index {idx}: {e}")
                continue
        
        self.features_df = pd.DataFrame(training_samples)
        
        print(f"\n✅ Created {len(self.features_df)} training samples")
        if errors > 0:
            print(f"⚠️ Encountered {errors} errors during feature creation (ignored).")
        if len(self.features_df) > 0:
            print(f"✅ Number of features: {len(self.features_df.columns)}")
        else:
            print("❌ No features were created! Check errors above.")
        
        return self.features_df
    
    def _create_synthetic_panel_hourly(self):
        """Create synthetic panel hourly statistics"""
        hours = range(24)
        data = []
        for hour in hours:
            if 6 <= hour <= 18:  # Daytime
                voltage = np.random.normal(1.0, 0.1)
                current = np.random.normal(200, 50)
                power = np.random.normal(200, 50)
                temp = np.random.normal(35, 5)
                humidity = np.random.normal(60, 10)
            else:  # Nighttime
                voltage = np.random.normal(0.5, 0.1)
                current = np.random.normal(10, 5)
                power = np.random.normal(5, 2)
                temp = np.random.normal(25, 3)
                humidity = np.random.normal(70, 10)
            
            data.append({
                'panel_voltage_mean': voltage,
                'panel_current_mean': current,
                'panel_power_mean': power,
                'panel_temp_mean': temp,
                'panel_humidity_mean': humidity
            })
        
        return pd.DataFrame(data, index=hours)
    
    def _create_synthetic_user_stats(self):
        """Create synthetic user statistics by class"""
        return pd.DataFrame({
            'avg_app_usage_min': [50, 120, 200, 300, 450],
            'avg_screen_on_hours': [2, 4, 6, 8, 11],
            'avg_battery_drain_mah': [500, 1000, 1500, 2000, 2800],
            'avg_data_usage_mb': [200, 500, 1000, 1800, 2500],
            'avg_apps_installed': [20, 35, 50, 70, 90]
        }, index=[1, 2, 3, 4, 5])
    
    def create_target_variable(self):
        """
        Create target variable (sync_probability) based on:
        - High probability (>0.6): Good conditions
        - Medium probability (0.2-0.6): Moderate conditions
        - Low probability (<0.2): Poor conditions
        """
        print("\n" + "="*60)
        print("STEP 4: CREATING TARGET VARIABLE")
        print("="*60)
        
        if not hasattr(self, 'features_df') or len(self.features_df) == 0:
            print("❌ No features available. Run create_features() first.")
            return None
        
        df = self.features_df.copy()
        
        # Calculate base probability from multiple factors
        # Factor 1: Solar availability (0-1)
        solar_factor = df['irradiance'] / 1000
        solar_factor = solar_factor.clip(0, 1)
        
        # Factor 2: Battery state (0-1)
        battery_factor = df['battery_soc'] / 100
        battery_factor = battery_factor.clip(0, 1)
        
        # Factor 3: Signal quality (0-1)
        signal_factor = (df['rssi'] + 100) / 50  # -100 to -50 becomes 0 to 1
        signal_factor = signal_factor.clip(0, 1)
        
        # Factor 4: User activity (0-1)
        user_factor = df['app_usage_hourly'] / 30  # Max 30 min/hour
        user_factor = user_factor.clip(0, 1)
        
        # Factor 5: Time of day boost
        time_factor = df['is_daytime'] * 0.2
        
        # Weighted combination
        df['sync_probability'] = (
            0.30 * solar_factor +
            0.25 * battery_factor +
            0.20 * signal_factor +
            0.15 * user_factor +
            0.10 * time_factor
        ).clip(0, 1)
        
        # Create categorical target
        conditions = [
            df['sync_probability'] > 0.6,
            df['sync_probability'] >= 0.2,
            df['sync_probability'] < 0.2
        ]
        choices = [2, 1, 0]  # 2=high, 1=medium, 0=low
        df['sync_priority'] = np.select(conditions, choices)
        
        print("\n📊 Target Variable Statistics:")
        print(f"   Probability range: {df['sync_probability'].min():.3f} - {df['sync_probability'].max():.3f}")
        print(f"   Mean probability: {df['sync_probability'].mean():.3f}")
        print(f"   Std probability: {df['sync_probability'].std():.3f}")
        print("\n   Priority distribution:")
        
        high_pct = len(df[df['sync_priority']==2]) / len(df) * 100
        medium_pct = len(df[df['sync_priority']==1]) / len(df) * 100
        low_pct = len(df[df['sync_priority']==0]) / len(df) * 100
        
        print(f"     High (2): {len(df[df['sync_priority']==2]):6d} samples ({high_pct:.1f}%)")
        print(f"     Medium (1): {len(df[df['sync_priority']==1]):6d} samples ({medium_pct:.1f}%)")
        print(f"     Low (0): {len(df[df['sync_priority']==0]):6d} samples ({low_pct:.1f}%)")
        
        self.target_df = df[['sync_probability', 'sync_priority']]
        return self.target_df
    
    def prepare_for_training(self, test_size=0.2, random_state=42, feature_subset=None):
        """
        Select features and scale for model training.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing.
        random_state : int
            Random seed.
        feature_subset : list, optional
            List of feature names to use. If None, uses the full default list.
        """
        print("\n" + "="*60)
        print("STEP 5: PREPARING DATA FOR MODEL TRAINING")
        print("="*60)
        
        if not hasattr(self, 'features_df') or len(self.features_df) == 0:
            print("❌ No features available. Run create_features() first.")
            return None
        if not hasattr(self, 'target_df') or len(self.target_df) == 0:
            print("❌ No target available. Run create_target_variable() first.")
            return None
        
        # Default feature set (full)
        default_features = [
            # Temporal
            'hour', 'is_daytime', 'is_weekend', 'month',
            # Weather
            'temperature', 'humidity', 'irradiance', 'wind_speed',
            # Panel
            'panel_voltage', 'panel_current', 'panel_power', 'panel_temperature',
            # Battery
            'battery_voltage', 'battery_current', 'battery_temperature', 'battery_soc',
            # Signal
            'rssi', 'snr', 'distance', 'transmission_power',
            # User
            'app_usage_hourly', 'screen_on_hourly', 'data_usage_hourly', 
            'battery_drain_hourly', 'apps_installed'
        ]
        
        # Use subset if provided, otherwise full set
        if feature_subset is not None:
            feature_columns = feature_subset
        else:
            feature_columns = default_features
        
        # Check which features are available
        available_features = [col for col in feature_columns if col in self.features_df.columns]
        missing_features = [col for col in feature_columns if col not in self.features_df.columns]
        
        if missing_features:
            print(f"⚠️ Warning: Missing requested features: {missing_features}")
        
        if not available_features:
            print("❌ No requested features available in dataframe.")
            return None
        
        X = self.features_df[available_features].copy()
        y_prob = self.target_df['sync_probability']
        y_class = self.target_df['sync_priority']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Scale features to 0-1 range
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Split into train/test
        X_train, X_test, y_prob_train, y_prob_test, y_class_train, y_class_test = train_test_split(
            X_scaled, y_prob, y_class, test_size=test_size, random_state=random_state, stratify=y_class
        )
        
        # Save training data
        X_train.to_csv(os.path.join(self.training_path, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(self.training_path, 'X_test.csv'), index=False)
        y_prob_train.to_csv(os.path.join(self.training_path, 'y_prob_train.csv'), index=False)
        y_prob_test.to_csv(os.path.join(self.training_path, 'y_prob_test.csv'), index=False)
        y_class_train.to_csv(os.path.join(self.training_path, 'y_class_train.csv'), index=False)
        y_class_test.to_csv(os.path.join(self.training_path, 'y_class_test.csv'), index=False)
        
        # Save feature names and scaler
        with open(os.path.join(self.training_path, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(available_features))
        
        joblib.dump(scaler, os.path.join(self.training_path, 'feature_scaler.pkl'))
        
        print(f"\n✅ Training data saved to: {self.training_path}")
        print(f"   Features used: {len(available_features)}")
        print(f"   Total samples: {len(X_scaled)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        print(f"\n📋 Feature list ({len(available_features)} features):")
        for i, col in enumerate(available_features[:10]):
            print(f"   {i+1:2d}. {col}")
        if len(available_features) > 10:
            print(f"   ... and {len(available_features)-10} more")
        
        return X_train, X_test, y_prob_train, y_prob_test, y_class_train, y_class_test, available_features
    
    def create_features_for_all_provinces(self, n_samples_per_province=20000):
        """
        Create features for all provinces and combine them
        """
        print("\n" + "="*60)
        print("STEP: CREATING FEATURES FOR ALL PROVINCES")
        print("="*60)
        
        all_features = []
        
        for province in self.weather_data.keys():
            print(f"\n🔄 Processing {province.title()}...")
            df_prov = self.create_features(province=province, n_samples=n_samples_per_province)
            all_features.append(df_prov)
        
        self.features_df = pd.concat(all_features, ignore_index=True)
        print(f"\n✅ Combined dataset: {len(self.features_df)} samples from {len(self.weather_data)} provinces")
        
        return self.features_df


# Run feature engineering
if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Step 1: Load all processed data
    engineer.load_processed_data()
    
    # Step 2: Create features using ALL provinces (20,000 samples per province)
    features = engineer.create_features_for_all_provinces(n_samples_per_province=20000)
    
    # Step 3: Create target variable
    if features is not None and len(features) > 0:
        target = engineer.create_target_variable()
        
        # Step 4: Prepare for training with a reduced feature subset
        RELIABLE_FEATURES = [
            'hour',
            'is_daytime',
            'month',
            'temperature',
            'humidity',
            'irradiance',
            'battery_voltage',
            'rssi',
            'app_usage_hourly', 
            'panel_voltage', 
            'panel_current', 
            'panel_power',
            
        ]
        # =========================================================================
        
        if target is not None:
            result = engineer.prepare_for_training(feature_subset=RELIABLE_FEATURES)
            
            if result is not None:
                X_train, X_test, y_prob_train, y_prob_test, y_class_train, y_class_test, feature_names = result
                
                print("\n" + "="*60)
                print("✅ FEATURE ENGINEERING COMPLETE!")
                print("="*60)
                print(f"\nNext step: Run model training with:")
                print("   python src/sync_scheduler_model.py")
    else:
        print("\n❌ Feature creation failed. Please check the errors above.")