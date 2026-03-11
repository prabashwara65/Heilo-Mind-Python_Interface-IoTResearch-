""" data perparation module for sync schedular
loads and perprocessed all 5 datasets """

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class DataPreparator:
    def __init__(self, raw_path='data/raw/', processed_path='data/processed/'):
        self.raw_path = raw_path
        self.processed_path = processed_path
        os.makedirs(self.processed_path, exist_ok=True)

        #store loaded datasets
        self.weather_data = {}
        self.battery_data = None
        self.signal_data = None
        self.panel_data = None
        self.user_data = None

        # list of all 9 provinces
        self.provinces = ['central', 'eastern', 'north_central','north_western', 'northern', 'sabaragamuwa','southern', 'uva', 'western']

    def load_all_datasets(self):
        """Load all 5 datasets with error handling """
        print("LOADING ALL 5 DATASETS")

        #1. load Nasa weather data
        nasa_path = os.path.join(self.raw_path, 'nasa_power_data')
        successful_provinces = 0

        for province in self.provinces:
            try:
                #construct filename
                filename = f"{province}.csv"
                file_path = os.path.join(nasa_path, filename)

                #check if file exists
                if not os.path.exists(file_path):
                    #try alternative naming convention
                    alt_filename = province.replace('_', '') + '.csv'
                    file_path = os.path.join(nasa_path, alt_filename)
                
                if os.path.exists(file_path):
                    # First, read the file to find where the actual data starts
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Find the line with column headers (contains YEAR, MO, DY, HR)
                    header_line_index = None
                    for i, line in enumerate(lines):
                        if 'YEAR' in line and 'MO' in line and 'DY' in line and 'HR' in line:
                            header_line_index = i
                            break
                    
                    if header_line_index is not None:
                        # Read the CSV starting from the header line
                        df = pd.read_csv(file_path, skiprows=header_line_index)
                        
                        #clean the column names
                        df.columns = df.columns.str.strip()

                        #create timestamp 
                        df['TIMESTAMP'] = pd.to_datetime(
                            df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
                            format='%Y-%m-%d-%H'
                        )

                        #handle missing values 
                        for col in ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS10M']:
                            if col in df.columns:
                                df[col] = df[col].replace(-999, np.nan)
                        
                        #fill missing values
                        df = df.fillna(method='ffill').fillna(method='bfill')

                        #add province name as a column
                        df['PROVINCE'] = province.title()

                        #store in dictionary with province as key 
                        self.weather_data[province] = df

                        successful_provinces += 1
                        print(f"    {province.title()}: {len(df):,} records loaded")
                        print(f"      Date range: {df['TIMESTAMP'].min().date()} to {df['TIMESTAMP'].max().date()}")
                    else:
                        print(f"    {province.title()}: Could not find header row in file")
                else:
                    print(f"    {province.title()}: File not found at {file_path}")

            except Exception as e:
                print(f"    Error loading {province}: {e}")


        #2.Load Battery Data
        try:
                battery_path = os.path.join(self.raw_path,'battery_data')
                battery_files = [f for f in os.listdir(battery_path) if f.endswith('.csv')]

                if battery_files:
                    battery_file = os.path.join(battery_path, battery_files[0])
                    self.battery_data = pd.read_csv(battery_file)

                    #clean column names
                    self.battery_data.columns = self.battery_data.columns.str.strip()

                    #calculate state of charge (soc) from voltage
                    if 'Voltage_measured' in self.battery_data.columns:
                        self.battery_data['soc'] = ((self.battery_data['Voltage_measured'] - 3.0) / (4.2 - 3.0)*100).clip(0,100)

                    print(f"\n Battery data loaded: {len(self.battery_data):,} records")
                    print(f"   Voltage range: {self.battery_data['Voltage_measured'].min():.2f}-{self.battery_data['Voltage_measured'].max():.2f}V")
                    print(f"   Columns: {list(self.battery_data.columns)[:5]}...")
                else:
                    print(f"\n No battery files found in {battery_path}")
                    self.create_synthetic_battery_data()

        except Exception as e :
                print(f"\n Error loading battery data: {e}")
                print("   Creating synthetic battery data instead...")
                self.create_synthetic_battery_data()

        #load signal data
        try:
            signal_path = os.path.join(self.raw_path,'iot_signal_data')
            signal_files = [f for f in os.listdir(signal_path) if f.endswith('.csv')]

            if signal_files:
                signal_file = os.path.join(signal_path, signal_files[0])
                self.signal_data = pd.read_csv(signal_file)

                #clean column names
                self.signal_data.columns = self.signal_data.columns.str.strip()

                print(f"\n Signal data loaded: {len(self.signal_data):,} records")
                if 'RSSI' in self.signal_data.columns:
                    print(f"   RSSI range: {self.signal_data['RSSI'].min():.1f} to {self.signal_data['RSSI'].max():.1f} dBm")
                if 'SNR' in self.signal_data.columns:
                    print(f"   SNR range: {self.signal_data['SNR'].min():.1f} to {self.signal_data['SNR'].max():.1f} dB")
                print(f"   Columns: {list(self.signal_data.columns)[:5]}...")
            else:
                print(f"\n No signal files found in {signal_path}")
                self.create_synthetic_signal_data()
        
        except Exception as e:
            print(f"\n Error loading signal data: {e}")
            print("   Creating synthetic signal data instead...")
            self.create_synthetic_signal_data()

        #load panel data
        try:
            panel_path = os.path.join(self.raw_path, 'panel_data')
            panel_files = [f for f in os.listdir(panel_path) if f.endswith('.csv')]

            if panel_files:
                panel_file = os.path.join(panel_path, panel_files[0])
                self.panel_data = pd.read_csv(panel_file)

                #cleam column names
                self.panel_data.columns = self.panel_data.columns.str.strip()

                #convert datetime
                if 'Date Time' in self.panel_data.columns:
                    self.panel_data['DateTime'] = pd.to_datetime(self.panel_data['Date Time'])
                elif 'DateTime' in self.panel_data.columns:
                     self.panel_data['DateTime'] = pd.to_datetime(self.panel_data['DateTime'])

                print(f"\n Panel data loaded: {len(self.panel_data):,} records")

                if 'Bus Voltage(V)' in self.panel_data.columns:
                    print(f"   Voltage range: {self.panel_data['Bus Voltage(V)'].min():.2f}-{self.panel_data['Bus Voltage(V)'].max():.2f}V")
                if 'Current(mA)' in self.panel_data.columns:
                    print(f"   Current range: {self.panel_data['Current(mA)'].min():.2f}-{self.panel_data['Current(mA)'].max():.2f}mA")
                    print(f"   Columns: {list(self.panel_data.columns)[:5]}...")
            else:
                print(f"\n No panel files found in {panel_path}")
                self.create_synthetic_panel_data()
        
        except Exception as e:
            print(f"\n Error loading panel data: {e}")
            print("   Creating synthetic panel data instead...")
            self.create_synthetic_panel_data()

        #load user data
        try:
            user_path = os.path.join(self.raw_path, 'user_behavior_data')
            user_files = [f for f in os.listdir(user_path) if f.endswith('.csv')]

            if user_files:
                user_file = os.path.join(user_path, user_files[0])
                self.user_data = pd.read_csv(user_file)

                #clean column names
                self.user_data.columns = self.user_data.columns.str.strip()

                print(f"\n User data loaded: {len(self.user_data):,} records")

                if 'App Usage Time (min/day)' in self.user_data.columns:
                    print(f"   App usage range: {self.user_data['App Usage Time (min/day)'].min():.0f}-{self.user_data['App Usage Time (min/day)'].max():.0f} min/day")
                if 'User Behavior Class' in self.user_data.columns:
                    print(f"   User classes: {sorted(self.user_data['User Behavior Class'].unique())}")
                print(f"   Columns: {list(self.user_data.columns)[:5]}...")
            else:
                print(f"\n No user files found in {user_path}")
                self.create_synthetic_user_data()
            
        except Exception as e:
            print(f"\n Error loading user data: {e}")
            print("   Creating synthetic user data instead...")
            self.create_synthetic_user_data()

        print(" ALL DATASETS LOADED SUCCESSFULLY!")

        print(f"   - Weather: {len(self.weather_data)} provinces loaded")
        print(f"   - Battery: {len(self.battery_data) if self.battery_data is not None else 0} records")
        print(f"   - Signal: {len(self.signal_data) if self.signal_data is not None else 0} records")
        print(f"   - Panel: {len(self.panel_data) if self.panel_data is not None else 0} records")
        print(f"   - User: {len(self.user_data) if self.user_data is not None else 0} records")
        
        return True
    
    def create_synthetic_battery_data(self):
        """Create synthetic battery data if real data not available"""
        np.random.seed(42)
        n_samples = 10000
        
        self.battery_data = pd.DataFrame({
            'Voltage_measured': np.random.normal(3.7, 0.3, n_samples).clip(3.0, 4.2),
            'Current_measured': np.random.normal(0, 1, n_samples).clip(-2, 2),
            'Temperature_measured': np.random.normal(25, 5, n_samples).clip(10, 40),
        })
        self.battery_data['SOC'] = ((self.battery_data['Voltage_measured'] - 3.0) / (4.2 - 3.0) * 100).clip(0, 100)
        print(f" Created {n_samples:,} synthetic battery records")

    def create_synthetic_signal_data(self):
        """Create synthetic signal data"""
        np.random.seed(42)
        n_samples = 10000
        
        self.signal_data = pd.DataFrame({
            'RSSI': np.random.normal(-80, 15, n_samples).clip(-110, -30),
            'SNR': np.random.normal(10, 8, n_samples).clip(-10, 30),
            'Distance': np.random.uniform(10, 1000, n_samples),
            'Transmission_Power': np.random.choice([5, 10, 15, 20], n_samples),
            'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Fog'], n_samples)
        })
        print(f"Created {n_samples:,} synthetic signal records")

    def create_synthetic_panel_data(self):
        """Create synthetic panel data"""
        np.random.seed(42)
        n_samples = 10000
        
        hours = np.random.choice(range(24), n_samples)
        irradiance = np.where((hours >= 6) & (hours <= 18), 
                              np.random.uniform(100, 800, n_samples), 
                              np.random.uniform(0, 50, n_samples))
        
        self.panel_data = pd.DataFrame({
            'DateTime': pd.date_range(start='2024-01-01', periods=n_samples, freq='5min'),
            'Bus Voltage(V)': np.random.normal(1.0, 0.2, n_samples).clip(0.5, 1.8),
            'Current(mA)': irradiance * 0.2 + np.random.normal(0, 10, n_samples),
            'Power(mW)': irradiance * 0.2 + np.random.normal(0, 5, n_samples),
            'Temperature(oC)': np.random.normal(30, 5, n_samples).clip(20, 45),
            'Humidity(%)': np.random.normal(60, 15, n_samples).clip(30, 90)
        })
        print(f"Created {n_samples:,} synthetic panel records")

    def create_synthetic_user_data(self):
        """Create synthetic user data"""
        np.random.seed(42)
        n_samples = 700
        
        self.user_data = pd.DataFrame({
            'User ID': range(1, n_samples+1),
            'App Usage Time (min/day)': np.random.normal(200, 100, n_samples).clip(30, 600),
            'Screen On Time (hours/day)': np.random.normal(5, 2, n_samples).clip(1, 12),
            'Battery Drain (mAh/day)': np.random.normal(1500, 500, n_samples).clip(300, 3000),
            'Data Usage (MB/day)': np.random.normal(1000, 500, n_samples).clip(100, 2500),
            'User Behavior Class': np.random.choice([1, 2, 3, 4, 5], n_samples)
        })
        print(f"Created {n_samples:,} synthetic user records")

    def get_province_summary(self):
        """Get summary of all loaded provinces"""
        if not self.weather_data:
            print("No province data loaded")
            return
        
        print("PROVINCE DATA SUMMARY")
        
        summary = []
        for province, df in self.weather_data.items():
            summary.append({
                'Province': province.title(),
                'Records': len(df),
                'Date Range': f"{df['TIMESTAMP'].min().date()} to {df['TIMESTAMP'].max().date()}",
                'Avg Temp': f"{df['T2M'].mean():.1f}°C",
                'Avg Irradiance': f"{df['ALLSKY_SFC_SW_DWN'].mean():.0f} W/m²",
                'Max Irradiance': f"{df['ALLSKY_SFC_SW_DWN'].max():.0f} W/m²"
            })
        
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def get_province_data(self, province_name):
        """Get data for a specific province"""
        province_key = province_name.lower().replace(' ', '_')
        if province_key in self.weather_data:
            return self.weather_data[province_key]
        else:
            print(f"Province '{province_name}' not found. Available: {list(self.weather_data.keys())}")
            return None
        
    def save_processed_data(self):
        """Save cleaned datasets - each province separately"""
        print("\n Saving processed datasets...")
        
        # Save each province separately
        if self.weather_data:
            province_dir = os.path.join(self.processed_path, 'provinces')
            os.makedirs(province_dir, exist_ok=True)
            
            for province, df in self.weather_data.items():
                filename = f"{province}_cleaned.csv"
                filepath = os.path.join(province_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"    Saved {province.title()} to {filepath}")
        
        # Save other datasets
        if self.battery_data is not None:
            self.battery_data.to_csv(os.path.join(self.processed_path, 'battery_cleaned.csv'), index=False)
            print(f"    Saved battery data")
        
        if self.signal_data is not None:
            self.signal_data.to_csv(os.path.join(self.processed_path, 'signal_cleaned.csv'), index=False)
            print(f"    Saved signal data")
        
        if self.panel_data is not None:
            self.panel_data.to_csv(os.path.join(self.processed_path, 'panel_cleaned.csv'), index=False)
            print(f"    Saved panel data")
        
        if self.user_data is not None:
            self.user_data.to_csv(os.path.join(self.processed_path, 'user_cleaned.csv'), index=False)
            print(f"    Saved user data")
        
        print(f"\n All datasets saved to: {self.processed_path}")


    
if __name__ == "__main__":
    preparator = DataPreparator()
    success = preparator.load_all_datasets()
    
    if success:
        # Show province summary
        preparator.get_province_summary()
        
        # Example: Access specific province
        western_data = preparator.get_province_data('western')
        if western_data is not None:
            print(f"\n Sample from Western Province:")
            print(western_data[['TIMESTAMP', 'T2M', 'ALLSKY_SFC_SW_DWN']].head())
        
        # Save all processed data
        preparator.save_processed_data()
