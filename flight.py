
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def create_sample_data():
    """Generate sample flight data"""
    np.random.seed(42)
    n = 1000
    
    air_time = np.random.normal(120, 30, n)
    distance = np.random.normal(800, 200, n)
    delay = air_time * 0.1 + distance * 0.01 + np.random.normal(0, 10, n)
    
    return pd.DataFrame({
        'AirTime': air_time,
        'Distance': distance,
        'ArrDelayMinutes': delay
    })

def analyze_data(df):
    """Quick data analysis"""
    print(f"Dataset: {len(df)} flights")
    print(f"Average delay: {df['ArrDelayMinutes'].mean():.1f} minutes")
    
    # Simple plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['ArrDelayMinutes'], bins=30, alpha=0.7)
    plt.title('Delay Distribution')
    plt.xlabel('Minutes')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['AirTime'], df['ArrDelayMinutes'], alpha=0.5)
    plt.title('Air Time vs Delay')
    plt.xlabel('Air Time (min)')
    plt.ylabel('Delay (min)')
    
    plt.tight_layout()
    plt.show()

def build_model(df):
    """Train prediction model"""
    X = df[['AirTime', 'Distance']]
    y = df['ArrDelayMinutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Model RMSE: {rmse:.2f} minutes")
    return model

def predict_delay(model):
    """Interactive prediction"""
    print("\n🔮 Flight Delay Predictor")
    
    try:
        air_time = float(input("Air time (minutes): "))
        distance = float(input("Distance (miles): "))
        
        prediction = model.predict([[air_time, distance]])[0]
        print(f"Predicted delay: {prediction:.1f} minutes")
        
        if prediction < 0:
            print("✅ Early arrival expected!")
        elif prediction < 15:
            print("✅ On time")
        else:
            print("⚠️ Delay expected")
            
    except ValueError:
        print("Please enter valid numbers")

def main():
    print("🛫 Flight Delay Analysis\n")
    
    # Load or create data
    try:
        file_path = input("CSV file path (Enter for sample data): ").strip()
        if file_path:
            df = pd.read_csv(file_path)
        else:
            df = create_sample_data()
    except:
        print("Using sample data...")
        df = create_sample_data()
    
    # Analysis
    analyze_data(df)
    
    # Build model
    model = build_model(df)
    
    # Interactive predictions
    while True:
        predict_delay(model)
        if input("\nAnother prediction? (y/n): ").lower() != 'y':
            break
    
    print("Thanks for flying! ✈️")

if __name__ == "__main__":
    main()