import pandas as pd
import numpy as np

# Simulate customer data
np.random.seed(42)
n = 1000

data = {
    'CustomerID': range(1, n+1),
    'Age': np.random.randint(18, 65, size=n),
    'Gender': np.random.choice(['Male', 'Female'], size=n),
    'Region': np.random.choice(['Lagos', 'Abuja', 'Kano', 'Port Harcourt'], size=n),
    'Plan': np.random.choice(['Basic', 'Standard', 'Premium'], size=n),
    'ContractLength': np.random.choice([12, 24, 36], size=n),
    'MonthlySpend': np.random.uniform(10, 100, size=n).round(2),
    'LatePayments': np.random.poisson(1, size=n),
    'Complaints': np.random.poisson(0.5, size=n),
    'DataUsage_GB': np.random.normal(5, 2, size=n).round(2),
    'SatisfactionScore': np.random.randint(1, 6, size=n),
    'Churn': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('telecom_customer_data.csv', index=False)
print("CSV file 'telecom_customer_data.csv' created!")
print("Data simulation complete. DataFrame created with shape:", df.shape)