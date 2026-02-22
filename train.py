import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("⏳ Generating dataset...")

# 1. Create a dummy dataset for apartments in Egypt
np.random.seed(42) # To get the same random numbers every time
data = {
    'Area': np.random.randint(80, 300, 500), # Area between 80 and 300 sqm
    'Bedrooms': np.random.randint(2, 6, 500), # 2 to 5 bedrooms
    'Location': np.random.choice(['Nasr City', 'Maadi', 'Zayed', 'New Cairo'], 500),
    'Finishing': np.random.choice(['Super Lux', 'Extra Super Lux', 'Without Finishing'], 500)
}
df = pd.DataFrame(data)

# 2. Add realistic logical pricing based on the features
df['Price'] = (df['Area'] * 15000) + (df['Bedrooms'] * 50000)
df.loc[df['Location'] == 'New Cairo', 'Price'] += 500000
df.loc[df['Location'] == 'Zayed', 'Price'] += 600000
df.loc[df['Finishing'] == 'Super Lux', 'Price'] += 200000
df.loc[df['Finishing'] == 'Extra Super Lux', 'Price'] += 400000

print("⚙️ Preprocessing data...")

# 3. Data Preprocessing (One-Hot Encoding)
# Machine Learning models only understand numbers, so we convert text (like 'Maadi') into 1s and 0s.
X = pd.get_dummies(df.drop('Price', axis=1))
y = df['Price']

print("🧠 Training the Random Forest Model...")

# 4. Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("💾 Saving the model...")

# 5. Save the trained model and the column names for later use in the Web App
joblib.dump(model, 'model.pkl')
joblib.dump(list(X.columns), 'columns.pkl')

print("✅ Model trained and saved successfully!")