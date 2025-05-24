import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

def load_data(path='large_weather_dataset.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['dayofyear'] = df['date'].dt.dayofyear
    return df

def train_model(df):
    X = df[['latitude', 'longitude', 'dayofyear']]
    y = df['temperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Model training complete. Test R^2:", model.score(X_test, y_test))
    return model

def save_model(model, filename='model/weather_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename='model/weather_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)


    
df = load_data()
model = train_model(df)
    
if not os.path.exists('model'):
    os.mkdir('model')
save_model(model)
