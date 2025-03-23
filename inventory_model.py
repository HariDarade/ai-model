import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import random
from datetime import datetime, timedelta
import joblib

# Set seed for reproducibilityimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import random
from datetime import datetime, timedelta
import joblib

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- Constants ---
NUM_PATIENTS = 200
TOTAL_BEDS = {'ICU': 20, 'Cardiology': 40, 'General Surgery': 60, 'Infectious Diseases': 40, 'ER': 40}
DEPARTMENTS = list(TOTAL_BEDS.keys())
DIAGNOSIS_TYPES = ['Heart Attack', 'Appendectomy', 'Respiratory Infection', 'Stroke', 'Fracture']
SEVERITY_LEVELS = ['Mild', 'Moderate', 'Severe']
INVENTORY_ITEMS = ['Oxygen Tanks', 'IV Fluids', 'Syringes', 'Bandages', 'Medications']
STOCK_LEVELS = {'Oxygen Tanks': 20, 'IV Fluids': 30, 'Syringes': 50, 'Bandages': 40, 'Medications': 25}
CURRENT_OCCUPANCY = {dept: int(num_beds * 0.5) for dept, num_beds in TOTAL_BEDS.items()}
HISTORICAL_USAGE = {item: [random.uniform(5, 15) for _ in range(7)] for item in INVENTORY_ITEMS}

# Seasonal adjustment factor (mock: higher demand in winter months)
def get_seasonal_factor(admission_date):
    month = admission_date.month
    if month in [12, 1, 2]:  # Winter (Dec-Feb)
        return 1.2
    elif month in [6, 7, 8]:  # Summer (Jun-Aug)
        return 0.8
    else:
        return 1.0

# --- Helper Functions ---
def compute_occupancy_rate(department):
    return CURRENT_OCCUPANCY[department] / TOTAL_BEDS[department]

def generate_inventory_usage(severity, department, length_of_stay, admission_date):
    base_usage = {item: 0 for item in INVENTORY_ITEMS}
    
    if severity == 'Severe':
        base_usage['Oxygen Tanks'] = random.randint(3, 6)
        base_usage['IV Fluids'] = random.randint(2, 5)
        base_usage['Medications'] = random.randint(3, 6)
    elif severity == 'Moderate':
        base_usage['IV Fluids'] = random.randint(1, 3)
        base_usage['Medications'] = random.randint(1, 3)
    else:  # Mild
        base_usage['IV Fluids'] = random.randint(0, 1)
        base_usage['Medications'] = random.randint(0, 2)
    
    if department == 'General Surgery':
        base_usage['Syringes'] = random.randint(2, 4)
        base_usage['Bandages'] = random.randint(2, 4)
    elif department in ['ICU', 'Infectious Diseases']:
        base_usage['Oxygen Tanks'] += random.randint(1, 3)
        base_usage['Medications'] += random.randint(1, 2)
    
    # Normalize usage by length of stay and apply seasonal factor
    seasonal_factor = get_seasonal_factor(admission_date)
    for item in base_usage:
        base_usage[item] = (base_usage[item] / 7) * length_of_stay * seasonal_factor
    
    return base_usage

def generate_synthetic_data():
    patient_data = []
    for i in range(NUM_PATIENTS):
        patient_id = i + 1
        department = random.choice(DEPARTMENTS)
        bed_id = f"{department[:3].upper()}-{random.randint(1, TOTAL_BEDS[department])}"
        admission_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        diagnosis = random.choice(DIAGNOSIS_TYPES)
        severity = random.choice(SEVERITY_LEVELS)
        
        if department == 'ER':
            length_of_stay = random.randint(1, 2)
        elif department == 'ICU':
            length_of_stay = random.randint(5, 14) if severity == 'Severe' else random.randint(3, 7)
        else:
            length_of_stay = random.randint(3, 10) if severity in ['Moderate', 'Severe'] else random.randint(1, 5)
        
        discharge_date = admission_date + timedelta(days=length_of_stay)
        age = random.randint(20, 80)
        comorbidities = random.randint(0, 3)
        occupancy_rate = compute_occupancy_rate(department)
        day_of_week = admission_date.weekday()
        patient_load = CURRENT_OCCUPANCY[department]
        
        patient_data.append([patient_id, department, bed_id, admission_date, diagnosis, length_of_stay,
                             discharge_date, severity, age, comorbidities, occupancy_rate, day_of_week, patient_load])
    
    columns = ['PatientID', 'Department', 'BedID', 'AdmissionDate', 'DiagnosisType', 'LengthOfStay',
               'DischargeDate', 'Severity', 'Age', 'Comorbidities', 'OccupancyRate', 'DayOfWeek', 'PatientLoad']
    return pd.DataFrame(patient_data, columns=columns)

def preprocess_data(df, categorical_cols, numerical_cols):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_cols]).toarray()
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_data, columns=categorical_feature_names)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_cols)

    X = pd.concat([df_scaled, df_encoded], axis=1)
    return X, encoder, scaler

# --- Training ---
df = generate_synthetic_data()
categorical_cols = ['Department', 'DiagnosisType', 'Severity']
numerical_cols = ['LengthOfStay', 'Age', 'Comorbidities', 'OccupancyRate', 'DayOfWeek', 'PatientLoad']
X, encoder, scaler = preprocess_data(df, categorical_cols, numerical_cols)

feature_names = X.columns
joblib.dump(feature_names, 'inventory_feature_names.pkl')
joblib.dump(encoder, 'inventory_encoder.pkl')
joblib.dump(scaler, 'inventory_scaler.pkl')

inventory_df = pd.DataFrame([generate_inventory_usage(row['Severity'], row['Department'], row['LengthOfStay'], row['AdmissionDate']) for _, row in df.iterrows()])
for item in INVENTORY_ITEMS:
    y_inventory = inventory_df[item]
    X_train, X_test, y_train, y_test = train_test_split(X, y_inventory, test_size=0.2, random_state=42)
    
    # Use XGBoost with hyperparameter tuning
    model = XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f'inventory_model_{item.lower().replace(" ", "_")}.pkl')
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Inventory Model {item} - Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Best hyperparameters for {item}: {grid_search.best_params_}')

# --- Prediction Function with Alerts and Trends ---
def predict_hospital_inventory(patients):
    encoder = joblib.load('inventory_encoder.pkl')
    scaler = joblib.load('inventory_scaler.pkl')
    feature_names = joblib.load('inventory_feature_names.pkl')
    
    patient_df = pd.DataFrame(patients)
    categorical_cols = ['Department', 'DiagnosisType', 'Severity']
    numerical_cols = ['LengthOfStay', 'Age', 'Comorbidities', 'OccupancyRate', 'DayOfWeek', 'PatientLoad']
    
    encoded_data = encoder.transform(patient_df[categorical_cols]).toarray()
    scaled_data = scaler.transform(patient_df[numerical_cols])
    
    df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_cols)
    patient_X = pd.concat([df_scaled, df_encoded], axis=1)
    
    patient_X = patient_X.reindex(columns=feature_names, fill_value=0)
    
    inventory_predictions = {}
    alerts = []
    trends = []
    for item in INVENTORY_ITEMS:
        model = joblib.load(f'inventory_model_{item.lower().replace(" ", "_")}.pkl')
        item_predictions = model.predict(patient_X)
        total_usage = item_predictions.sum()
        inventory_predictions[item] = total_usage
        
        # Check against stock levels
        stock = STOCK_LEVELS[item]
        if total_usage > stock:
            alerts.append(f"Alert: {item} usage ({total_usage:.1f} units) exceeds stock level ({stock} units). Restock immediately.")
        
        # Compute historical trend
        historical_avg = np.mean(HISTORICAL_USAGE[item])
        trends.append(f"Trend: Average {item} usage over the last 7 days is {historical_avg:.1f} units.")

    return inventory_predictions, alerts, trends
np.random.seed(42)
random.seed(42)

# --- Constants ---
NUM_PATIENTS = 200
TOTAL_BEDS = {'ICU': 20, 'Cardiology': 40, 'General Surgery': 60, 'Infectious Diseases': 40, 'ER': 40}
DEPARTMENTS = list(TOTAL_BEDS.keys())
DIAGNOSIS_TYPES = ['Heart Attack', 'Appendectomy', 'Respiratory Infection', 'Stroke', 'Fracture']
SEVERITY_LEVELS = ['Mild', 'Moderate', 'Severe']
INVENTORY_ITEMS = ['Oxygen Tanks', 'IV Fluids', 'Syringes', 'Bandages', 'Medications']
STOCK_LEVELS = {'Oxygen Tanks': 20, 'IV Fluids': 30, 'Syringes': 50, 'Bandages': 40, 'Medications': 25}  # Mock stock levels

# Simulate current bed occupancy (50% occupancy)
CURRENT_OCCUPANCY = {dept: int(num_beds * 0.5) for dept, num_beds in TOTAL_BEDS.items()}

# Simulate historical inventory usage (last 7 days)
HISTORICAL_USAGE = {item: [random.uniform(5, 15) for _ in range(7)] for item in INVENTORY_ITEMS}

# --- Helper Functions ---
def compute_occupancy_rate(department):
    return CURRENT_OCCUPANCY[department] / TOTAL_BEDS[department]

def generate_inventory_usage(severity, department):
    base_usage = {item: 0 for item in INVENTORY_ITEMS}
    
    if severity == 'Severe':
        base_usage['Oxygen Tanks'] = random.randint(3, 6)
        base_usage['IV Fluids'] = random.randint(2, 5)
        base_usage['Medications'] = random.randint(3, 6)
    elif severity == 'Moderate':
        base_usage['IV Fluids'] = random.randint(1, 3)
        base_usage['Medications'] = random.randint(1, 3)
    else:  # Mild
        base_usage['IV Fluids'] = random.randint(0, 1)
        base_usage['Medications'] = random.randint(0, 2)
    
    if department == 'General Surgery':
        base_usage['Syringes'] = random.randint(2, 4)
        base_usage['Bandages'] = random.randint(2, 4)
    elif department in ['ICU', 'Infectious Diseases']:
        base_usage['Oxygen Tanks'] += random.randint(1, 3)
        base_usage['Medications'] += random.randint(1, 2)
    
    return base_usage

def generate_synthetic_data():
    patient_data = []
    for i in range(NUM_PATIENTS):
        patient_id = i + 1
        department = random.choice(DEPARTMENTS)
        bed_id = f"{department[:3].upper()}-{random.randint(1, TOTAL_BEDS[department])}"
        admission_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        diagnosis = random.choice(DIAGNOSIS_TYPES)
        severity = random.choice(SEVERITY_LEVELS)
        
        if department == 'ER':
            length_of_stay = random.randint(1, 2)
        elif department == 'ICU':
            length_of_stay = random.randint(5, 14) if severity == 'Severe' else random.randint(3, 7)
        else:
            length_of_stay = random.randint(3, 10) if severity in ['Moderate', 'Severe'] else random.randint(1, 5)
        
        discharge_date = admission_date + timedelta(days=length_of_stay)
        age = random.randint(20, 80)
        comorbidities = random.randint(0, 3)
        occupancy_rate = compute_occupancy_rate(department)
        
        patient_data.append([patient_id, department, bed_id, admission_date, diagnosis, length_of_stay,
                             discharge_date, severity, age, comorbidities, occupancy_rate])
    
    columns = ['PatientID', 'Department', 'BedID', 'AdmissionDate', 'DiagnosisType', 'LengthOfStay',
               'DischargeDate', 'Severity', 'Age', 'Comorbidities', 'OccupancyRate']
    return pd.DataFrame(patient_data, columns=columns)

def preprocess_data(df, categorical_cols, numerical_cols):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_cols]).toarray()
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_data, columns=categorical_feature_names)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_cols)

    X = pd.concat([df_scaled, df_encoded], axis=1)
    return X, encoder, scaler

# --- Training ---
df = generate_synthetic_data()
categorical_cols = ['Department', 'DiagnosisType', 'Severity']
numerical_cols = ['LengthOfStay', 'Age', 'Comorbidities', 'OccupancyRate']
X, encoder, scaler = preprocess_data(df, categorical_cols, numerical_cols)

feature_names = X.columns
joblib.dump(feature_names, 'inventory_feature_names.pkl')
joblib.dump(encoder, 'inventory_encoder.pkl')
joblib.dump(scaler, 'inventory_scaler.pkl')

inventory_df = pd.DataFrame([generate_inventory_usage(row['Severity'], row['Department']) for _, row in df.iterrows()])
for item in INVENTORY_ITEMS:
    y_inventory = inventory_df[item]
    X_train, X_test, y_train, y_test = train_test_split(X, y_inventory, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, f'inventory_model_{item.lower().replace(" ", "_")}.pkl')
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Inventory Model {item} - Mean Absolute Error (MAE): {mae:.2f}')

# --- Prediction Function with Alerts and Trends ---
def predict_hospital_inventory(patients):
    encoder = joblib.load('inventory_encoder.pkl')
    scaler = joblib.load('inventory_scaler.pkl')
    feature_names = joblib.load('inventory_feature_names.pkl')
    
    patient_df = pd.DataFrame(patients)
    categorical_cols = ['Department', 'DiagnosisType', 'Severity']
    numerical_cols = ['LengthOfStay', 'Age', 'Comorbidities', 'OccupancyRate']
    
    encoded_data = encoder.transform(patient_df[categorical_cols]).toarray()
    scaled_data = scaler.transform(patient_df[numerical_cols])
    
    df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_cols)
    patient_X = pd.concat([df_scaled, df_encoded], axis=1)
    
    patient_X = patient_X.reindex(columns=feature_names, fill_value=0)
    
    inventory_predictions = {}
    alerts = []
    trends = []
    for item in INVENTORY_ITEMS:
        model = joblib.load(f'inventory_model_{item.lower().replace(" ", "_")}.pkl')
        item_predictions = model.predict(patient_X)
        total_usage = item_predictions.sum()
        inventory_predictions[item] = total_usage
        
        # Check against stock levels
        stock = STOCK_LEVELS[item]
        if total_usage > stock:
            alerts.append(f"Alert: {item} usage ({total_usage:.1f} units) exceeds stock level ({stock} units). Restock immediately.")
        
        # Compute historical trend (average usage over the last 7 days)
        historical_avg = np.mean(HISTORICAL_USAGE[item])
        trends.append(f"Trend: Average {item} usage over the last 7 days is {historical_avg:.1f} units.")

    return inventory_predictions, alerts, trends