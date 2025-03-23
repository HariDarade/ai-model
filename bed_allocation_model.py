import pandas as pd
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
CURRENT_OCCUPANCY = {dept: int(num_beds * 0.5) for dept, num_beds in TOTAL_BEDS.items()}

# --- Helper Functions ---
def compute_occupancy_rate(department):
    return CURRENT_OCCUPANCY[department] / TOTAL_BEDS[department]

def update_occupancy(department, change):
    CURRENT_OCCUPANCY[department] = max(0, min(TOTAL_BEDS[department], CURRENT_OCCUPANCY[department] + change))

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
            base_los = random.randint(1, 2)
        elif department == 'ICU':
            base_los = random.randint(5, 14) if severity == 'Severe' else random.randint(3, 7)
        else:
            base_los = random.randint(3, 10) if severity in ['Moderate', 'Severe'] else random.randint(1, 5)
        length_of_stay = max(1, int(base_los + random.uniform(-2, 2)))
        
        discharge_date = admission_date + timedelta(days=length_of_stay)
        age = random.randint(20, 80)
        comorbidities = random.randint(0, 3)
        occupancy_rate = compute_occupancy_rate(department)
        day_of_week = admission_date.weekday()
        patient_load = CURRENT_OCCUPANCY[department]
        
        update_occupancy(department, 1)
        
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
joblib.dump(feature_names, 'bed_allocation_feature_names.pkl')
joblib.dump(encoder, 'bed_allocation_encoder.pkl')
joblib.dump(scaler, 'bed_allocation_scaler.pkl')

y_bed = (df['DischargeDate'] - df['AdmissionDate']).dt.days
X_train, X_test, y_train, y_test = train_test_split(X, y_bed, test_size=0.2, random_state=42)

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
joblib.dump(best_model, 'bed_allocation_model.pkl')
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Bed Allocation Model - Mean Absolute Error (MAE): {mae:.2f}')
print(f'Best hyperparameters: {grid_search.best_params_}')

# --- Prediction Function ---
def predict_bed_availability(patients):
    model = joblib.load('bed_allocation_model.pkl')
    encoder = joblib.load('bed_allocation_encoder.pkl')
    scaler = joblib.load('bed_allocation_scaler.pkl')
    feature_names = joblib.load('bed_allocation_feature_names.pkl')
    
    patient_df = pd.DataFrame(patients)
    categorical_cols = ['Department', 'DiagnosisType', 'Severity']
    numerical_cols = ['LengthOfStay', 'Age', 'Comorbidities', 'OccupancyRate', 'DayOfWeek', 'PatientLoad']
    
    encoded_data = encoder.transform(patient_df[categorical_cols]).toarray()
    scaled_data = scaler.transform(patient_df[numerical_cols])
    
    df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_cols)
    patient_X = pd.concat([df_scaled, df_encoded], axis=1)
    
    patient_X = patient_X.reindex(columns=feature_names, fill_value=0)
    return model.predict(patient_X)

# --- Discharge Simulation Function ---
def simulate_discharges(patients):
    discharges = []
    current_date = datetime.now()
    for idx, patient in enumerate(patients):
        discharge_date = patient['AdmissionDate'] + timedelta(days=patient['LengthOfStay'])
        days_until_discharge = (discharge_date - current_date).days
        if days_until_discharge <= 1:
            discharges.append((idx + 1, patient['Department'], days_until_discharge))
            update_occupancy(patient['Department'], -1)
    return discharges