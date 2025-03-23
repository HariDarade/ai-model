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
STAFFING_BASELINE = 1 / 5  # 1 staff per 5 patients as a baseline
STAFFING_FACTORS = {'Severe': 2, 'Moderate': 1, 'Mild': 0.5}
SHIFT_FACTORS = {'Day': 1.0, 'Night': 1.2}  # Night shifts require more staff

# --- Helper Functions ---
def compute_occupancy_rate(department):
    return CURRENT_OCCUPANCY[department] / TOTAL_BEDS[department]

def update_occupancy(department, change):
    CURRENT_OCCUPANCY[department] = max(0, min(TOTAL_BEDS[department], CURRENT_OCCUPANCY[department] + change))

def generate_synthetic_data():
    patient_data = []
    department_staff_needs_day = {dept: 0 for dept in DEPARTMENTS}
    department_staff_needs_night = {dept: 0 for dept in DEPARTMENTS}
    
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
        shift = random.choice(['Day', 'Night'])
        
        # Simulate staff needs per patient
        staff_needed = STAFFING_BASELINE * STAFFING_FACTORS[severity] * SHIFT_FACTORS[shift]
        if shift == 'Day':
            department_staff_needs_day[department] += staff_needed
        else:
            department_staff_needs_night[department] += staff_needed
        
        update_occupancy(department, 1)
        
        patient_data.append([patient_id, department, bed_id, admission_date, diagnosis, length_of_stay,
                             discharge_date, severity, age, comorbidities, occupancy_rate, day_of_week, patient_load, shift])
    
    # Aggregate staff needs per department and shift
    staff_data_day = []
    staff_data_night = []
    for dept in DEPARTMENTS:
        patient_data.append([f"{dept}_staff_day", dept, None, None, None, None, None, None, None, None, compute_occupancy_rate(dept), datetime.now().weekday(), CURRENT_OCCUPANCY[dept], 'Day'])
        patient_data.append([f"{dept}_staff_night", dept, None, None, None, None, None, None, None, None, compute_occupancy_rate(dept), datetime.now().weekday(), CURRENT_OCCUPANCY[dept], 'Night'])
        staff_data_day.append(department_staff_needs_day[dept])
        staff_data_night.append(department_staff_needs_night[dept])
    
    columns = ['PatientID', 'Department', 'BedID', 'AdmissionDate', 'DiagnosisType', 'LengthOfStay',
               'DischargeDate', 'Severity', 'Age', 'Comorbidities', 'OccupancyRate', 'DayOfWeek', 'PatientLoad', 'Shift']
    return pd.DataFrame(patient_data, columns=columns), pd.Series(staff_data_day + staff_data_night)

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
df, y_staff = generate_synthetic_data()
df_for_training = df.tail(2 * len(DEPARTMENTS))  # Use only the aggregated department rows for staff prediction
categorical_cols = ['Department', 'Severity', 'Shift']
numerical_cols = ['OccupancyRate', 'DayOfWeek', 'PatientLoad']
X, encoder, scaler = preprocess_data(df_for_training, categorical_cols, numerical_cols)

feature_names = X.columns
joblib.dump(feature_names, 'staff_allocation_feature_names.pkl')
joblib.dump(encoder, 'staff_allocation_encoder.pkl')
joblib.dump(scaler, 'staff_allocation_scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y_staff, test_size=0.2, random_state=42)

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
joblib.dump(best_model, 'staff_allocation_model.pkl')
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Staff Allocation Model - Mean Absolute Error (MAE): {mae:.2f}')
print(f'Best hyperparameters: {grid_search.best_params_}')

# --- Prediction Function ---
def predict_staff_allocation(patients):
    model = joblib.load('staff_allocation_model.pkl')
    encoder = joblib.load('staff_allocation_encoder.pkl')
    scaler = joblib.load('staff_allocation_scaler.pkl')
    feature_names = joblib.load('staff_allocation_feature_names.pkl')
    
    # Aggregate patient data by department
    department_data = {dept: {'PatientLoad': 0, 'OccupancyRate': compute_occupancy_rate(dept), 'SeverityCounts': {'Severe': 0, 'Moderate': 0, 'Mild': 0}} for dept in DEPARTMENTS}
    for patient in patients:
        dept = patient['Department']
        department_data[dept]['PatientLoad'] += 1
        department_data[dept]['SeverityCounts'][patient['Severity']] += 1
    
    # Prepare data for prediction for both shifts
    aggregated_data = []
    for dept in DEPARTMENTS:
        for shift in ['Day', 'Night']:
            data = {
                'Department': dept,
                'Severity': 'Severe',  # Dummy value, will be encoded
                'Shift': shift,
                'OccupancyRate': department_data[dept]['OccupancyRate'],
                'DayOfWeek': datetime.now().weekday(),
                'PatientLoad': department_data[dept]['PatientLoad']
            }
            aggregated_data.append(data)
    
    patient_df = pd.DataFrame(aggregated_data)
    categorical_cols = ['Department', 'Severity', 'Shift']
    numerical_cols = ['OccupancyRate', 'DayOfWeek', 'PatientLoad']
    
    encoded_data = encoder.transform(patient_df[categorical_cols]).toarray()
    scaled_data = scaler.transform(patient_df[numerical_cols])
    
    df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_cols)
    patient_X = pd.concat([df_scaled, df_encoded], axis=1)
    
    patient_X = patient_X.reindex(columns=feature_names, fill_value=0)
    
    staff_predictions = model.predict(patient_X)
    
    suggestions = []
    for idx, (dept, shift) in enumerate([(dept, shift) for dept in DEPARTMENTS for shift in ['Day', 'Night']]):
        staff = staff_predictions[idx]
        if staff > 0:
            suggestions.append(f"{dept} requires {staff:.1f} staff members for the {shift} shift.")
    
    return suggestions

# Simulate staff allocation prediction for testing
if __name__ == "__main__":
    patients = [
        {'Department': 'Cardiology', 'Severity': 'Severe', 'OccupancyRate': compute_occupancy_rate('Cardiology'), 'DayOfWeek': datetime.now().weekday(), 'PatientLoad': CURRENT_OCCUPANCY['Cardiology']},
        {'Department': 'General Surgery', 'Severity': 'Moderate', 'OccupancyRate': compute_occupancy_rate('General Surgery'), 'DayOfWeek': datetime.now().weekday(), 'PatientLoad': CURRENT_OCCUPANCY['General Surgery']},
        {'Department': 'Infectious Diseases', 'Severity': 'Severe', 'OccupancyRate': compute_occupancy_rate('Infectious Diseases'), 'DayOfWeek': datetime.now().weekday(), 'PatientLoad': CURRENT_OCCUPANCY['Infectious Diseases']}
    ]
    staff_suggestions = predict_staff_allocation(patients)
    print("\nStaff Allocation Suggestions:")
    for suggestion in staff_suggestions:
        print(suggestion)