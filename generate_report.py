# generate_report.py
import pandas as pd
import numpy as np
import random
import re
import os
import torch
import speech_recognition as sr
import pyttsx3
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datetime import datetime, timedelta

# Set environment variable to disable TensorFlow oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Constants for departments
TOTAL_BEDS = {'ICU': 20, 'Cardiology': 40, 'General Surgery': 60, 'Infectious Diseases': 40, 'ER': 40}
DEPARTMENTS = list(TOTAL_BEDS.keys())

# Simulated occupancy rates (initially set for demonstration)
OCCUPANCY_RATES = {dept: 0.95 for dept in DEPARTMENTS}  # 95% occupancy initially
OCCUPIED_BEDS = {dept: int(rate * TOTAL_BEDS[dept]) for dept, rate in OCCUPANCY_RATES.items()}

# --- Mock Functions for Bed Allocation and Wait Time ---
def compute_occupancy_rate(dept):
    """Return the current occupancy rate for a department."""
    return OCCUPANCY_RATES.get(dept, 0.0)

def update_occupancy(dept, delta):
    """Update the occupancy rate for a department."""
    current_occupied = OCCUPIED_BEDS.get(dept, 0)
    new_occupied = max(0, min(TOTAL_BEDS[dept], current_occupied + delta))
    OCCUPIED_BEDS[dept] = new_occupied
    OCCUPANCY_RATES[dept] = new_occupied / TOTAL_BEDS[dept]

def predict_bed_availability(patients):
    """
    Predict the number of days until a bed is available for each patient.
    Prioritize based on severity and department.
    """
    bed_predictions = []
    for patient in patients:
        dept = patient['Department']
        severity = patient['Severity']
        current_occupancy = compute_occupancy_rate(dept)
        occupied_beds = OCCUPIED_BEDS[dept]
        available_beds = TOTAL_BEDS[dept] - occupied_beds

        # Base wait time in days (before prioritization)
        if available_beds > 0:
            days = 0  # Bed available immediately
        else:
            # Simulate discharge rate: assume 10% of beds free up each day
            discharge_rate_per_day = 0.1
            beds_needed = 1  # Each patient needs 1 bed
            days_to_free = beds_needed / (discharge_rate_per_day * TOTAL_BEDS[dept])
            days = max(0.1, days_to_free)  # At least 0.1 days (2.4 hours)

        # Adjust wait time based on severity and department
        if severity == 'Severe':
            days *= 0.2  # Reduce wait time significantly for severe cases (e.g., 20% of base wait)
        elif severity == 'Moderate':
            days *= 0.5  # Halve the wait time for moderate cases
        # Low severity keeps the base wait time

        # Prioritize ER and ICU
        if dept in ['ER', 'ICU']:
            days *= 0.3  # Further reduce wait time for ER/ICU (e.g., 30% of adjusted wait)

        # Ensure wait time is realistic (e.g., max 1 day for severe cases)
        if severity == 'Severe':
            days = min(days, 0.25)  # Max 6 hours (0.25 days) for severe cases
        else:
            days = min(days, 1.0)  # Max 1 day for other cases

        bed_predictions.append(days)

        # Simulate admitting the patient (for simplicity in this mock)
        if available_beds > 0:
            update_occupancy(dept, 1)  # Occupy a bed

    return bed_predictions

def predict_wait_time(patients):
    """
    Predict the wait time in hours for each patient to get a bed.
    Based on bed availability prediction, adjusted for severity and department.
    """
    bed_predictions = predict_bed_availability(patients)  # Use bed predictions as a base
    wait_predictions = []
    for idx, (days, patient) in enumerate(zip(bed_predictions, patients)):
        # Convert days to hours
        hours = days * 24

        # Further adjust based on severity and department (already factored in bed predictions, but fine-tune here)
        severity = patient['Severity']
        dept = patient['Department']

        # Ensure severe cases wait no more than 4 hours
        if severity == 'Severe':
            hours = min(hours, 4.0)
        elif severity == 'Moderate':
            hours = min(hours, 12.0)
        else:
            hours = min(hours, 24.0)

        # ER and ICU should have even shorter wait times
        if dept in ['ER', 'ICU']:
            hours = min(hours, 2.0 if severity == 'Severe' else 6.0)

        wait_predictions.append(hours)

    return wait_predictions

def simulate_discharges(patients):
    """
    Simulate discharges to free up beds. Assume shorter stays for less severe cases.
    """
    discharge_alerts = []
    for idx, patient in enumerate(patients):
        severity = patient['Severity']
        length_of_stay = patient['LengthOfStay']
        # Adjust discharge time based on severity
        if severity == 'Severe':
            days_until_discharge = length_of_stay * 0.5  # Severe cases may stay longer, but assume some improvement
        else:
            days_until_discharge = length_of_stay * 0.3  # Moderate/Low cases discharge faster

        # Cap discharge time to be more realistic
        days_until_discharge = min(days_until_discharge, 2.0)  # Max 2 days until discharge
        if days_until_discharge <= 1.0:  # Notify if discharge is within 1 day
            discharge_alerts.append((idx + 1, patient['Department'], days_until_discharge))

        # Simulate freeing up a bed
        if days_until_discharge <= 0.5:  # If discharging within 12 hours, free up a bed
            update_occupancy(patient['Department'], -1)

    return discharge_alerts

# Mock other functions for completeness (simplified)
def predict_hospital_inventory(patients):
    total_inventory = {'Oxygen Tanks': 50.0, 'Medications': 200.0}
    inventory_alerts = []
    inventory_trends = ['Oxygen Tanks: Stable', 'Medications: Increasing']
    if total_inventory['Oxygen Tanks'] < 60:
        inventory_alerts.append("Low on Oxygen Tanks! Current stock: 50.0 units.")
    return total_inventory, inventory_alerts, inventory_trends

def predict_staff_allocation(patients):
    staff_suggestions = {}
    for patient in patients:
        dept = patient['Department']
        # Only allocate staff to actual departments
        if dept in DEPARTMENTS:
            if dept in staff_suggestions:
                staff_suggestions[dept] += 2  # Increment nurse count for this department
            else:
                staff_suggestions[dept] = 2  # Initial assignment of 2 nurses
    
    # Format suggestions with consistent phrasing
    formatted_suggestions = []
    for dept, num_nurses in staff_suggestions.items():
        formatted_suggestions.append(f"Assign {num_nurses} additional nurses to {dept} for the next shift.")
    
    return formatted_suggestions

# --- Initialize Text-to-Speech Engine ---
def initialize_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)  # Default optimized rate
        engine.setProperty('volume', 0.9)
        return engine
    except Exception as e:
        print(f"Warning: Failed to initialize text-to-speech engine: {e}")
        print("Voice responses will be disabled.")
        return None

# --- Initialize Speech Recognizer ---
def initialize_speech_recognizer():
    try:
        recognizer = sr.Recognizer()
        return recognizer
    except Exception as e:
        print(f"Warning: Failed to initialize speech recognizer: {e}")
        print("Voice input will be disabled.")
        return None

# --- Prediction Inputs ---
new_patients = [
    {
        'Department': 'Cardiology', 'DiagnosisType': 'Heart Attack', 'Severity': 'Severe',
        'LengthOfStay': 10, 'Age': 65, 'Comorbidities': 2, 'OccupancyRate': compute_occupancy_rate('Cardiology'),
        'AdmissionDate': datetime.now(),
        'DayOfWeek': datetime.now().weekday(),
        'PatientLoad': compute_occupancy_rate('Cardiology') * TOTAL_BEDS['Cardiology'],
        'Shift': 'Day'
    },
    {
        'Department': 'General Surgery', 'DiagnosisType': 'Appendectomy', 'Severity': 'Moderate',
        'LengthOfStay': 5, 'Age': 45, 'Comorbidities': 1, 'OccupancyRate': compute_occupancy_rate('General Surgery'),
        'AdmissionDate': datetime.now(),
        'DayOfWeek': datetime.now().weekday(),
        'PatientLoad': compute_occupancy_rate('General Surgery') * TOTAL_BEDS['General Surgery'],
        'Shift': 'Day'
    },
    {
        'Department': 'Infectious Diseases', 'DiagnosisType': 'Respiratory Infection', 'Severity': 'Severe',
        'LengthOfStay': 12, 'Age': 30, 'Comorbidities': 0, 'OccupancyRate': compute_occupancy_rate('Infectious Diseases'),
        'AdmissionDate': datetime.now(),
        'DayOfWeek': datetime.now().weekday(),
        'PatientLoad': compute_occupancy_rate('Infectious Diseases') * TOTAL_BEDS['Infectious Diseases'],
        'Shift': 'Day'
    }
]

# --- Generate Predictions ---
try:
    bed_predictions = predict_bed_availability(new_patients)
    wait_predictions = predict_wait_time(new_patients)
    total_inventory, inventory_alerts, inventory_trends = predict_hospital_inventory(new_patients)
    staff_suggestions = predict_staff_allocation(new_patients)
except Exception as e:
    print(f"Error generating predictions: {e}")
    raise SystemExit("Exiting due to prediction generation failure.")

# --- Simulate Discharges ---
discharge_alerts = simulate_discharges(new_patients)

# --- Patient Risk Assessment ---
def assess_patient_risk(patients):
    risk_assessments = []
    for idx, patient in enumerate(patients):
        severity_score = 10 if patient['Severity'] == 'Severe' else 5 if patient['Severity'] == 'Moderate' else 1
        age_score = patient['Age'] / 10  # Higher age increases risk
        comorbidity_score = patient['Comorbidities'] * 2  # Each comorbidity adds risk
        total_risk = severity_score + age_score + comorbidity_score
        risk_level = 'High' if total_risk >= 15 else 'Medium' if total_risk >= 10 else 'Low'
        risk_assessments.append((idx + 1, patient['Department'], risk_level, total_risk))
    return risk_assessments

risk_assessments = assess_patient_risk(new_patients)

# --- Print Predictions ---
for idx, (days, patient) in enumerate(zip(bed_predictions, new_patients)):
    print(f"Patient {idx+1} ({patient['Department']}): Predicted bed availability in {days:.1f} days")

for idx, hours in enumerate(wait_predictions):
    print(f"Patient {idx+1}: Predicted wait time of {hours:.1f} hours")

print("\nHospital-Wide Inventory Needs:")
for item, value in total_inventory.items():
    print(f"  {item}: {value:.1f} units")

print("\nInventory Alerts:")
if inventory_alerts:
    for alert in inventory_alerts:
        print(f"  {alert}")
else:
    print("  No inventory alerts at this time.")

print("\nInventory Trends (Last 7 Days):")
for trend in inventory_trends:
    print(f"  {trend}")

print("\nDischarge Alerts:")
if discharge_alerts:
    for patient_id, dept, days in discharge_alerts:
        print(f"Patient {patient_id} in {dept} is expected to be discharged in {days:.1f} day(s). Prepare for bed turnover.")
else:
    print("  No patients are nearing discharge.")

# --- Patient Prioritization ---
def prioritize_patients(bed_predictions, wait_predictions, patients):
    prioritization = []
    for idx, (days, hours, patient) in enumerate(zip(bed_predictions, wait_predictions, patients)):
        # Adjusted priority score to give more weight to severity and wait time
        severity_weight = 20 if patient['Severity'] == 'Severe' else 10 if patient['Severity'] == 'Moderate' else 5
        dept_priority = 10 if patient['Department'] in ['ER', 'ICU'] else 0  # Higher priority for ER/ICU
        priority_score = severity_weight + dept_priority + hours
        prioritization.append((idx + 1, priority_score, patient['Department'], hours, days))
    
    prioritization.sort(key=lambda x: x[1], reverse=True)
    return prioritization

priority_list = prioritize_patients(bed_predictions, wait_predictions, new_patients)
print("\nPatient Prioritization for Bed Allocation:")
for rank, (patient_id, score, dept, wait, stay) in enumerate(priority_list, 1):
    print(f"Rank {rank}: Patient {patient_id} in {dept} (Wait: {wait:.1f} hours, Stay: {stay:.1f} days, Priority Score: {score:.1f})")

print("\nPatient Risk Assessments:")
for patient_id, dept, risk_level, risk_score in risk_assessments:
    print(f"Patient {patient_id} in {dept}: Risk Level - {risk_level} (Score: {risk_score:.1f})")

print("\nStaff Allocation Predictions:")
for suggestion in staff_suggestions:
    print(f"  {suggestion}")

# --- Capacity Forecast (Next 24 Hours) ---
def forecast_capacity(bed_predictions, patients):
    forecast = {dept: compute_occupancy_rate(dept) for dept in DEPARTMENTS}
    discharges = simulate_discharges(patients)
    for _, dept, _ in discharges:
        forecast[dept] = compute_occupancy_rate(dept)
    
    forecast_messages = []
    for dept, rate in forecast.items():
        forecast_messages.append(f"{dept} capacity in the next 24 hours: {rate:.2f} occupancy rate ({int(rate * TOTAL_BEDS[dept])}/{TOTAL_BEDS[dept]} beds occupied).")
    return forecast_messages

capacity_forecast = forecast_capacity(bed_predictions, new_patients)
print("\nCapacity Forecast (Next 24 Hours):")
for message in capacity_forecast:
    print(f"  {message}")

# --- Generate Report with GPT-2 ---
def generate_hospital_report(bed_predictions, wait_predictions, total_inventory, patients, priority_list, staff_suggestions, capacity_forecast, risk_assessments):    # Fallback report in case GPT-2 fails
    fallback_report = (
        "Hospital Resource Allocation Report:\n"
        "This report provides predictions for bed allocation, wait times, inventory needs, staff allocation, capacity forecasts, and patient risk assessments for a hospital.\n\n"
        "Current Scenario:\n"
    )
    for idx, patient in enumerate(patients):
        fallback_report += f"Patient {idx+1}: {patient['DiagnosisType']} in {patient['Department']} (Severity: {patient['Severity']})\n"
    
    fallback_report += "\nBed Allocation Predictions:\n"
    for idx, (days, patient) in enumerate(zip(bed_predictions, patients)):
        fallback_report += f"Patient {idx+1} in {patient['Department']} is expected to have a bed available in {days:.1f} days.\n"
    
    fallback_report += "\nWait Time Predictions:\n"
    for idx, hours in enumerate(wait_predictions):
        fallback_report += f"Patient {idx+1} is expected to wait {hours:.1f} hours for a bed.\n"
    
    fallback_report += "\nHospital-Wide Inventory Needs:\n"
    for item, value in total_inventory.items():
        fallback_report += f"{item}: {value:.1f} units\n"
    
    fallback_report += "\nPatient Prioritization for Bed Allocation:\n"
    for rank, (patient_id, score, dept, wait, stay) in enumerate(priority_list, 1):
        fallback_report += f"Rank {rank}: Patient {patient_id} in {dept} (Wait: {wait:.1f} hours, Stay: {stay:.1f} days)\n"
    
    fallback_report += "\nPatient Risk Assessments:\n"
    for patient_id, dept, risk_level, risk_score in risk_assessments:
        fallback_report += f"Patient {patient_id} in {dept}: Risk Level - {risk_level} (Score: {risk_score:.1f})\n"
    
    fallback_report += "\nStaff Allocation Predictions:\n"
    for suggestion in staff_suggestions:
        fallback_report += f"{suggestion}\n"
    
    fallback_report += "\nCapacity Forecast (Next 24 Hours):\n"
    for message in capacity_forecast:
        fallback_report += f"{message}\n"
    
    fallback_report += "\nActionable Insights:\n"
    fallback_report += "Prioritize bed allocation for severe cases and ensure inventory is restocked.\n"

    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
    except Exception as e:
        print(f"Error loading GPT-2 model: {e}")
        return fallback_report  # Use fallback if GPT-2 fails

    prompt = (
        "Hospital Resource Allocation Report:\n"
        "This report provides predictions for bed allocation, wait times, inventory needs, staff allocation, capacity forecasts, and patient risk assessments for a hospital. "
        "Focus strictly on the provided data and predictions. "
        "Avoid terms like 'survey', 'study', 'insurance', 'inpatient', or external references. "
        "Provide concise actionable insights in complete sentences, avoiding placeholders like 'Patient 1:', 'Patient 2:'.\n\n"
        "Current Scenario:\n"
    )
    for idx, patient in enumerate(patients):
        prompt += f"Patient {idx+1}: {patient['DiagnosisType']} in {patient['Department']} (Severity: {patient['Severity']})\n"
    
    prompt += "\nBed Allocation Predictions:\n"
    for idx, (days, patient) in enumerate(zip(bed_predictions, patients)):
        prompt += f"Patient {idx+1} in {patient['Department']} is expected to have a bed available in {days:.1f} days.\n"
    
    prompt += "\nWait Time Predictions:\n"
    for idx, hours in enumerate(wait_predictions):
        prompt += f"Patient {idx+1} is expected to wait {hours:.1f} hours for a bed.\n"
    
    prompt += "\nHospital-Wide Inventory Needs:\n"
    for item, value in total_inventory.items():
        prompt += f"{item}: {value:.1f} units\n"
    
    prompt += "\nPatient Prioritization for Bed Allocation:\n"
    for rank, (patient_id, score, dept, wait, stay) in enumerate(priority_list, 1):
        prompt += f"Rank {rank}: Patient {patient_id} in {dept} (Wait: {wait:.1f} hours, Stay: {stay:.1f} days)\n"
    
    prompt += "\nPatient Risk Assessments:\n"
    for patient_id, dept, risk_level, risk_score in risk_assessments:
        prompt += f"Patient {patient_id} in {dept}: Risk Level - {risk_level} (Score: {risk_score:.1f})\n"
    
    prompt += "\nStaff Allocation Predictions:\n"
    for suggestion in staff_suggestions:
        prompt += f"{suggestion}\n"
    
    prompt += "\nCapacity Forecast (Next 24 Hours):\n"
    for message in capacity_forecast:
        prompt += f"{message}\n"
    
    prompt += (
        "\nActionable Insights:\n"
        "Example: Prioritize bed allocation for the patient in Infectious Diseases due to a longer stay. Restock Oxygen Tanks and Medications to meet high demand.\n"
    )

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    try:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            num_return_sequences=1,
            no_repeat_ngram_size=6,
            do_sample=True,
            top_k=40,
            top_p=0.8,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id
        )
    except Exception as e:
        print(f"Error generating report with GPT-2: {e}")
        return fallback_report  # Use fallback if generation fails

    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    lines = report.split('\n')
    filtered_lines = [
        line for line in lines 
        if not any(keyword in line.lower() for keyword in ['survey', 'study', 'insurance', 'inpatient', 'http'])
        and not line.strip().endswith(":")
        and line.strip()
    ]
    return '\n'.join(filtered_lines)

# --- AI Agent for Automated Responses ---
def ai_agent_response(query, bed_predictions, wait_predictions, total_inventory, patients, priority_list, inventory_alerts, inventory_trends, staff_suggestions, capacity_forecast, discharge_alerts, risk_assessments):
    try:
        query = query.lower()

        # Define intents using regex patterns for better matching
        intents = {
            "bed_availability": r"\b(bed|availability|stay|length)\b",
            "wait_time": r"\b(wait|time|how long|delay)\b",
            "inventory": r"\b(inventory|stock|supplies|oxygen|medications)\b",
            "general_status": r"\b(status|hospital|report|general)\b",
            "priority": r"\b(priority|prioritize|urgent)\b",
            "emergency": r"\b(emergency|urgent|critical)\b",
            "trends": r"\b(trend|history|historical|past)\b",
            "staffing": r"\b(staff|staffing|personnel)\b",
            "capacity": r"\b(capacity|forecast|occupancy|space)\b",
            "discharge": r"\b(discharge|release|leaving)\b",
            "risk": r"\b(risk|patient risk|danger)\b",
            "switch_to_text": r"\b(switch to text|text mode)\b",
            "switch_to_voice": r"\b(switch to voice|voice mode)\b",
            "queue_status": r"\b(queue|position|status|where am i|how many ahead)\b",
            "arrival_time": r"\b(arrive|when to come|when should i)\b"
        }

        # Match intent based on regex patterns
        matched_intent = None
        for intent, pattern in intents.items():
            if re.search(pattern, query):
                matched_intent = intent
                break

        # Generate response based on intent
        if matched_intent == "bed_availability":
            response = "Here are the predicted bed availabilities:\n"
            for idx, (days, patient) in enumerate(zip(bed_predictions, patients)):
                response += f"Patient {idx+1} in {patient['Department']} is expected to have a bed available in {days:.1f} days.\n"
            return response

        elif matched_intent == "wait_time":
            response = "Here are the predicted wait times:\n"
            for idx, hours in enumerate(wait_predictions):
                response += f"Patient {idx+1} is expected to wait {hours:.1f} hours for a bed.\n"
            return response

        elif matched_intent == "inventory":
            response = "Here are the hospital-wide inventory needs:\n"
            for item, value in total_inventory.items():
                response += f"{item}: {value:.1f} units\n"
            if inventory_alerts:
                response += "\nInventory Alerts:\n"
                for alert in inventory_alerts:
                    response += f"{alert}\n"
            return response

        elif matched_intent == "general_status":
            return generate_hospital_report(bed_predictions, wait_predictions, total_inventory, patients, priority_list, staff_suggestions, capacity_forecast, risk_assessments)

        elif matched_intent == "priority":
            response = "Patient Prioritization for Bed Allocation:\n"
            for rank, (patient_id, score, dept, wait, stay) in enumerate(priority_list, 1):
                response += f"Rank {rank}: Patient {patient_id} in {dept} (Wait: {wait:.1f} hours, Stay: {stay:.1f} days, Priority Score: {score:.1f})\n"
            return response

        elif matched_intent == "emergency":
            dept_match = re.search(r"\b(in|at)\s+(\w+)", query)
            if dept_match:
                dept = dept_match.group(2).title()
                if dept in DEPARTMENTS:
                    try:
                        update_occupancy(dept, -1)  # Free up a bed immediately for emergency
                        response = f"Emergency handled in {dept}. Prioritized bed allocation by freeing up resources.\n"
                        response += f"Current occupancy rate in {dept}: {compute_occupancy_rate(dept):.2f}\n"
                        response += "Please check the updated hospital status for new predictions."
                    except Exception as e:
                        response = f"Error handling emergency in {dept}: {e}"
                else:
                    response = f"Department '{dept}' not recognized. Valid departments are: {', '.join(DEPARTMENTS)}."
            else:
                response = "Please specify the department for the emergency (e.g., 'Emergency in ICU')."
            return response

        elif matched_intent == "trends":
            response = "Inventory Trends (Last 7 Days):\n"
            for trend in inventory_trends:
                response += f"{trend}\n"
            return response

        elif matched_intent == "staffing":
            response = "Staff Allocation Predictions:\n"
            for suggestion in staff_suggestions:
                response += f"{suggestion}\n"
            return response

        elif matched_intent == "capacity":
            response = "Capacity Forecast (Next 24 Hours):\n"
            for message in capacity_forecast:
                response += f"{message}\n"
            return response

        elif matched_intent == "discharge":
            if discharge_alerts:
                response = "Discharge Alerts:\n"
                for patient_id, dept, days in discharge_alerts:
                    response += f"Patient {patient_id} in {dept} is expected to be discharged in {days:.1f} day(s). Prepare for bed turnover.\n"
            else:
                response = "No patients are nearing discharge at this time."
            return response

        elif matched_intent == "risk":
            response = "Patient Risk Assessments:\n"
            for patient_id, dept, risk_level, risk_score in risk_assessments:
                response += f"Patient {patient_id} in {dept}: Risk Level - {risk_level} (Score: {risk_score:.1f})\n"
            return response

        elif matched_intent == "switch_to_text":
            return "SWITCH_TO_TEXT"

        elif matched_intent == "switch_to_voice":
            return "SWITCH_TO_VOICE"

        elif matched_intent == "queue_status":
            patient_id_match = re.search(r"\bpatient\s*(\d+)\b", query)
            dept_match = re.search(r"\b(in|at)\s+(\w+)", query)
            if patient_id_match and dept_match:
                patient_id = int(patient_id_match.group(1))
                dept = dept_match.group(2).title()
                if dept in DEPARTMENTS:
                    return check_queue_status(patient_id, dept)
                else:
                    return f"Department '{dept}' not recognized. Valid departments are: {', '.join(DEPARTMENTS)}."
            else:
                return "Please specify your patient ID and department (e.g., 'What is the queue status for Patient 1 in Cardiology?')."

        elif matched_intent == "arrival_time":
            patient_id_match = re.search(r"\bpatient\s*(\d+)\b", query)
            if patient_id_match:
                patient_id = int(patient_id_match.group(1))
                patient_info = next((p for p in appointments if p['PatientID'] == patient_id), None)
                if patient_info:
                    arrival_time = patient_info['AppointmentTime'] - timedelta(minutes=10)
                    return f"Arrive at {arrival_time.strftime('%Y-%m-%d %H:%M:%S')} for your appointment."
                else:
                    return "Patient not found. Please book an appointment first."
            else:
                return "Please specify your patient ID (e.g., 'When should Patient 1 arrive?')."

        else:
            return "I'm sorry, I didn't understand your query. Please ask about bed availability, wait times, inventory needs, the general hospital status, patient prioritization, historical inventory trends, staffing needs, capacity forecasts, discharge alerts, patient risk assessments, queue status, or arrival time."

    except Exception as e:
        return f"Error processing query: {e}"

# --- Appointment and Queue Management ---
appointments = []
queue_position = {dept: [] for dept in DEPARTMENTS}

def book_appointment(patient_info):
    wait_time = predict_wait_time([patient_info])[0]
    appointment_time = datetime.now() + timedelta(hours=wait_time)
    patient_info['AppointmentTime'] = appointment_time
    patient_info['WaitTime'] = wait_time
    patient_info['PatientID'] = len(appointments) + 1
    
    appointments.append(patient_info)
    queue_position[patient_info['Department']].append(patient_info['PatientID'])
    
    queue_pos = queue_position[patient_info['Department']].index(patient_info['PatientID']) + 1
    arrival_time = appointment_time - timedelta(minutes=10)
    
    return {
        'PatientID': patient_info['PatientID'],
        'Department': patient_info['Department'],
        'AppointmentTime': appointment_time.strftime('%Y-%m-%d %H:%M:%S'),
        'WaitTime': wait_time,
        'QueuePosition': queue_pos,
        'ArrivalTime': arrival_time.strftime('%Y-%m-%d %H:%M:%S')
    }

def check_queue_status(patient_id, department):
    if not queue_position[department] or patient_id not in queue_position[department]:
        return "Patient not found in the queue for this department."
    
    queue_pos = queue_position[department].index(patient_id) + 1
    patient_info = next((p for p in appointments if p['PatientID'] == patient_id), None)
    if patient_info:
        wait_time = patient_info['WaitTime']
        return f"You are {queue_pos} in the {department} queue. Expected wait time: {wait_time:.1f} hours."
    return "Error retrieving wait time."

# --- Simulated Call Interface with Simplified Voice Support ---
def simulate_call_interface():
    # Initialize speech recognizer and text-to-speech engine
    recognizer = initialize_speech_recognizer()
    tts_engine = initialize_tts()
    use_voice = recognizer is not None and tts_engine is not None

    print("\n--- Welcome to the Hospital Resource AI Agent (Patient Edition) ---")
    welcome_message = (
        "This system helps patients book appointments, check queue status, and know when to arrive. "
        "You can also get updates on bed availability, wait times, and hospital status. "
        "Interact with me via voice or text. To use voice, speak your query when prompted. "
        "If voice is not available, type your query. Type or say 'EXIT' to end."
    )
    print(welcome_message)
    if use_voice and tts_engine:
        tts_engine.say(welcome_message)
        tts_engine.runAndWait()

    while True:
        try:
            if use_voice and recognizer:
                print("\nListening for your query (speak now)...")
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                try:
                    query = recognizer.recognize_google(audio)
                    print(f"You said: {query}")
                except sr.UnknownValueError:
                    retry_message = "Sorry, I couldn't understand your speech. Please repeat your query or type it."
                    print(retry_message)
                    if tts_engine:
                        tts_engine.say(retry_message)
                        tts_engine.runAndWait()
                    query = input("Your query (type): ")
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}. Falling back to text input.")
                    query = input("Your query (type): ")
            else:
                query = input("\nYour query (type): ")

            if query.upper() == 'EXIT':
                goodbye_message = "Thank you for using the Hospital Resource AI Agent. Goodbye!"
                print(goodbye_message)
                if use_voice and tts_engine:
                    tts_engine.say(goodbye_message)
                    tts_engine.runAndWait()
                break
            
            response = ai_agent_response(query, bed_predictions, wait_predictions, total_inventory, new_patients, priority_list, inventory_alerts, inventory_trends, staff_suggestions, capacity_forecast, discharge_alerts, risk_assessments)
            
            if response == "SWITCH_TO_TEXT":
                use_voice = False
                response = "Switched to text input mode. Please type your query."
            elif response == "SWITCH_TO_VOICE":
                if recognizer is None or tts_engine is None:
                    response = "Voice mode is unavailable due to initialization failure. Please continue using text input."
                else:
                    use_voice = True
                    response = "Switched to voice input mode. Please speak your query when prompted."

            # Optimize TTS for long responses by splitting into chunks
            print("\nAI Agent Response:")
            print(response)
            if use_voice and tts_engine:
                lines = response.split('\n')
                for line in lines:
                    if line.strip():
                        tts_engine.say(line)
                        tts_engine.runAndWait()
        except Exception as e:
            error_message = f"An error occurred: {e}. Please try again or type 'EXIT' to end the call."
            print(error_message)
            if use_voice and tts_engine:
                tts_engine.say(error_message)
                tts_engine.runAndWait()

# --- Generate and Print Report ---
report = generate_hospital_report(bed_predictions, wait_predictions, total_inventory, new_patients, priority_list, staff_suggestions, capacity_forecast, risk_assessments)
print("\nGenerated Report:")
print(report)

# --- Run the Call Interface ---
if __name__ == "__main__":
    simulate_call_interface()