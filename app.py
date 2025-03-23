# app.py
from flask import Flask, request, jsonify, send_file, render_template
import os
from generate_report import ai_agent_response, predict_bed_availability, predict_wait_time, predict_hospital_inventory, predict_staff_allocation, simulate_discharges, assess_patient_risk, prioritize_patients, forecast_capacity, generate_hospital_report, new_patients, book_appointment
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='static')

# Generate predictions once at startup
bed_predictions = predict_bed_availability(new_patients)
wait_predictions = predict_wait_time(new_patients)
total_inventory, inventory_alerts, inventory_trends = predict_hospital_inventory(new_patients)
staff_suggestions = predict_staff_allocation(new_patients)
discharge_alerts = simulate_discharges(new_patients)
risk_assessments = assess_patient_risk(new_patients)
priority_list = prioritize_patients(bed_predictions, wait_predictions, new_patients)
capacity_forecast = forecast_capacity(bed_predictions, new_patients)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    department = data.get('department', '')
    
    # Append department to query if provided
    if department:
        query = f"{query} in {department}"
    
    response = ai_agent_response(
        query, bed_predictions, wait_predictions, total_inventory, new_patients,
        priority_list, inventory_alerts, inventory_trends, staff_suggestions,
        capacity_forecast, discharge_alerts, risk_assessments
    )
    return jsonify({'response': response})

@app.route('/api/generate_report', methods=['GET'])
def generate_report():
    report = generate_hospital_report(
        bed_predictions, wait_predictions, total_inventory, new_patients,
        priority_list, staff_suggestions, capacity_forecast, risk_assessments
    )
    
    # Save the report to a file
    report_filename = f"Hospital_Report_{datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    # Return the report content and the file for download
    return jsonify({
        'report': report,
        'download_url': f"/download/{report_filename}"
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(filename, as_attachment=True)

@app.route('/api/book_appointment', methods=['POST'])
def handle_book_appointment():
    data = request.get_json()
    patient_info = {
        'Department': data.get('department', ''),
        'DiagnosisType': data.get('diagnosis', ''),
        'Severity': data.get('severity', 'Moderate'),
        'LengthOfStay': int(data.get('lengthOfStay', 5)),
        'Age': int(data.get('age', 30)),
        'Comorbidities': int(data.get('comorbidities', 0)),
        'OccupancyRate': 0.95,  # Mock value
        'AdmissionDate': datetime.now(),
        'DayOfWeek': datetime.now().weekday(),
        'PatientLoad': 0,  # Mock value
        'Shift': 'Day'
    }
    appointment_details = book_appointment(patient_info)
    return jsonify(appointment_details)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)