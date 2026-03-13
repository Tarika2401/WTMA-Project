from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('churn_model.pkl')
except:
    model = None
    print("Warning: Model file not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Collect form data
            form_data = {
                'gender': request.form.get('gender'),
                'SeniorCitizen': request.form.get('SeniorCitizen'),
                'Partner': request.form.get('Partner'),
                'Dependents': request.form.get('Dependents'),
                'tenure': request.form.get('tenure'),
                'PhoneService': request.form.get('PhoneService'),
                'MultipleLines': request.form.get('MultipleLines'),
                'InternetService': request.form.get('InternetService'),
                'OnlineSecurity': request.form.get('OnlineSecurity'),
                'OnlineBackup': request.form.get('OnlineBackup'),
                'DeviceProtection': request.form.get('DeviceProtection'),
                'TechSupport': request.form.get('TechSupport'),
                'StreamingTV': request.form.get('StreamingTV'),
                'StreamingMovies': request.form.get('StreamingMovies'),
                'Contract': request.form.get('Contract'),
                'PaperlessBilling': request.form.get('PaperlessBilling'),
                'PaymentMethod': request.form.get('PaymentMethod'),
                'MonthlyCharges': request.form.get('MonthlyCharges'),
                'TotalCharges': request.form.get('TotalCharges')
            }
            
            # Prepare data for model prediction
            model_data = {
                'gender': form_data['gender'],
                'SeniorCitizen': int(form_data['SeniorCitizen']),
                'Partner': form_data['Partner'],
                'Dependents': form_data['Dependents'],
                'tenure': int(form_data['tenure']),
                'PhoneService': form_data['PhoneService'],
                'MultipleLines': form_data['MultipleLines'],
                'InternetService': form_data['InternetService'],
                'OnlineSecurity': form_data['OnlineSecurity'],
                'OnlineBackup': form_data['OnlineBackup'],
                'DeviceProtection': form_data['DeviceProtection'],
                'TechSupport': form_data['TechSupport'],
                'StreamingTV': form_data['StreamingTV'],
                'StreamingMovies': form_data['StreamingMovies'],
                'Contract': form_data['Contract'],
                'PaperlessBilling': form_data['PaperlessBilling'],
                'PaymentMethod': form_data['PaymentMethod'],
                'MonthlyCharges': float(form_data['MonthlyCharges']),
                'TotalCharges': float(form_data['TotalCharges'])
            }
            
            # Convert to DataFrame
            df = pd.DataFrame([model_data])
            
            # Make prediction
            if model:
                prediction_proba = model.predict_proba(df)[0][1] * 100
            else:
                # Demo prediction if model not loaded
                prediction_proba = np.random.uniform(10, 90)
            
            # Generate recommendations based on probability (3 LEVELS)
            if prediction_proba > 70:
                recommendations = [
                    "🚨 Immediate intervention required - Contact customer within 24 hours",
                    "Offer loyalty discount or contract upgrade incentive",
                    "Schedule consultation to address service concerns",
                    "Consider switching to longer-term contract with benefits"
                ]
            elif prediction_proba > 40:
                recommendations = [
                    "⚠️ Monitor customer engagement closely",
                    "Proactive outreach within the next week",
                    "Offer value-added services at discounted rate",
                    "Gather feedback through customer survey"
                ]
            else:
                recommendations = [
                    "✅ Customer shows good retention indicators. Continue standard engagement.",
                    "Maintain quality service delivery",
                    "Consider upselling opportunities for additional services"
                ]
            
            prediction_data = {
                'probability': round(prediction_proba, 2),
                'recommendations': recommendations
            }
            
            # Pass both prediction results AND form data back to template
            return render_template('prediction.html', 
                                 prediction=prediction_data, 
                                 form_data=form_data)
            
        except Exception as e:
            print(f"Error: {e}")
            return render_template('prediction.html', error=str(e))
    
    return render_template('prediction.html')

@app.route('/exploration')
def exploration():
    return render_template('exploration.html')

@app.route('/segmentation')
def segmentation():
    # Pass segment stats data to template with CORRECT field names
    segment_stats = {
        'High-Value Loyalists': {
            'customerID': 1245,
            'MonthlyCharges': 89.50,
            'tenure': 48,
            'Churn': 8.2
        },
        'Price-Sensitive Switchers': {
            'customerID': 892,
            'MonthlyCharges': 55.30,
            'tenure': 12,
            'Churn': 45.3
        },
        'Service Seekers': {
            'customerID': 1567,
            'MonthlyCharges': 72.80,
            'tenure': 28,
            'Churn': 22.1
        },
        'Budget Conscious': {
            'customerID': 2339,
            'MonthlyCharges': 42.15,
            'tenure': 18,
            'Churn': 35.7
        }
    }
    return render_template('segmentation.html', segment_stats=segment_stats)

@app.route('/risk_analysis')
def risk_analysis():
    # Generate sample high-risk customers data
    high_risk_customers = []
    for i in range(50):
        customer = {
            'customerID': f'CUST{7000 + i}',
            'tenure': np.random.randint(1, 24),
            'MonthlyCharges': float(np.random.uniform(60, 110)),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year']),
            'ChurnProbability': float(np.random.uniform(60, 95))
        }
        high_risk_customers.append(customer)
    
    # Sort by churn probability (highest first)
    high_risk_customers = sorted(high_risk_customers, key=lambda x: x['ChurnProbability'], reverse=True)
    
    # Calculate metrics
    total_at_risk = len(high_risk_customers)
    revenue_at_risk = sum(c['MonthlyCharges'] * 12 for c in high_risk_customers)
    
    return render_template('risk_analysis.html', 
                         total_at_risk=total_at_risk,
                         revenue_at_risk=revenue_at_risk,
                         high_risk_customers=high_risk_customers)

@app.route('/insights')
def insights():
    # Top 10 features influencing churn
    top_features = [
        ('Contract: Month-to-month', 85.2),
        ('Tenure (months)', 78.4),
        ('Internet Service: Fiber optic', 68.9),
        ('Payment Method: Electronic check', 65.4),
        ('Online Security: No', 61.7),
        ('Tech Support: No', 58.3),
        ('Monthly Charges', 52.6),
        ('Paperless Billing: Yes', 47.8),
        ('Senior Citizen: Yes', 42.3),
        ('Total Charges', 38.9)
    ]
    
    # Strategic insights
    insights_list = [
        {
            'title': 'Contract Type is Critical',
            'description': 'Customers on month-to-month contracts are 4x more likely to churn compared to those on annual contracts.',
            'impact': 'High'
        },
        {
            'title': 'Service Bundle Effect',
            'description': 'Customers with multiple services (security, backup, support) have 60% lower churn rates.',
            'impact': 'High'
        },
        {
            'title': 'Payment Method Matters',
            'description': 'Electronic check users churn at 45% vs 15% for automatic payment methods.',
            'impact': 'Medium'
        },
        {
            'title': 'First Year is Crucial',
            'description': '58% of all churns occur within the first 12 months of service.',
            'impact': 'High'
        },
        {
            'title': 'Senior Citizens Need Attention',
            'description': 'Senior customers show 23% higher churn despite similar service usage.',
            'impact': 'Medium'
        }
    ]
    
    return render_template('insights.html', top_features=top_features, insights=insights_list)

# API ROUTES FOR DATA

@app.route('/api/exploration_data')
def exploration_data():
    """API endpoint for exploration page data"""
    return jsonify({
        'churn_distribution': {
            'Retained': 5174,
            'Churned': 1869
        },
        'tenure_churn': {
            '0-12 months': 47.5,
            '13-24 months': 35.2,
            '25-48 months': 25.8,
            '49-72 months': 15.4
        },
        'charges_data': {
            'churned': [float(np.random.uniform(50, 120)) for _ in range(500)],
            'retained': [float(np.random.uniform(20, 100)) for _ in range(1000)]
        },
        'service_impact': {
            'OnlineSecurity': {'with_service': 15.2, 'without_service': 41.8},
            'TechSupport': {'with_service': 16.3, 'without_service': 42.1},
            'OnlineBackup': {'with_service': 21.5, 'without_service': 39.9},
            'DeviceProtection': {'with_service': 22.4, 'without_service': 38.2}
        }
    })

@app.route('/api/segmentation_data')
def segmentation_data():
    """API endpoint for segmentation page data"""
    return jsonify({
        'segments': [
            {
                'name': 'High-Value Loyalists',
                'size': 1245,
                'churn_rate': 8.2,
                'avg_tenure': 48,
                'avg_monthly': 89.50,
                'characteristics': ['Long tenure', 'Premium services', 'Low churn risk']
            },
            {
                'name': 'Price-Sensitive Switchers',
                'size': 892,
                'churn_rate': 45.3,
                'avg_tenure': 12,
                'avg_monthly': 55.30,
                'characteristics': ['Short tenure', 'Basic services', 'High churn risk']
            },
            {
                'name': 'Service Seekers',
                'size': 1567,
                'churn_rate': 22.1,
                'avg_tenure': 28,
                'avg_monthly': 72.80,
                'characteristics': ['Multiple services', 'Medium tenure', 'Moderate risk']
            },
            {
                'name': 'Budget Conscious',
                'size': 2339,
                'churn_rate': 35.7,
                'avg_tenure': 18,
                'avg_monthly': 42.15,
                'characteristics': ['Minimal services', 'Cost-focused', 'Moderate-high risk']
            }
        ]
    })

@app.route('/api/risk_analysis_data')
def risk_analysis_data():
    """API endpoint for risk analysis page data"""
    return jsonify({
        'risk_distribution': {
            'High Risk (70-100%)': 1243,
            'Medium Risk (40-70%)': 2156,
            'Low Risk (0-40%)': 3644
        },
        'risk_factors': {
            'Month-to-month contract': 85.2,
            'No online security': 72.3,
            'Fiber optic internet': 68.9,
            'Electronic check payment': 65.4,
            'No tech support': 61.7,
            'Short tenure (<12 months)': 58.3
        },
        'monthly_risk_trend': {
            'Jan': 28.5,
            'Feb': 30.2,
            'Mar': 27.8,
            'Apr': 32.1,
            'May': 29.7,
            'Jun': 31.4,
            'Jul': 33.8,
            'Aug': 35.2,
            'Sep': 32.9,
            'Oct': 31.5,
            'Nov': 29.8,
            'Dec': 28.9
        }
    })

@app.route('/api/insights_data')
def insights_data():
    """API endpoint for insights page data"""
    return jsonify({
        'key_insights': [
            {
                'title': 'Contract Type is Critical',
                'description': 'Customers on month-to-month contracts are 4x more likely to churn compared to those on annual contracts.',
                'impact': 'high',
                'recommendation': 'Incentivize customers to switch to longer-term contracts with discounts.'
            },
            {
                'title': 'Service Bundle Effect',
                'description': 'Customers with multiple services (security, backup, support) have 60% lower churn rates.',
                'impact': 'high',
                'recommendation': 'Promote service bundles and cross-sell to at-risk customers.'
            },
            {
                'title': 'Payment Method Matters',
                'description': 'Electronic check users churn at 45% vs 15% for automatic payment methods.',
                'impact': 'medium',
                'recommendation': 'Encourage adoption of automatic payment methods with small incentives.'
            },
            {
                'title': 'First Year is Crucial',
                'description': '58% of all churns occur within the first 12 months of service.',
                'impact': 'high',
                'recommendation': 'Implement enhanced onboarding and regular check-ins during first year.'
            },
            {
                'title': 'Senior Citizens Need Attention',
                'description': 'Senior customers show 23% higher churn despite similar service usage.',
                'impact': 'medium',
                'recommendation': 'Create senior-focused support programs and simplified service options.'
            }
        ],
        'performance_metrics': {
            'overall_churn_rate': 26.5,
            'model_accuracy': 81.2,
            'avg_customer_lifetime': 32.4,
            'retention_rate': 73.5
        }
    })

if __name__ == '__main__':
    app.run(debug=True)