import streamlit as st
import pandas as pd
import pickle
import os


# Page configuration
st.set_page_config(
   page_title="Patient Risk Management System",
   page_icon="🏥",
   layout="wide"
)


# Custom CSS for header styling
st.markdown("""
   <style>
   .main-header {
       font-size: 2.5rem;
       font-weight: bold;
       color: #1f77b4;
       text-align: center;
       padding: 1rem 0;
   }
   </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset(dataset_path="Health_Risk_Dataset.csv"):
   """Load the patient dataset."""
   if os.path.exists(dataset_path):
       df = pd.read_csv(dataset_path)
       # Convert Risk_Score to numeric, treating empty strings as NaN
       if 'Risk_Score' in df.columns:
           df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
       return df
   return pd.DataFrame()


@st.cache_resource
def load_model(model_path="trained_pipeline.pkl"):
   """Load the trained model."""
   if os.path.exists(model_path):
       try:
           with open(model_path, 'rb') as f:
               pipeline, scaler = pickle.load(f)
           return pipeline, scaler
       except Exception as e:
           st.error(f"Error loading model: {e}")
           return None, None
   return None, None


def calculate_risk_score(patient_data, pipeline, scaler):
   """Calculate risk score for a patient."""
   if pipeline is None or scaler is None:
       return None
  
   try:
       # Ensure patient_data is a DataFrame
       if not isinstance(patient_data, pd.DataFrame):
           patient_data = pd.DataFrame([patient_data])
      
       # Prepare features in correct order
       feature_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale',
                      'Systolic_BP', 'Heart_Rate', 'Temperature', 'Consciousness', 'On_Oxygen']
      
       # Ensure all required columns exist
       missing_cols = [col for col in feature_cols if col not in patient_data.columns]
       if missing_cols:
           raise ValueError(f"Missing required columns: {missing_cols}")
      
       # Extract only the feature columns as a DataFrame (pipeline needs column names)
       X = patient_data[feature_cols].copy()
      
       # Predict - pipeline expects DataFrame with column names
       raw_prediction = pipeline.predict(X)
       risk_score = scaler.transform(raw_prediction.reshape(-1, 1))[0][0]
       return round(risk_score, 2)
   except Exception as e:
       st.error(f"Error calculating risk score: {e}")
       import traceback
       st.error(f"Traceback: {traceback.format_exc()}")
       return None


def generate_patient_id(df):
   """Generate next available Patient ID."""
   if df.empty or 'Patient_ID' not in df.columns:
       return "P0001"
  
   existing_ids = df['Patient_ID'].astype(str).str.extract(r'P(\d+)')[0]
   if existing_ids.notna().any():
       existing_ids = existing_ids.dropna().astype(int)
       next_id = int(existing_ids.max()) + 1
   else:
       next_id = 1
   return f"P{next_id:04d}"


def assess_disease_risk(patient):
   """Assess what disease the patient is most at risk for based on vital signs."""
   # Respiratory risk factors
   respiratory_rate = patient.get('Respiratory_Rate', 20)
   oxygen_saturation = patient.get('Oxygen_Saturation', 98)
   on_oxygen = patient.get('On_Oxygen', 0)
  
   # Cardiovascular risk factors
   systolic_bp = patient.get('Systolic_BP', 120)
   heart_rate = patient.get('Heart_Rate', 75)
  
   # Infection risk factors
   temperature = patient.get('Temperature', 37.0)
  
   # Calculate risk scores for different conditions
   risks = {}
  
   # Respiratory diseases (Pneumonia, COPD, Asthma)
   resp_score = 0
   if respiratory_rate > 24:
       resp_score += 2
   elif respiratory_rate > 20:
       resp_score += 1
   if oxygen_saturation < 92:
       resp_score += 3
   elif oxygen_saturation < 95:
       resp_score += 2
   elif oxygen_saturation < 97:
       resp_score += 1
   if on_oxygen == 1:
       resp_score += 2
   if temperature > 38.5:
       resp_score += 1
  
   if resp_score >= 3:
       risks['Respiratory Disease'] = {
           'score': resp_score,
           'diseases': ['Pneumonia', 'COPD exacerbation', 'Severe Asthma'],
           'prevention': [
               'Avoid smoking and secondhand smoke',
               'Get vaccinated (flu and pneumonia vaccines)',
               'Practice good hand hygiene to prevent infections',
               'Use prescribed inhalers as directed',
               'Avoid air pollutants and allergens',
               'Maintain good indoor air quality',
               'Seek immediate medical attention if breathing worsens'
           ]
       }
  
   # Cardiovascular diseases (Hypertension, Heart Failure, Arrhythmia)
   cardio_score = 0
   if systolic_bp > 140:
       cardio_score += 3
   elif systolic_bp > 130:
       cardio_score += 2
   elif systolic_bp > 120:
       cardio_score += 1
   if heart_rate > 100:
       cardio_score += 2
   elif heart_rate > 90:
       cardio_score += 1
   if heart_rate < 60:
       cardio_score += 1
  
   if cardio_score >= 2:
       risks['Cardiovascular Disease'] = {
           'score': cardio_score,
           'diseases': ['Hypertension', 'Heart Failure', 'Arrhythmia'],
           'prevention': [
               'Maintain a healthy diet low in sodium and saturated fats',
               'Exercise regularly (at least 150 minutes per week)',
               'Monitor blood pressure regularly',
               'Take prescribed medications as directed',
               'Limit alcohol consumption',
               'Manage stress through relaxation techniques',
               'Maintain a healthy weight'
           ]
       }
  
   # Sepsis/Infection
   infection_score = 0
   if temperature > 39.0:
       infection_score += 3
   elif temperature > 38.5:
       infection_score += 2
   elif temperature > 38.0:
       infection_score += 1
   if heart_rate > 100 and temperature > 38.0:
       infection_score += 1
   if respiratory_rate > 22 and temperature > 38.0:
       infection_score += 1
  
   if infection_score >= 2:
       risks['Sepsis/Infection'] = {
           'score': infection_score,
           'diseases': ['Sepsis', 'Severe Infection', 'Bacteremia'],
           'prevention': [
               'Seek immediate medical attention for high fever',
               'Complete prescribed antibiotic courses',
               'Practice good wound care and hygiene',
               'Get recommended vaccinations',
               'Monitor for signs of infection (fever, chills, rapid breathing)',
               'Stay hydrated and rest adequately',
               'Follow infection control protocols'
           ]
       }
  
   # Hypoxia/Oxygen deficiency
   if oxygen_saturation < 90:
       risks['Severe Hypoxia'] = {
           'score': 5,
           'diseases': ['Severe Hypoxia', 'Respiratory Failure'],
           'prevention': [
               'Use supplemental oxygen as prescribed',
               'Seek immediate emergency medical care',
               'Avoid high altitudes',
               'Monitor oxygen levels regularly',
               'Follow respiratory therapy recommendations',
               'Avoid smoking and pollutants'
           ]
       }
  
   # Determine primary risk
   if not risks:
       return {
           'primary_risk': 'Low Risk',
           'diseases': ['No immediate high-risk conditions detected'],
           'prevention': [
               'Maintain regular health check-ups',
               'Follow a balanced diet and exercise routine',
               'Get recommended screenings and vaccinations',
               'Monitor vital signs regularly'
           ]
       }
  
   # Get the highest risk
   primary_risk = max(risks.items(), key=lambda x: x[1]['score'])
  
   return {
       'primary_risk': primary_risk[0],
       'diseases': primary_risk[1]['diseases'],
       'prevention': primary_risk[1]['prevention']
   }


def calculate_all_risk_scores(df, pipeline, scaler):
   """Calculate risk scores for all patients that don't have them."""
   if pipeline is None or scaler is None:
       return df, 0
  
   # Identify patients without risk scores
   if 'Risk_Score' not in df.columns:
       df['Risk_Score'] = None
  
   # Convert Risk_Score to numeric, treating empty strings and invalid values as NaN
   df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
  
   # Find patients that need risk scores calculated
   missing_scores = df['Risk_Score'].isna()
  
   if not missing_scores.any():
       return df, 0
  
   # Get patients that need scores
   patients_to_score = df[missing_scores].copy()
  
   if len(patients_to_score) == 0:
       return df, 0
  
   # Prepare features - explicitly exclude Risk_Score and other non-feature columns
   feature_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale',
                  'Systolic_BP', 'Heart_Rate', 'Temperature', 'Consciousness', 'On_Oxygen']
  
   # Check if all required columns exist
   missing_cols = [col for col in feature_cols if col not in patients_to_score.columns]
   if missing_cols:
       st.error(f"Missing required feature columns: {missing_cols}")
       return df, 0
  
   try:
       # Extract ONLY the feature columns as DataFrame (explicitly exclude Risk_Score)
       # Make sure we only include the exact columns the model expects
       X = patients_to_score[feature_cols].copy()
      
       # Predict risk scores
       raw_predictions = pipeline.predict(X)
       risk_scores = scaler.transform(raw_predictions.reshape(-1, 1)).flatten()
      
       # Update risk scores in the dataframe
       df.loc[missing_scores, 'Risk_Score'] = risk_scores.round(2)
      
       return df, len(patients_to_score)
   except ValueError as e:
       # This error suggests the pipeline expects different columns
       error_msg = str(e)
       if "columns are missing" in error_msg:
           st.error(f"Model expects different columns than provided.")
           st.error(f"Error: {error_msg}")
           st.info("This might mean the model was trained with different features. Try retraining the model with: python riskcalculator.py")
       else:
           st.error(f"Error calculating risk scores: {e}")
       import traceback
       st.error(f"Traceback: {traceback.format_exc()}")
       return df, 0
   except Exception as e:
       st.error(f"Error calculating risk scores: {e}")
       import traceback
       st.error(f"Traceback: {traceback.format_exc()}")
       return df, 0


def main():
   # Header
   st.markdown('<h1 class="main-header">Pulse</h1>', unsafe_allow_html=True)
  
   # Load data and model
   df = load_dataset()
   pipeline, scaler = load_model()
  
   # Sidebar navigation
   st.sidebar.title("Navigation")
   page = st.sidebar.radio(
       "Choose an option",
       ["Dashboard", "Add Patient", "Remove Patient", "View Patient", "All Patients"]
   )
  
   # Dashboard
   if page == "Dashboard":
       st.header("Dashboard")
      
       # Tableau Dashboard Embed
       tableau_html = """
       <div class='tableauPlaceholder' id='viz1763318125151' style='position: relative'>
           <noscript>
               <a href='#'>
                   <img alt='Datathon Dashboard ' src='https://public.tableau.com/static/images/Da/Datathon_Project/DatathonDashboard/1_rss.png' style='border: none' />
               </a>
           </noscript>
           <object class='tableauViz' style='display:none;'>
               <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
               <param name='embed_code_version' value='3' />
               <param name='site_root' value='' />
               <param name='name' value='Datathon_Project&#47;DatathonDashboard' />
               <param name='tabs' value='no' />
               <param name='toolbar' value='yes' />
               <param name='static_image' value='https://public.tableau.com/static/images/Da/Datathon_Project/DatathonDashboard/1.png' />
               <param name='animate_transition' value='yes' />
               <param name='display_static_image' value='yes' />
               <param name='display_spinner' value='yes' />
               <param name='display_overlay' value='yes' />
               <param name='display_count' value='yes' />
               <param name='language' value='en-US' />
               <param name='filter' value='publish=yes' />
           </object>
       </div>
       <script type='text/javascript'>
           var divElement = document.getElementById('viz1763318125151');
           var vizElement = divElement.getElementsByTagName('object')[0];
           if (divElement.offsetWidth > 800) {
               vizElement.style.width = '100%';
               vizElement.style.height = '827px';
           } else if (divElement.offsetWidth > 500) {
               vizElement.style.width = '100%';
               vizElement.style.height = '827px';
           } else {
               vizElement.style.width = '100%';
               vizElement.style.height = '1327px';
           }
           var scriptElement = document.createElement('script');
           scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
           vizElement.parentNode.insertBefore(scriptElement, vizElement);
       </script>
       """
      
       st.components.v1.html(tableau_html, height=850, scrolling=True)
      
       # Link to full dashboard
       st.markdown("---")
       st.markdown("**View full dashboard:** [Open in Tableau Public](https://public.tableau.com/views/Datathon_Project/DatathonDashboard?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)")
  
   # Add Patient
   elif page == "➕ Add Patient":
       st.header("Add New Patient")
      
       with st.form("add_patient_form"):
           col1, col2 = st.columns(2)
          
           with col1:
               auto_id = generate_patient_id(df)
               patient_id = st.text_input("Patient ID", value=auto_id, help="Leave as is for auto-generation")
               respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=100, value=20, step=1)
               oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=98, step=1)
               o2_scale = st.selectbox("O2 Scale", options=[0, 1, 2], index=0)
               systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=120, step=1)
          
           with col2:
               heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=75, step=1)
               temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
               consciousness = st.selectbox("Consciousness", options=['A', 'P', 'U', 'V'],
                                          format_func=lambda x: {'A': 'Alert', 'P': 'Pain', 'U': 'Unresponsive', 'V': 'Verbal'}[x])
               on_oxygen = st.selectbox("On Oxygen", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
               risk_level = st.selectbox("Risk Level (Optional)", options=[None, 'Low', 'Medium', 'High'], index=0)
          
           submitted = st.form_submit_button("➕ Add Patient", type="primary", use_container_width=True)
          
           if submitted:
               # Validation
               patient_exists = False
               if not df.empty and 'Patient_ID' in df.columns:
                   patient_exists = patient_id in df['Patient_ID'].values
              
               if patient_exists:
                   st.error(f"❌ Patient ID {patient_id} already exists!")
               else:
                   # Create new patient record
                   new_patient = {
                       'Patient_ID': patient_id,
                       'Respiratory_Rate': respiratory_rate,
                       'Oxygen_Saturation': oxygen_saturation,
                       'O2_Scale': o2_scale,
                       'Systolic_BP': systolic_bp,
                       'Heart_Rate': heart_rate,
                       'Temperature': temperature,
                       'Consciousness': consciousness,
                       'On_Oxygen': on_oxygen,
                       'Risk_Level': risk_level if risk_level else 'Unknown'
                   }
                  
                   # Calculate risk score if model is available
                   # Create DataFrame with proper data types
                   patient_df = pd.DataFrame([{
                       'Respiratory_Rate': float(respiratory_rate),
                       'Oxygen_Saturation': float(oxygen_saturation),
                       'O2_Scale': int(o2_scale),
                       'Systolic_BP': float(systolic_bp),
                       'Heart_Rate': float(heart_rate),
                       'Temperature': float(temperature),
                       'Consciousness': str(consciousness),
                       'On_Oxygen': int(on_oxygen)
                   }])
                  
                   if pipeline is not None and scaler is not None:
                       risk_score = calculate_risk_score(patient_df, pipeline, scaler)
                       if risk_score is not None:
                           new_patient['Risk_Score'] = risk_score
                  
                   # Add to dataframe (use new_patient dict which has all columns)
                   patient_df_full = pd.DataFrame([new_patient])
                   if df.empty:
                       df = patient_df_full
                   else:
                       df = pd.concat([df, patient_df_full], ignore_index=True)
                  
                   # Save to CSV
                   df.to_csv("Health_Risk_Dataset.csv", index=False)
                   st.cache_data.clear()
                  
                   st.success(f"✅ Patient {patient_id} added successfully!")
                   if 'Risk_Score' in new_patient:
                       st.info(f"📊 Calculated Risk Score: {new_patient['Risk_Score']:.2f}")
  
   # Remove Patient
   elif page == "❌ Remove Patient":
       st.header("Remove Patient")
      
       if df.empty or 'Patient_ID' not in df.columns:
           st.warning("No patients in the dataset.")
       else:
           patient_ids = df['Patient_ID'].tolist()
           selected_id = st.selectbox("Select Patient ID to Remove", options=patient_ids)
          
           if selected_id:
               # Show patient details before removal
               patient = df[df['Patient_ID'] == selected_id].iloc[0]
               st.subheader("Patient Details")
              
               # Display key info including Risk_Score
               col1, col2 = st.columns(2)
               with col1:
                   st.write(f"**Patient ID:** {patient['Patient_ID']}")
                   if 'Risk_Score' in patient and pd.notna(patient['Risk_Score']):
                       st.write(f"**Risk Score:** {patient['Risk_Score']:.2f}")
                   if 'Risk_Level' in patient:
                       st.write(f"**Risk Level:** {patient['Risk_Level']}")
               with col2:
                   st.write(f"**Respiratory Rate:** {patient['Respiratory_Rate']}")
                   st.write(f"**Oxygen Saturation:** {patient['Oxygen_Saturation']}%")
                   st.write(f"**Heart Rate:** {patient['Heart_Rate']} bpm")
              
               st.write("**Full Details:**")
               st.json(patient.to_dict())
              
               if st.button("🗑️ Remove Patient", type="primary"):
                   # Remove patient
                   df = df[df['Patient_ID'] != selected_id].reset_index(drop=True)
                   df.to_csv("Health_Risk_Dataset.csv", index=False)
                   st.cache_data.clear()
                   st.success(f"✅ Patient {selected_id} removed successfully!")
                   st.rerun()
  
   # View Patient
   elif page == "👤 View Patient":
       st.header("View Patient Details")
      
       if df.empty or 'Patient_ID' not in df.columns:
           st.warning("No patients in the dataset.")
       else:
           patient_ids = df['Patient_ID'].tolist()
           selected_id = st.selectbox("Select Patient ID", options=patient_ids)
          
           if selected_id:
               patient = df[df['Patient_ID'] == selected_id].iloc[0]
              
               col1, col2 = st.columns(2)
              
               with col1:
                   st.subheader("Basic Information")
                   st.write(f"**Patient ID:** {patient['Patient_ID']}")
                   # Show Risk Score prominently
                   if 'Risk_Score' in patient and pd.notna(patient['Risk_Score']):
                       # Display risk score with color coding
                       risk_score = patient['Risk_Score']
                       if risk_score >= 70:
                           color = "🔴"
                           risk_label = "High Risk"
                       elif risk_score >= 40:
                           color = "🟡"
                           risk_label = "Medium Risk"
                       else:
                           color = "🟢"
                           risk_label = "Low Risk"
                       st.markdown(f"**{color} Risk Score:** {risk_score:.2f} ({risk_label})")
                   elif 'Risk_Score' in df.columns:
                       st.write("**Risk Score:** Not calculated")
                   if 'Risk_Level' in patient:
                       st.write(f"**Risk Level:** {patient['Risk_Level']}")
              
               with col2:
                   st.subheader("Vital Signs")
                   st.write(f"**Respiratory Rate:** {patient['Respiratory_Rate']} breaths/min")
                   st.write(f"**Oxygen Saturation:** {patient['Oxygen_Saturation']}%")
                   st.write(f"**Systolic BP:** {patient['Systolic_BP']} mmHg")
                   st.write(f"**Heart Rate:** {patient['Heart_Rate']} bpm")
                   st.write(f"**Temperature:** {patient['Temperature']}°C")
              
               # Disease Risk Assessment
               st.subheader("🔍 Disease Risk Assessment")
               disease_risk = assess_disease_risk(patient.to_dict())
              
               st.markdown(f"**Primary Risk Category:** {disease_risk['primary_risk']}")
              
               st.markdown("**Most At Risk For:**")
               for disease in disease_risk['diseases']:
                   st.write(f"  • {disease}")
              
               st.markdown("**Prevention & Management Recommendations:**")
               with st.expander("View Prevention Tips", expanded=True):
                   for tip in disease_risk['prevention']:
                       st.write(f"  ✓ {tip}")
              
               st.subheader("Full Details")
               st.json(patient.to_dict())
  
   # All Patients
   elif page == "📋 All Patients":
       st.header("All Patients")
      
       if df.empty:
           st.warning("No patients in the dataset.")
       else:
           # Search and filter
           col1, col2 = st.columns(2)
           with col1:
               search_term = st.text_input("🔍 Search by Patient ID", "")
           with col2:
               if 'Risk_Level' in df.columns:
                   risk_filter = st.selectbox("Filter by Risk Level",
                                            options=['All'] + df['Risk_Level'].unique().tolist())
               else:
                   risk_filter = 'All'
          
           # Filter data
           filtered_df = df.copy()
           if search_term:
               filtered_df = filtered_df[filtered_df['Patient_ID'].str.contains(search_term, case=False)]
           if risk_filter != 'All' and 'Risk_Level' in filtered_df.columns:
               filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
          
           st.write(f"Showing {len(filtered_df)} of {len(df)} patients")
          
           # Prepare display columns with Risk_Score prominently placed
           if not filtered_df.empty:
               # Start with essential columns
               display_cols = ['Patient_ID']
              
               # Add Risk_Score early if it exists
               if 'Risk_Score' in filtered_df.columns:
                   display_cols.append('Risk_Score')
              
               # Add other important columns
               other_cols = ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP',
                           'Heart_Rate', 'Temperature', 'Consciousness', 'On_Oxygen',
                           'O2_Scale', 'Risk_Level']
              
               for col in other_cols:
                   if col in filtered_df.columns and col not in display_cols:
                       display_cols.append(col)
              
               # Create formatted display dataframe
               display_df = filtered_df[display_cols].copy()
              
               # Format Risk_Score for better visibility
               if 'Risk_Score' in display_df.columns:
                   display_df['Risk_Score'] = display_df['Risk_Score'].apply(
                       lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                   )
              
               # Display table
               st.dataframe(display_df, use_container_width=True, height=400)
           else:
               st.info("No patients match the search criteria.")
          
           # Download button
           csv = filtered_df.to_csv(index=False)
           st.download_button(
               label="📥 Download Filtered Data as CSV",
               data=csv,
               file_name=f"patients_{risk_filter.lower()}.csv",
               mime="text/csv"
           )


if __name__ == "__main__":
   main()
