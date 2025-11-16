# Pulse
2025 Datagood Datathon project

## **Team:**
Aryan Achuthan, Arvind Krishna Sivakumar, Kabilesh Yuvaraj


## **Our Goal:**
To create a software that allows medical institutions and professionals to better allocate their time to the patients who need it most urgently.

## **Our Process:**
By using a Gradient boosted decision tree in Python using the provided Health_Risk_Dataset file we can calculate a more nuanced risk score which is a number from 0-100 rather than just low, medium, high, normal, ok. This score is saved in the Health_Risk_Dataset_With_Scores.csv. The risk calculation and saving was done in the riskcalculator.py file

## **Tableau Integration:**
Then we create a Tableau visualization where doctors can see which patients are at the most risk. Graphs of their heart rate vs respiratory rate plotted against each other and oxygen saturation vs systolic BP which show the risk for heart disease and hypotension respectively. The doctors can see generalized views of all the patients as well as each individual patients graphs.  

We created a UI in streamlit in the patient_ui.py file that displays this Tableau dashboard and adds features where the doctor can view the dashboard, view all patients, add a patient, and remove a patient.

## **Running Files**
After downloading all necessary libaries within requirements.txt, run the command below in the terminal. Streamlit will direct you to the UI of Pulse.
python -m streamlit run patient_ui.py
