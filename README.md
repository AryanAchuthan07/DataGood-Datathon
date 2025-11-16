# DataGood-Datathon 2025
Datagood Datathon project, hosted by DataGood

Team: Aryan Achuthan, Arvind Krishna Sivakumar, Kabilesh Yuvaraj

Our goal: To create a software that allows medical institutions and professionals to better allocate their time to the patients who need it most urgently.

Our process: By using a Gradient boosted decision tree in Python using the provided Health_Risk_Dataset file we can calculate a more nuanced risk score which is a number from 0-100 rather than just low, medium, high, normal, ok. This score is saved in the Health_Risk_Dataset_With_Scores.csv.

Then we create a Tableau visualization where doctors can see which patients are at the most risk. Graphs of their heart rate vs respiratory rate plotted against each other and oxygen saturation vs systolic BP which show the risk for heart disease and hypotension respectively. The doctors can see generalized views of all the patients as well as each individual patients graphs. 

We created a UI in streamlit that displays this Tableau and adds features where you can add and remove patients.
