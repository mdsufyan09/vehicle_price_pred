Mobile Price Range Predictor
Predicts a phone’s price range (0–3) — Low, Medium, High, or Very High — using its specifications and a trained ML model.

Tech Stack
Python, pandas, scikit-learn, numpy, streamlit, joblib

Files
main.py → trains & saves model
app.py → Streamlit web app
mobile_price_model.pkl → trained model
dataset.csv → dataset
README.md → this file

Run Locally
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn streamlit joblib
Train model:
python main.py

Run app:
streamlit run app.py

Features
Predicts phone price range from specs
Clean Streamlit UI with Yes/No options
Shows key importance of RAM, battery, and display specs

Model Performance
Accuracy: ~0.89
Algorithm: Random Forest Classifier
