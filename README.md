# 🚗 CO2 Emission Predictor (ICE Vehicles)

A Streamlit web app that predicts CO2 emissions for Internal Combustion Engine (ICE) vehicles using Linear Regression models trained on EV vs ICE vehicle data from 2015–2026.

---

## 📦 Requirements

- Python 3.8 or higher
- pip

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/suryapro34/DatascienceAlp.git
cd DatascienceAlp
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**

The app will automatically open at:
```
http://localhost:8501
```

---

## 📁 File Structure

```
├── app.py                                  # Main Streamlit application
├── EV_vs_ICE_Vehicle_Specs_2015_2026.csv   # Dataset
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## 🧭 How to Use the App

1. **Dataset & EDA tab** — View the dataset and explore charts
2. **Training & Evaluation tab** — Choose a model (SLR or MLR), set test size, and click 🚀 Train Model
3. **Predict New Values tab** — Enter engine specs and click 🔍 Predict to get a CO2 estimate
4. **History tab** — View all past predictions and training runs, with option to download as CSV

---

## 📌 Notes

- Make sure `EV_vs_ICE_Vehicle_Specs_2015_2026.csv` is in the same folder as `app.py`
- Train a model first before making predictions
- History resets when the app is restarted
