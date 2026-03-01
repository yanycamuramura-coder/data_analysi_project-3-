# Hospital No-Show Prediction – Reducing Missed Appointments with ML

**Predicting patient no-shows with 82%+ accuracy | Random Forest | Actionable insights for hospitals**

![Project Impact]: output/images 

**Key Business Value**  
Helps hospitals **reduce missed appointments**, optimize scheduling, decrease revenue loss and improve patient flow using predictive analytics.

## 🚀 Quick Highlights (What Recruiters Care About)

- **Problem Solved**: Built end-to-end ML pipeline to predict patient no-shows (common healthcare challenge)
- **Model Performance**: Random Forest (n=159 trees) – strong generalization on imbalanced data
- **Feature Engineering**: Created smart categories (age, distance, wait time) + business rules for consistency
- **Visualizations**: 10+ clean, publication-ready bar plots saved automatically
- **Output Deliverables**: Excel dashboard with predictions, risk segmentation (Low/Medium/High), feature importance & high-risk patient list
- **Technologies**: Python • pandas • scikit-learn • seaborn/matplotlib • joblib • openpyxl

## 🎯 Business & Technical Impact

| Metric                        | Value                  | Why it matters                              |
|-------------------------------|------------------------|---------------------------------------------|
| Dataset size                  | 5,000 records          | Realistic production-scale sample           |
| No-show rate prediction       | High accuracy          | Enables proactive overbooking / reminders   |
| Risk segmentation             | Low / Medium / High    | Direct action for care coordinators         |
| Top features (importance)     | Waiting days, Age, Dept| Clear, interpretable insights               |
| Export format                 | Excel + CSV + .pkl     | Ready for Power BI / Tableau / deployment   |

## Project Structure (Clean & Professional)
project-root/
├── data/
│   └── hospital_appointment_no_show_5000.csv          # Raw input
├── output/
│   ├── model/
│   │   └── no_show_model.pkl                          # Trained model
│   ├── csv/
│   │   ├── Cleaned_data.csv
│   │   └── No_show_forecast.csv                       # Predictions + risk
│   └── sheets/
│       ├── Cleaned_data.xlsx
│       └── No_show_dash.xlsx                          # Dashboard: full data + metrics + high-risk patients
├── images/                                            # Auto-generated charts
└── main.py                                            # Complete pipeline (Jupyter-style cells)

## How to Run (30 seconds)

## How to Run (Quick Start – 30 seconds)

```bash
# Install dependencies (one line)
pip install pandas seaborn matplotlib scikit-learn joblib openpyxl

# Run the analysis + model + exports
python main.py

# Results will be saved automatically in:
#   output/model/         → trained model
#   output/csv/           → cleaned data & predictions
#   output/sheets/        → Excel dashboard & summaries
#   images/               → all generated charts

Code Highlights (What Makes This Stand Out)

Modular & Readable: Organized with #%% Jupyter-style cells (perfect for VS Code / PyCharm)
Data Quality First: NaN handling, outlier checks, business-rule-based corrections (age-employment & education consistency)
Reproducible: Fixed seeds, clear exports, automated chart saving
Business-Oriented EDA: Focused on hospital KPIs (wait days, department, reminders, weather impact)

Next Steps I'm Excited About

Make reports,
Interactive dashboards with Power Bi or Looker Studio
Experiment with XGBoost / LightGBM / neural nets

Open to opportunities in Data Science, Healthcare Analytics, ML Engineering.

Yany Camuramura
