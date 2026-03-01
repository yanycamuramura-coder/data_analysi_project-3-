# Hospital Appointment Analysis & No-Show Prediction – EDA-Driven Insights for Better Care

**Exploratory Data Analysis + ML Prediction | Actionable Reports on Population Health, Operations & No-Show Reduction**

![Project Impact]:  images/

**Key Business Value**  
Empowers hospitals with **data-driven EDA reports** on patient demographics, population health trends, and operational factors to improve attendance, enhance patient care, reduce no-shows, and optimize resources.

## 🚀 Quick Highlights

- **Comprehensive EDA Focus**: In-depth analysis of demographics (age, gender, city type), health conditions (diabetes, hypertension), and operations (scheduling, reminders, weather impact) with visual reports
- **Population Health Insights**: Reports on disease distributions by age/gender/city to support preventive care and public health strategies
- **Operational Recommendations**: Suggestions for reducing no-shows via better scheduling, targeted reminders, and addressing factors like wait times or travel barriers
- **ML Prediction Model**: Random Forest Classifier (n_estimators=159, max_depth=10) for forecasting no-shows with risk segmentation (Low/Medium/High)
- **Feature Engineering**: Custom categorizations (age bins, distance groups) + domain rules for data consistency
- **Deliverables**: Visual charts, Excel dashboards, health/population reports, and high-risk patient lists — all structured for easy use
- **Tech Stack**: Python • pandas • scikit-learn • seaborn • matplotlib • joblib • openpyxl

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

