import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

os.makedirs("output", exist_ok=True)
os.makedirs("output/csv", exist_ok=True)
os.makedirs("output/sheets", exist_ok=True)
os.makedirs("output/model", exist_ok=True)
os.makedirs("images", exist_ok=True)

path = "data/hospital_appointment_no_show_5000.csv"

#%% Data Loading and Initial Standardization
df = pd.read_csv(path)

# Standardizing the Dataframe
df.columns = df.columns.str.strip().str.capitalize()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.capitalize() if isinstance(x, str) else x)

# Treating NAN values
df["Email_reminder"] = df["Email_reminder"].map({
    "Yes": 1,
    "No": 0
})

df["Education_level"] = df["Education_level"].fillna("Unknown")

df["Employment_status"] = df["Employment_status"].fillna("Unknown")

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Age"] = df["Age"].astype(int)

df["Travel_time_min"] = df["Travel_time_min"].fillna(df["Travel_time_min"].mean())

df["Distance_km"] = df["Distance_km"].fillna(df["Distance_km"].mean())

#%% Categorizations and Consistency Corrections
# Age Category
bins = [0, 2, 12, 17, 29, 30, 60]
labels = ["Baby", "Children", "Adolescent", "Young Adult", "Adult", "Elderly"]

df["Age_category"] = pd.cut(df["Age"], bins=bins, labels=labels)

# Distance km
bins = [0.0, 5.0, 15.2, df["Distance_km"].max()]

labels = ["Short distance", "Medium distance", "Long Distance"]

df["Distance_category"] = pd.cut(df["Distance_km"], bins=bins, labels=labels)

# Travel time min
min_ = df["Travel_time_min"].min()
max_ = df["Travel_time_min"].max()

bins = [min_, 20.0, 40.0, max_]

labels = ["Short_time", "Medium_time", "Long_time"]

df["Travel_time_category"] = pd.cut(df["Travel_time_min"], bins=bins, labels=labels)

# Waiting Days
bins = [1, 7, 14, 21]
labels = ["One Week", "Two Weeks", "Three Weeks"]

df["Waiting_days_category"] = pd.cut(df["Waiting_days"], bins=bins, labels=labels)

# Employment Status Correction
valid_rules = {
    "Baby": ["Unemployed"],
    "Children": ["Unemployed"],
    "Adolescent": ["Unemployed", "Employed"],
    "Young Adult": ["Unemployed", "Employed"],
    "Adult": ["Unemployed", "Employed"],
    "Elderly": ["Unemployed", "Employed"]
}

def correct_rows(row):
    age = row["Age_category"]
    employment = row["Employment_status"]

    if employment not in valid_rules.get(age, []):
        return "Unknown"
    return employment

df["Employment_status"] = df.apply(correct_rows, axis=1)

# Education Level Correction
valid_rules = {
    "Baby": ["No education"],
    "Children": ["Primary"],
    "Adolescent": ["Primary", "Secondary", "Unknown"],
    "Young Adult": ["Secondary", "Higher", "Unknown"],
    "Adult": ["Secondary", "Higher", "Unknown"],
    "Elderly": ["Primary", "Secondary", "Higher", "Unknown"]
}

def correct(row):
    age = row["Age_category"]
    educ = row["Education_level"]

    if educ not in valid_rules.get(age, []):
        return "Unknown"
    return educ

df["Education_level"] = df.apply(correct, axis=1)

# Insurance Status
df["Insurance_status"] = df["Insurance_status"].map({
    "Insured": 1,
    "Uninsured": 0
})

# Drop Email_reminder
df = df.drop("Email_reminder", axis=1)

#%% Demographics (EDA)
age_category_count = (df["Age_category"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
age_category_count.columns = ["Age_category", "Count"]
print(age_category_count)

age_category_prop = (df["Age_category"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
age_category_prop.columns = ["Age_category", "Proportion"]
print(age_category_prop)

age_prop = (df.groupby("Age_category")["Gender"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
age_prop.columns = ["Age_category", "Gender", "Proportion"]
print(age_prop)

age_conv = (df.groupby("Age_category", observed=True)["No_show"]
    .mean()
    .mul(100)
    .reset_index()
)

print(age_conv)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
sns.despine(top=True, right=True)
plt.figure(figsize=(5, 10))

ax = sns.barplot(data=age_conv, x="Age_category", y="No_show", color="#121C72")
plt.title("No Show Rate Per Age Category")
plt.xlabel("Age Category", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

gender_dist = (df["Gender"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
gender_dist.columns = ["Gender", "Dist"]
print(gender_dist)

gender_appday = (df.groupby("Gender")["Appointment_day"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
gender_appday.columns = ["Gender", "Appointment_day", "Proportion"]
print(gender_appday)

gender_apptime = (df.groupby("Gender")["Appointment_time_slot"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
gender_apptime.columns = ["Gender", "Appointment_time_slot", "Proportion"]
print(gender_apptime)

gender_conv = (df.groupby("Gender", observed=True)["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
print(gender_conv)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=gender_conv, x="Gender", y="No_show", color="#121C72")
plt.title("No Show Rate Per Gender")
plt.xlabel("Gender", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)


city_type_prop = (df["City_type"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
city_type_prop.columns = ["City_type", "Proportion"]
print(city_type_prop)

city_type_conv = (df.groupby("City_type")["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
print(city_type_conv)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=city_type_conv, x="City_type", y="No_show", color="#121C72")
plt.title("No Show Rate Per City Type")
plt.xlabel("City Type", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

employment_sts_count = (df["Employment_status"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
employment_sts_count.columns = ["Employment_status", "Count"]
print(employment_sts_count)

employment_gen_sts = (df.groupby("Employment_status")["Gender"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
employment_gen_sts.columns = ["Employment_status", "Gender", "Proportion"]
print(employment_gen_sts)


employment_age_sts = (df.groupby("Employment_status")["Age_category"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
employment_age_sts.columns = ["Employment_status", "Age_category", "Proportion"]
print(employment_age_sts)

education_lvl_count = (df["Education_level"]
    .value_counts()
    .reset_index()
)
education_lvl_count.columns = ["Education_level", "Count"]
print(education_lvl_count)

education_gender = (df.groupby("Gender")["Education_level"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)

print(education_gender)

education_age = (df.groupby("Age_category")["Education_level"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
print(education_age)

insured = df[df["Insurance_status"] == 1]
gender_i = insured["Gender"].value_counts()
print("Insurance Status per Gender")
print(gender_i)

age_i = insured["Age_category"].value_counts()
print("Insurance Status per Age")
print(age_i)

city_i = insured["City_type"].value_counts(normalize=True).mul(100)
print("Insurance Status per City type")
print(city_i)

#%% Population Health (EDA)
#Diabetes
diabetes = df[df["Diabetes"] == 1]
print("Diabetes Cases Distribution")

gender_d = diabetes["Gender"].value_counts()
print("Per Gender\n",gender_d)

age_d = diabetes["Age_category"].value_counts()
print("Per Age Category\n",age_d)

city_d = diabetes["City_type"].value_counts(normalize=True).mul(100)
print("Per City Type\n",city_d)

#Hypertension
hypertension = df[df["Hypertension"] == 1]
print("Hypertension Cases Distribution")

gender_h = hypertension["Gender"].value_counts()
print("Per Gender\n",gender_h)

age_h = hypertension["Age_category"].value_counts()
print("Per Age Category\n",age_h)

city_h = hypertension["City_type"].value_counts(normalize=True).mul(100)
print("Per City Type\n",city_h)

#Chronic Disease
c_disease = df[df["Chronic_disease"] == 1]
print("Chronic Disease Cases Distribution")

gender_c = c_disease["Gender"].value_counts()
print("Per Gender\n",gender_c)

age_c = c_disease["Age_category"].value_counts()
print("Per Age Category\n",age_c)

city_c = c_disease["City_type"].value_counts(normalize=True).mul(100)
print("Per City Type\n",city_c)
#%% Hospital Reports / Operational (EDA)
# Distance Category
distance_category_prop = (df["Distance_category"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
distance_category_prop.columns = ["Distance_category", "Proportion"]
print("Distance Category Proportion\n",distance_category_prop)

distance_category_conv = (df.groupby("Distance_category")["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=distance_category_conv, x="Distance_category", y="No_show", color="#121C72")
plt.title("No Show Rate Per Distance Category")
plt.xlabel("Distance Category", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

# Travel Time Category
travel_time_count = (df["Travel_time_category"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
travel_time_count.columns = ["Time_category", "Count"]
print("Travel Time Category Distribution (Count)\n",travel_time_count)

travel_time_conv = (df.groupby("Travel_time_category")["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=travel_time_conv, x="Travel_time_category", y="No_show", color="#121C72")
plt.title("No Show Rate Per Travel Time Category")
plt.xlabel("Travel Time Category", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

# Appointment Day
app_day_dist = (df["Appointment_day"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
app_day_dist.columns = ["Appointment_day", "Count"]
print("Appointment Day Distribution (Count)\n",app_day_dist)

app_day_conv = (df.groupby("Appointment_day", observed=True)["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=app_day_conv, x="Appointment_day", y="No_show", color="#121C72")
plt.title("No Show Rate Per Appointment Day")
plt.xlabel("Appointment Day", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

#Appointment Time Slot
app_time_slot_count = (df["Appointment_time_slot"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
app_time_slot_count.columns = ["Appointment_time_slot", "Count"]
print("Appointment Time Slot Distribution (Count)",app_time_slot_count )

app_time_slot_prop = (df["Appointment_time_slot"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
app_time_slot_prop.columns = ["Appointment_time_slot", "Proportion"]
print("Appointment Time Slot Distribution (Proportion)",app_time_slot_prop )


app_time_slot_conv = (df.groupby("Appointment_time_slot")["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=app_time_slot_conv, x="Appointment_time_slot", y="No_show", color="#121C72")
plt.title("No Show Rate Per Appointment Time Slot")
plt.xlabel("Appointment Time Slot", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

# Department
dep_count = (df["Department"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
dep_count.columns = ["Department", "Count"]
print("Department Distribution (Count)\n",dep_count)
dep_prop = (df["Department"]
    .value_counts(normalize=True)
    .sort_values(ascending=False)
    .mul(100)
    .reset_index()
)
dep_prop.columns = ["Department", "Proportion"]
print("Department Distribution (Proportion)\n",dep_prop)

dep_conv = (df.groupby("Department")["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=dep_conv, x="Department", y="No_show", color="#121C72")
plt.title("No Show Rate Per Department")
plt.xlabel("Department", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

# Waiting Days
waiting_days_count = (df["Waiting_days_category"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
waiting_days_count.columns = ["Waiting_days", "Count"]
print("Wainting Days Distribution (Count)\n",waiting_days_count)

waiting_days_prop = (df["Waiting_days_category"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
waiting_days_prop.columns = ["Waiting_days", "Proportion"]
print("Wainting Days Distribution (Proportion)\n",waiting_days_prop)

waiting_days_conv = (df.groupby("Waiting_days_category")["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=waiting_days_conv, x="Waiting_days_category", y="No_show", color="#121C72")
plt.title("No Show Rate Per Waiting Days Category")
plt.xlabel("Waiting Days Category", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

cros_t = pd.crosstab(df["Previous_appointments"], df["No_show"], normalize="columns") * 100
print("No Show Conversion Per Previous Appointments\n",cros_t)

# Reminders
sms_rmdr_count = (df.groupby("City_type").agg(
    Sms_reminder_count=("Sms_reminder", "count"),
    No_show_rate=("No_show", "mean")
))

sms_rmdr_count["No_show_rate"] = sms_rmdr_count["No_show_rate"] * 100

print("SMS Distribution Per City Type and Conversion Rate\n",sms_rmdr_count)

num_count = (df["Num_reminders"]
    .value_counts()
    .sort_values(ascending=False)
    .reset_index()
)
print("Num Reminders Distribution (Count)\n",num_count)

num_conv = (df.groupby("Num_reminders")["No_show"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .reset_index()
)
print("No Show Rate Per Num Reminders\n",num_conv)

crosst = pd.crosstab(
    index=[df["Num_reminders"], df["Employment_status"]],
    columns=df["No_show"],
    normalize="columns"
) * 100

print("Cross Table\n",crosst)
# Day Status(Rainyday/Holiday)
# Rainy Day
rainy_day_prop = (df["Rainy_day"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
rainy_day_prop.columns = ["Rainy_day", "Proportion"]
print("Rainy Day Distribution(Proportion)\n",rainy_day_prop)

rainy_day_conv = (df.groupby("Rainy_day")["No_show"]
    .mean()
    .mul(100)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=rainy_day_conv, x="Rainy_day", y="No_show", color="#121C72")
plt.title("No Show Rate Per Rainy Day Status")
plt.xlabel("Rainy Day", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

# Holiday
pub_holiday_prop = (df["Public_holiday"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index()
)
pub_holiday_prop.columns = ["Public_holiday", "Proportion"]
print("Holiday Appointment Proportion\n",pub_holiday_prop)

holiday_conv = (df.groupby("Public_holiday")["No_show"]
    .mean()
    .mul(100)
    .reset_index()
)

plt.figure(figsize=(5, 10))

ax = sns.barplot(data=holiday_conv, x="Public_holiday", y="No_show", color="#121C72")
plt.title("No Show Rate Per Public Holiday Status")
plt.xlabel("Public Holiday", loc="center")
plt.ylabel("Conversion Rate", loc="center")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.3)

for bar in ax.patches:
    ax.set_alpha(0.8)

rainy_d = df[df["Rainy_day"] == 1]
holiday = rainy_d[rainy_d["Public_holiday"] == 0]
No_show_rate = holiday["No_show"].value_counts(normalize=True) * 100

print("No Show Rate for  Rainy Day and no Public holiday\n",No_show_rate)


#%% Model Training and Prediction
categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].astype(str)

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(["Patient_id", "No_show"], axis=1)
y = df_encoded["No_show"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=159, random_state=42, max_depth=10)
model.fit(X_train, y_train)

df_encoded["No_show_pred"] = model.predict(X)
df_encoded["Proba"] = model.predict_proba(X)[:, 1]

df_result = df.copy()
df_result["No_show_pred"] = df_encoded["No_show_pred"].values
df_result["Proba"] = df_encoded["Proba"].values
df_result["Risk_seg"] = pd.cut(df_result["Proba"], bins=[0, 0.3, 0.7, 1], labels=["Low", "Medium", "High"])

#%% Exporting Results
# Chart Auto-save
for fig_num in plt.get_fignums():
    plt.figure(fig_num)
    plt.show(block=False)

for i, fig_num in enumerate(plt.get_fignums(), start=1):
    fig = plt.figure(fig_num)
    fig.savefig(f"images/chart_{i}.png", dpi=300)

joblib.dump(model, "output/model/no_show_model.pkl")

cleaned_data = df.drop(["Age", "Distance_km", "Travel_time_min", "Waiting_days"], axis=1)
cleaned_data.to_csv("output/csv/Cleaned_data.csv", index=False)
df_result.to_csv("output/csv/No_show_forecast.csv", index=False)

cleaned_data.to_excel("output/sheets/Cleaned_data.xlsx", index=False)

with pd.ExcelWriter("output/sheets/No_show_dash.xlsx") as writer:
    df_result.to_excel(writer, sheet_name="Complete_data", index=False)

    metrics = {
        "Metrics": ["Train_accuracy", "Test_accuracy", "Total_patients", "No_show_pred%", "High_risk%"],
        "Values": [
            model.score(X_train, y_train),
            model.score(X_test, y_test),
            len(df_result),
            (df_result["No_show_pred"].sum() / len(df_result)) * 100,
            ((df_result["Risk_seg"] == "High").sum() / len(df_result)) * 100
        ]
    }
    pd.DataFrame(metrics).to_excel(writer, sheet_name="Model_Metrics", index=False)

    importance = pd.DataFrame({
        "Variables": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    importance.to_excel(writer, sheet_name="Variables_importance", index=False)

    high_risk = df_result[df_result["Risk_seg"] == "High"].sort_values("Proba", ascending=False)
    high_risk.head(20).to_excel(writer, sheet_name="High_Risk_Patients", index=False)