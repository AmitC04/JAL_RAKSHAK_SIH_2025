# SIH-2025
MODELED TRAINED


https://nbviewer.org/github/AmitC04/SIH-2025/blob/main/EDA__DATA.ipynb



```markdown
# 🌊 JAL RAKSHAK — Smart India Hackathon 2025  
### **AI-Powered Water Quality & Health Risk Analysis System**

---

## 📘 Overview  
**JAL RAKSHAK** is a Smart India Hackathon (SIH 2025) project built to analyze water-quality parameters and predict the associated **health risks** using Data Science and Machine Learning.

This system performs:  
- Full exploratory data analysis  
- Cleaning + merging of multi-source datasets  
- Disease–water quality correlation  
- Risk-level generation  
- ML-based risk prediction  
- Exportable model (.pkl)  
- Ready-to-use datasets for deployment & dashboards  

The goal is to support **government agencies, environmental bodies, health departments, and NGOs** in taking data-backed decisions for public health and water-safety improvement.

---

## 📁 Project Structure  

```

├── EDA__DATA.ipynb                       # Full EDA: cleaning, merging, visualizations

├── EDA_Report_Diseases.html              # Auto-generated interactive EDA report

├── NE_WaterQuality_with_Diseases.csv     # Merged water-quality + diseases dataset

├── NE_WaterQuality_with_RiskLevels.csv   # Dataset with computed risk levels

├── SIH_MODEL_TRAINED.ipynb               # ML model training notebook

├── random_forest_model.pkl               # Trained Random Forest model

├── final_nhs-wq_pre_2023_compressed.xlsx # Source dataset

└── README.md                             # Documentation (this file)

````

---

## 🎯 Project Objectives  
- Analyze regional water-quality parameters  
- Identify diseases associated with poor water quality  
- Build an explainable ML model for **risk classification**  
- Create reusable datasets suitable for dashboards or APIs  
- Deliver a system that supports **preventive public-health action**

---

## 🚀 Features  
- 📊 **Complete EDA** with visual insights (correlation heatmaps, distributions, trends)  
- 🧹 **Automatic preprocessing** of water-quality datasets  
- 🔗 **Disease mapping** with region-wise merging  
- ⚠️ **Risk level generation** based on threshold analysis  
- 🤖 **Random Forest ML model** for health-risk prediction  
- 📄 **Exportable HTML EDA report**  
- 🗂️ **Cleaned datasets** ready for external projects  

---

## 🛠️ Getting Started

### 🔹 Requirements  
- Python 3.x  
- pip  
- Jupyter Notebook  

### 🔹 Installation  
```bash
# Clone the repository
git clone https://github.com/AmitC04/JAL_RAKSHAK_SIH_2025.git
cd JAL_RAKSHAK_SIH_2025

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# .\venv\Scripts\activate       # Windows

# Install dependencies (if requirements.txt is added later)
pip install -r requirements.txt
````

---

## 📊 Usage Guide

### 1️⃣ Run Exploratory Data Analysis

Open the notebook:

```
EDA__DATA.ipynb
```

This generates:

* Cleaned datasets
* Merged water-quality + disease tables
* Correlation visualizations
* Risk thresholds
* HTML report (already included)

### 2️⃣ Use the Processed Datasets

Files generated after EDA:

* `NE_WaterQuality_with_Diseases.csv`
* `NE_WaterQuality_with_RiskLevels.csv`

### 3️⃣ Train or Modify the ML Model

Open:

```
SIH_MODEL_TRAINED.ipynb
```

You can retrain, tune, or replace the model.

### 4️⃣ Use the Trained Model

Model file:

```
random_forest_model.pkl
```

Load it into any Python script for prediction.

---

## 📈 Results & Insights

The EDA Report provides:

* Correlation between water-quality parameters and diseases
* Map-based risk distribution
* Parameter-wise severity scoring
* Region-level risk classification

This helps identify critical regions requiring **intervention and resource allocation**.

---

## 🧰 Technologies Used

* Python (NumPy, Pandas)
* Scikit-Learn
* Matplotlib & Seaborn
* Jupyter Notebook
* CSV/Excel datasets
* Git & GitHub

---

## 🤝 Contributing

Contributions are welcome!
You can improve:

* Data visualizations
* Risk algorithms
* ML models
* Documentation
* Dashboard integration

Submit a pull request or open an issue.

---

## 🙌 Credits

* Developed by **Amit**
* Data collected from publicly available government & health datasets
* Analysis notebooks & ML models created for SIH 2025 problem statement

---

## 📄 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute with attribution.

---

## ⭐ Support

If this project helped you or you found it interesting, please **star ⭐ the repository** on GitHub!

```

---

If you want, I can also generate:  
✅ `requirements.txt`  
✅ A professional banner image for your GitHub  
✅ A better folder structure for SIH submission  
Just tell me!
```
