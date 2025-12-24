# 🌾 Agricultural Loan Risk & Advisory System  
### Decision Support System for Agricultural Finance (B.Com Project)

---

## 📌 Project Overview

Agricultural credit is a key pillar of rural development and farmer sustainability. However, loan stress and repayment issues often arise due to **income mismatch, excessive borrowing, credit discipline problems, and dependence on rainfed agriculture**.

The **Agricultural Loan Risk & Advisory System** is a **decision-support application** developed to analyze agricultural loan data and highlight **risk patterns**.  
The system focuses on **financial awareness and analytical understanding**, not on loan approval or rejection.

This project demonstrates how **commerce and finance concepts** can be supported using structured data and visual analysis.

---

## 🎯 Purpose of the Project

The main objectives of this project are:

- To study **risk factors involved in agricultural lending**
- To analyze borrower profiles using structured loan data
- To present insights in a **simple and non-technical manner**
- To promote **responsible borrowing and financial awareness**
- To apply **B.Com concepts** using practical analytical tools

---

## 🧠 What the System Does

- Accepts agricultural loan data through CSV files  
- Classifies borrower profiles into:
  - 🟢 Low Risk  
  - 🟡 Medium Risk  
  - 🔴 High Risk  
- Displays **clear visual dashboards** for easy understanding
- Provides **general improvement suggestions** for risk awareness
- Generates **downloadable CSV and PDF summary reports**

> ⚠️ The system does **not** approve or reject loans.

---

## 📊 Key Insights Provided

The dashboard helps users understand:

- Overall loan risk distribution
- Credit score quality of borrowers
- Relationship between income and loan amount
- Crop-wise risk patterns
- Impact of irrigation type on loan risk
- Income distribution across risk categories

All insights are presented using **simple charts and labels**, suitable for non-technical users.

---

## 📂 Dataset Information

The project uses **synthetically generated agricultural loan data** created solely for learning and analysis.

### Data Fields Used:
- Farmer Age  
- Land Size (Acres)  
- Annual Farm Income  
- Loan Amount  
- Crop Type  
- Irrigation Type  
- Existing Loans  
- Credit Score  

Datasets of **10,000, 50,000, and 100,000 records** were used to test consistency and scalability.

---

## 🧾 Risk Classification Approach (Conceptual)

Risk categorization is based on commonly accepted agricultural finance considerations:

- Credit repayment discipline
- Loan amount in relation to income
- Size of landholding
- Existing loan burden
- Dependence on rainfed agriculture

This approach is **educational and indicative**, not regulatory.

---

## 🧭 Advisory & Awareness Support

For each borrower profile, the system provides **general improvement guidance**, such as:

- Maintaining responsible loan levels
- Improving credit discipline
- Reducing multiple loan exposure
- Exploring group-based lending options
- Considering irrigation support schemes

These suggestions are **informational only** and do not constitute financial advice.

---

## 🛠️ Tools & Technologies Used

| Area | Tools |
|----|------|
| Application Interface | Streamlit |
| Data Handling | Pandas, NumPy |
| Risk Analysis | Rule-based & statistical model |
| Visual Analysis | Matplotlib |
| Reporting | CSV & PDF export |
| Data Source | Synthetic CSV data |

Technology is used strictly as a **support tool for financial analysis**, not as the primary focus.

---

## ▶️ How to Run the Project

1. Install required packages
     
```bash
pip install -r requirements.txt
```

2. Run the application
             
```bash 
streamlit run app.py     
```

 3.Upload an agricultural loan CSV file and explore insights

## 📄 CSV Input Format

To use this application, upload a CSV file with the following **required column headers**:

| Column Name            | Description |
|------------------------|-------------|
| farmer_age             | Age of the farmer (in years) |
| land_size_acres        | Total agricultural land owned (in acres) |
| annual_farm_income     | Annual income from farming (₹) |
| loan_amount            | Loan amount requested (₹) |
| crop_type              | Primary crop grown (Rice, Wheat, Maize, Cotton, Sugarcane) |
| irrigation_type        | Type of irrigation (Rainfed, Canal, Borewell) |
| existing_loans         | Number of active loans |
| credit_score           | Credit score indicator (300–850) |

### 📌 Important Notes
- All monetary values should be in **Indian Rupees (₹)**  
- Column names must match **exactly**  
- This system provides **risk insights only**, not loan approval decisions

### Data Dictionary (Selected Fields)

| Field Name | Description |
|----------|------------|
| crop_type | Type of crop cultivated (Rice, Wheat, Maize, Cotton, Sugarcane) |

| Field Name       | Description |
|----------------|-------------|
| irrigation_type | Primary source of irrigation (Rainfed, Canal, Borewell) |


## ⚖️ Legal & Ethical Note

1. This project is developed strictly for educational and analytical purposes  
2. No real farmer, bank, or credit bureau data is used  
3. No external APIs or live systems are connected  
4. The system does not make lending decisions  
5. The project follows ethical and responsible use of data  

## 🌱 Practical Use Cases

1. B.Com / Finance students – applied academic project  
2. NGOs & Cooperatives – understanding common risk drivers  
3. Training programs – financial literacy and awareness  
4. Academic research – agricultural finance analysis  

## 🔮 Future Scope

1. District or state-level risk analysis  
2. Year-wise agricultural loan trend comparison  
3. Inclusion of basic weather risk indicators  
4. Simplified mobile-friendly interface  


## 👤 Author

Kiran Chhetri
Bachelor of Commerce (B.Com)
Focus Area: Accounting, Taxation & Financial Analysis

## ⭐ Closing Note

This project highlights how financial and commerce principles can be combined with simple analytical tools to improve risk awareness and responsible lending, without replacing human judgment or institutional decision-making.
live demo (Streamlit):https://agriloandecisionsupport-kgb4neaappwslwtze6y7eyx.streamlit.app/
