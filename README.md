# Kaiburr Assessment 2025 — Task 5: Data Science (Text Classification)

## 📘 Overview
This project is part of the **Kaiburr Internship/Placement Assessment (2025)**.  
The objective is to perform **text classification** on real-world consumer complaint data using machine learning.

We aim to build a model that can automatically classify consumer complaints into one of the following categories:

| Label | Category |
|--------|-----------|
| 0 | Credit reporting, repair, or other personal consumer reports |
| 1 | Debt collection |
| 2 | Consumer Loan |
| 3 | Mortgage |

---

## 🧠 Problem Statement
Train a machine learning model capable of classifying textual consumer complaints into predefined product categories based on the complaint narrative.

Dataset source:  
🔗 [Consumer Complaint Database - data.gov](https://catalog.data.gov/dataset/consumer-complaint-database)

---

## ⚙️ Tools & Libraries Used
- **Python 3.10+**
- **Google Colab**
- **pandas**, **numpy**
- **scikit-learn**
- **nltk**
- **matplotlib**
- **seaborn**
- **joblib**

---

## 🧩 Dataset Details
The original dataset was **~4 GB** in size (millions of records).  
Since it’s too large for local or Colab training, we extracted a **subset** by filtering only the required product categories:

- Credit reporting, repair, or other personal consumer reports  
- Debt collection  
- Consumer Loan  
- Mortgage  

We then considered only **10,000 rows** for model training and testing.  
This subset provided a manageable dataset for demonstration and analysis.

---

## 🧪 Workflow Summary

### 1️⃣ Data Loading & Preprocessing
- Imported only the columns:  
  `Product` and `Consumer complaint narrative`
- Cleaned text (lowercasing, removing punctuation, removing stopwords)
- Converted text to TF-IDF features

### 2️⃣ Model Building
- **Algorithm:** Multinomial Naive Bayes  
- **Vectorizer:** TF-IDF (max_features = 5000)
- Split dataset: 80% training / 20% testing

### 3️⃣ Model Evaluation
- The trained model achieved an accuracy of **~90.65%** on the test data.
- Evaluation metrics were computed using `classification_report` and `confusion_matrix`.

---

## 📉 Observations & Issues
During manual testing with custom complaint texts, the model **always predicted**:
> “Credit reporting or other personal consumer reports”

This led us to investigate further, and we discovered the reason:

- The filtered subset contained a **large class imbalance**.  
  A majority of rows (narratives) belonged to *Credit reporting or other*.
- As a result, the model learned mostly from that dominant class.
- Both training and testing data had the same imbalance, which inflated the accuracy to 90.65%.

---

## 🛠️ Future Improvements
To resolve this imbalance problem, we plan to:
1. Create a **balanced dataset** by ensuring **equal number of narrations** from each category.  
2. Retrain and re-evaluate the model.  
3. Experiment with advanced algorithms such as **Logistic Regression** or **Random Forest** to compare results.  
4. Implement **data augmentation** or **SMOTE** for balancing in future iterations.

---

## 📷 Screenshots
1. Screenshot of the dataset filter page showing selected categories  
2. Notebook cell showing dataset shape and head  
3. Accuracy and classification report output (showing 90.65%)  
4. Testing with custom complaint samples and model predictions  

---

## 💾 Model Files
| File | Description |
|------|--------------|
| `complaint_model.pkl` | Saved Naive Bayes model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `Task5_DataScience_AdithyaReddy.ipynb` | Main Colab notebook |
| `README.md` | Documentation file (this one) |

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/kaiburr-task5-text-classification.git
   ```
2. Open the notebook in Google Colab or Jupyter:
  ```bash
  Task5_DataScience_AdithyaReddy.ipynb
  ```
3. Upload your dataset (filtered CSV).
4. Run all cells sequentially to:
    * Train the model
    * Evaluate it
    * Test with new complaint text
# Example Prediction

## Complaint Text
*   "The bank added wrong information to my credit report."
*   "I keep receiving calls from debt collectors about a loan I never took."
*   "I applied for a mortgage but it got delayed for months."
*   "My student loan interest rate was incorrect."

---

## Auditor
**Adithya Reddy**  
Kalburr Assessment – 2025  
Task 5: Data Science Example (Text Classification)

---

## Predicted Category
| Complaint | Predicted Category |
| :--- | :--- |
| The bank added wrong information... | Credit reporting, repair, or other |
| I keep receiving calls from debt collectors... | Debt collection |
| I applied for a mortgage... | Mortgage |
| My student loan interest rate... | Consumer Loan |
