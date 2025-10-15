# Kaiburr Assessment 2025 — Task 5: Data Science (Text Classification)

## 📘 Overview
This project is part of the **Kaiburr Internship/Placement Assessment (2025)**.  
The objective is to perform **text classification** on real-world consumer complaint data using machine learning.

I aimed to build a model that can automatically classify consumer complaints into one of the following categories:

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
- **imblearn**
- **xgboost**
- **matplotlib**
- **seaborn**
- **joblib**

---

## 🧩 Dataset Details
The original dataset was **~4 GB** in size (millions of records).  
Since it was too large for local or Colab training, I extracted and filtered only the required product categories:

- Credit reporting, repair, or other personal consumer reports  
- Debt collection  
- Consumer Loan  
- Mortgage  

Initially, a small subset of **~10,000 rows** was used for training and evaluation.\
<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/filters.png" width="25%">
<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/df_head.png" width="65%">

---

## 🧪 Experiment Workflow Summary

### 🔹 Phase 1 — Traditional Naive Bayes Model
I started in a traditional way by training a **Multinomial Naive Bayes** model using the dataset extracted from the given web source.  
The model achieved a **high accuracy (~94.4%)**, but when tested with **custom complaint samples**, it always predicted  
> “Credit reporting, repair, or other”

<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/evaluation_tradional.png" width="50%">
<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/test_traditional.png" width="50%">

This revealed that the dataset was **imbalanced**, having a much larger number of entries belonging to the *Credit reporting* category.

---

### 🔹 Phase 2 — Handling Class Imbalance
To address this imbalance, I applied **Random Under Sampling** using the `imblearn` library. 

<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/imbalanced.png" width="50%">
<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/balanced.png" width="50%">

The dataset was balanced and the Naive Bayes model retrained.  
However, this time, **accuracy dropped significantly**, indicating that the reduced sample size affected learning performance.

<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/evaluation_small_DS.png" width="50%">
<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/test_small_DS.png" width="50%">

---

### 🔹 Phase 3 — Expanding Dataset
I then **increased the dataset size** from around **278 records per class** to **25,000 records per class**, maintaining equal samples across all categories.  
A new Naive Bayes model was trained again.  
The accuracy improved but was **still below expectations**, suggesting the model’s limitation with complex textual data.

<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/evaluation_large_DS.png" width="50%">
<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/test_small_DS.png" width="50%">

---

### 🔹 Phase 4 — Final XGBoost Model
To overcome this, I replaced Naive Bayes with a stronger model — **XGBoost**.  
Using the **balanced dataset (25,000 per category)**, XGBoost was trained and fine-tuned, finally achieving an accuracy of **~93%**.  
This marked a significant improvement in classification performance and stability.

<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/evaluation_XGBoost.png" width="50%">
<img src="https://github.com/Adithya2369/Kaiburr_Task5/blob/cdbaf711bd29131d0774057b93cf7b41233363ff/pics/test_XGBoost.png" width="50%">

---

## 🧠 Process Summary
The workflow followed for this project involved several key data science steps:
### 1. Data Loading & Cleaning:
Imported the filtered Consumer Complaint Database and retained only relevant categories. Removed missing complaint texts and normalized the data for consistency.

### 2. Text Preprocessing:
   - Converted all text to lowercase
   - Removed punctuation, numbers, and stopwords

### 3. Feature Engineering:
Transformed the cleaned complaint narratives into numerical vectors using TF-IDF vectorization with a vocabulary size of 5000 features.

### 4.Train-Test Split:
Divided the dataset into 80% training and 20% testing subsets to evaluate model generalization.

### 5. Model Training & Evaluation:
   - Trained a baseline Naive Bayes classifier and analyzed results.
   - Experimented with undersampling to handle class imbalance.
   - Implemented XGBoost, which achieved the best performance (~93% accuracy).

### 6. Model Saving:
Exported both the model and vectorizer using joblib for future predictions and deployment.

---

## 📉 Key Learnings
- Class imbalance can increase model accuracy but reduce real-world reliability.  
- Under-sampling helps balance data but may lead to underfitting.  
- Increasing data volume and using robust models like XGBoost provides better generalization.

---

## 💾 Project Files
| File | Description |
|------|--------------|
| `Task5_Kaiburr_evaluation.ipynb` | Full experiment notebook — includes all model versions (Naive Bayes, undersampling, and XGBoost) with explanations and evaluations. |
| `Task5_Kaiburr.ipynb` | Final production-ready notebook using XGBoost with the best results (~93% accuracy). |
| `complaint_model.pkl` | Saved Naive Bayes model for baseline comparison. |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer used for the Naive Bayes model. |
| `xgboost_complaint_model.pkl` | Saved XGBoost model — final version. |
| `xgboost_tfidf_vectorizer.pkl` | TF-IDF vectorizer corresponding to the XGBoost model. |
| `pics\` | Folder containing screenshots of outputs, model evaluations, and proof of work (with name and timestamp). |
| `README.md` | Documentation file (this one). |

---

## 🚀 How to Run
1. Clone the repository:  
```bash
git clone https://github.com/Adithya2369/Kaiburr_Task5.git
cd kaiburr-task5
```

2. Open either notebook in **Google Colab** or **Jupyter**:
   - `Task5_Kaiburr_evaluation.ipynb` (to view all experimental steps)
   - `Task5_Kaiburr.ipynb` (to run the final trained model)

3. Upload your dataset or use the preprocessed CSV.
4. Run all cells sequentially to:
   - Train or load the model  
   - Evaluate accuracy and metrics  
   - Test with new complaint samples

---

## 🗾 Example Predictions

| Complaint Text | Predicted Category |
|-----------------|-------------------|
| “The bank added wrong information to my credit report.” | Credit reporting, repair, or other |
| “I keep receiving calls from debt collectors about a loan I never took.” | Debt collection |
| “I applied for a mortgage but it got delayed for months.” | Mortgage |
| “My student loan interest rate was incorrect.” | Consumer Loan |

---

## 👨‍💻 Author
**Adithya Reddy**  
Kaiburr Assessment 2025 — Task 5: Data Science  
**Topic:** Text Classification of Consumer Complaints

---

## 🔒 License Agreement
### Proprietary Rights Notice
This project and all associated materials, including but not limited to source code, documentation, models, and analysis, are the proprietary property of Kaiburr LLC.

---

## Copyright Notice
© 2025 Kaiburr LLC. All Rights Reserved.
