# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Step 2: Load dataset
# Download from https://archive.ics.uci.edu/dataset/352/online+retail
df = pd.read_csv(r"D:\Desktop\Sem_3\Python\Online Retail.csv")

print(df.shape)
df.head()

# Step 3: Data Cleaning
# Remove missing values
df = df.dropna()

# Remove cancellations (InvoiceNo starting with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative/zero quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

print("Remaining rows:", df.shape)



# Step 4: Feature Engineering (Aggregate at product level with Description)
product_df = df.groupby(['StockCode', 'Description']).agg({
    'Quantity': 'sum',
    'InvoiceNo': 'nunique',
    'UnitPrice': 'mean',
    'CustomerID': 'nunique'
}).reset_index()

product_df.rename(columns={
    'Quantity': 'Total_Quantity',
    'InvoiceNo': 'Num_Transactions',
    'UnitPrice': 'Avg_UnitPrice',
    'CustomerID': 'Num_Customers'
}, inplace=True)

# Add Revenue
product_df['Revenue'] = product_df['Total_Quantity'] * product_df['Avg_UnitPrice']

# Calculate popularity threshold (median)
median_quantity = product_df['Total_Quantity'].median()

# Add Popularity label (1 = popular, 0 = not popular)
product_df['Popular'] = (product_df['Total_Quantity'] > median_quantity).astype(int)

# Save labeled dataset
product_df.to_csv("aggregated_products.csv", index=False)
print("Labeled dataset saved as 'aggregated_products.csv' with", product_df.shape[0], "products")

product_df.head()

# Step 5: Feature Selection
X = product_df[['Total_Quantity', 'Num_Transactions', 'Num_Customers', 'Revenue']]
y = product_df['Popular']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# Step 7: Logistic Regression Model
log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Popular','Popular'], yticklabels=['Not Popular','Popular'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Feature Importance (Coefficients)
importance = pd.DataFrame({
    'Feature': ['Total_Quantity', 'Num_Transactions', 'Num_Customers', 'Revenue'],
    'Coefficient': log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

sns.barplot(data=importance, x='Coefficient', y='Feature', palette='viridis')
plt.title("Logistic Regression - Feature Importance")
plt.show()



# Step 10: Function to predict popularity of a product by Description
def product_to_predict(description, product_df, model, scaler):
    # Search product by description (case-insensitive match)
    row = product_df[product_df['Description'].str.lower() == description.lower()]
    
    if row.empty:
        print(f"❌ Product '{description}' not found in dataset.")
        return None
    
    # Extract feature values
    X_new = row[['Total_Quantity', 'Num_Transactions', 'Num_Customers', 'Revenue']]
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Predict with trained model
    pred = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0][pred]
    
    # Map prediction
    label = "Popular ✅" if pred == 1 else "Not Popular ❌"
    
    print(f"📦 Product: {row['Description'].values[0]}")
    print(f"🔢 StockCode: {row['StockCode'].values[0]}")
    print(f"📊 Prediction: {label} (Confidence: {proba:.2f})")
    
    return pred
Des = input("Enter the description of the product: ")
product_to_predict(Des, product_df, log_reg, scaler)
