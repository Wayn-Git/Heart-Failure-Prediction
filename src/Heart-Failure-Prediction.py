# %% [markdown]
# # Importing Libraries

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from joblib import dump



# %% [markdown]
# # Loading And Analyzing data
# 

# %%
df = pd.read_csv('HeartFailureData.csv')

df.head()

# %% [markdown]
# ## Displaying + Checking the following: Information, Describe, Shape, Value Counts(Main Column), Null Values, Nan Values

# %%
df.info()

# %%
df.describe()

# %%
df.shape

# %%
df['DEATH_EVENT'].value_counts()

# %%
df.isnull().sum()

# %%
df.isna().sum()

# %% [markdown]
# ## Displaying The Coorealtion Matrix

# %%
correlation_matrix = df.corr()

plt.figure(figsize=(16, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.show()

# %% [markdown]
# ## Plotting the data with respect to Death Event

# %%
plt.figure(figsize=(15, 12))  # Add this before pairplot
sns.pairplot(df, hue='DEATH_EVENT', height=2.5, aspect=1.2)
plt.suptitle('Heart Failure Features by Death Event', y=1.02, fontsize=16)
plt.show()

# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# ## Seperating the target and train columns

# %%
X = df.drop(columns='DEATH_EVENT')
y = df['DEATH_EVENT']

# %% [markdown]
# ## Splitting the data for model training and testing

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Defining the models in a list

# %%
RFC = RandomForestClassifier(random_state=0)
LR = LogisticRegression(random_state=0)
KNN = KNeighborsClassifier(n_neighbors=5)
SVM = SVC(random_state=0)
DT = DecisionTreeClassifier(random_state=0)

models = [RFC, LR, KNN, SVM, DT]

# %% [markdown]
# # Training the model and displaying the result

# %%
model_result = {}
trained_models = {}

for model in models: 
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    trained_models[model.__class__.__name__] = model
    
    
    
    model_result[model.__class__.__name__] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    results = model_result[model.__class__.__name__]
    print(f"{model.__class__.__name__} - Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}\n")

# %% [markdown]
# ## Dispalying the CV_Score

# %%

for model in models:
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"{model.__class__.__name__}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %% [markdown]
# ## Overall Model comparision

# %%
results_df = pd.DataFrame(model_result).T
print("\n=== Model Comparison ===")
print(results_df.round(4))

# Find best model
best_model = results_df['recall'].idxmax()  # Focus on recall for medical data
print(f"\nBest model based on recall: {best_model}")

# %% [markdown]
# # Saving the best model for future use

# %%
best_model_name = results_df['recall'].idxmax()
best_trained_model = trained_models[best_model_name]
model_package = {
    'model': best_trained_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'model_name': best_model_name
}

filename = 'heartFailure_model.pkl'
dump(model_package, filename)
print(f"Model saved as {filename}")

# %%



