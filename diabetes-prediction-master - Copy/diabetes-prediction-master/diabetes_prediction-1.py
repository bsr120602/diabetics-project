#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[29]:


#loading the dataset to a pandas df
df = pd.read_csv('diabetes.csv')

#printing the first 5 rows
df.head()

#no of rows and cols
df.shape

#getting the statistical measures of the df
df.describe()

#no of diabetics and non-diabetics
df['Outcome'].value_counts()


# In[32]:


print('Before dropping duplicates: ', df.shape)
df = df.drop_duplicates()
print('After dropping duplicates: ', df.shape)


# In[33]:


df.isnull().sum()

"""Check for missing values"""

print('No of missing values in Glucose: ', df[df['Glucose'] == 0].shape[0])
print('No of missing values in BloodPressure: ', df[df['BloodPressure'] == 0].shape[0])
print('No of missing values in SkinThickness: ', df[df['SkinThickness'] == 0].shape[0])
print('No of missing values in Insulin: ', df[df['Insulin'] == 0].shape[0])
print('No of missing values in BMI: ', df[df['BMI'] == 0].shape[0])


# In[34]:


df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())

df.describe()


# In[38]:


import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for DataFrame operations

# Sample data (replace with actual data if needed)
data = {'Outcome': [0, 1], 'Count': [500, 268]}  # Assuming 'Count' represents occurrences
df = pd.DataFrame(data)

# Create a bar chart
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.bar(df['Outcome'], df['Count'])  # Use column names for clarity

# Set title and labels for the chart
plt.title('Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Count')

# Display the plot
plt.show()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with an 'Outcome' column
sns.countplot(x='Outcome', data=df)

# Set the title
plt.title('Outcome Distribution')

# Get the counts of each outcome category
N, P = df['Outcome'].value_counts()

# Print the counts
print("Negative (0):", N)
print("Positive (1):", P)

plt.grid(True)
plt.show()


# In[39]:


import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {'Outcome': [0, 1], 'Count': [500, 268]}
df = pd.DataFrame(data)

# Create a pie chart
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.pie(df['Count'], labels=df['Outcome'], autopct='%1.1f%%', startangle=140)

# Set title
plt.title('Outcome Distribution')

# Display the plot
plt.show()


# In[11]:


df.hist(bins=10,figsize=(10,10))
plt.show()


# In[12]:


#get correlations of each feature in the dataset
corr_mat = df.corr()
top_corr_features = corr_mat.index
plt.figure(figsize=(10,10))
#plot heat map
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# In[13]:


#separating the independent and dependent variables
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']
print(X.head())
print(y.head())


# In[14]:


"""Data Standardisation - Feature Scaling"""

scaler = StandardScaler()
scaler.fit(X)
standardised_data = scaler.transform(X)
print(standardised_data)

X = standardised_data
y = df.Outcome
print(X)
print(y)


# In[15]:


#80% is train, 20% is test
#random state is used to ensure a specific split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

print(X.shape, X_train.shape, X_test.shape)


# In[37]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='liblinear', multi_class='ovr')
lr_model.fit(X_train, y_train)


# In[18]:


"""2) K Neighbours Classifier"""

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)



# In[19]:


"""3) Naive Bayes Classifier"""

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


# In[20]:


"""4) Support Vector Machine(SVM)"""

from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)


# In[21]:


"""5) Decision tree"""

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[22]:


"""6) Random Forest"""

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(criterion='entropy')
rf_model.fit(X_train, y_train)


# In[23]:


#make the predictions using test data for all 6 models
lr_preds = lr_model.predict(X_test)

knn_preds = knn_model.predict(X_test)

nb_preds = nb_model.predict(X_test)

svm_preds = svm_model.predict(X_test)

dt_preds = dt_model.predict(X_test)

rf_preds = rf_model.predict(X_test)

#get the accuracy of the models
print('Accuracy score of Logistic Regression:', round(accuracy_score(y_test, lr_preds) * 100, 2))
print('Accuracy score of KNN:', round(accuracy_score(y_test, knn_preds) * 100, 2))
print('Accuracy score of Naive Bayes:', round(accuracy_score(y_test, nb_preds) * 100, 2))
print('Accuracy score of SVM:', round(accuracy_score(y_test, svm_preds) * 100, 2))
print('Accuracy score of Decision Tree:', round(accuracy_score(y_test, dt_preds) * 100, 2))
print('Accuracy score of Random Forest:', round(accuracy_score(y_test, rf_preds) * 100, 2))


# In[24]:


"""Save the Model with the Highest Accuracy using pickle"""

import pickle
pickle.dump(svm_model, open('svm_model.pkl', 'wb')) #svm has the highest accuracy

pickle.dump(scaler, open('scaler.pkl', 'wb')) #save the std scaler too


# In[ ]:




