#!/usr/bin/env python
# coding: utf-8

# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# In[44]:


data = pd.read_csv('healthcare-dataset.csv')
data.head().T


# In[3]:


data.head()


# In[4]:


data.drop(columns=['id'], inplace=True)
data.info()


# In[5]:


data.fillna(method="ffill")


# In[6]:


data.info()


# In[7]:


data.fillna(0)  # replaces all NaN values with 0


# In[8]:


# create a function to view pie charts
def piedist(data, column, labels):
    """
    Plots the distribution percentage of a categorical column
    in a pie chart.
    """
    dist = data[column].value_counts()
    colors = ['#66b3ff', '#99ff99', '#ff9999', '#ffcc99', '#be99ff']
    plt.pie(x=dist, labels=labels, autopct='%1.2f%%', pctdistance=0.5, colors=colors)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)


# In[9]:


plt.title('Distribution of Gender')
sns.countplot(x=data['gender'])


# In[11]:


# Number of people with gender of "Other"
(data['gender'] == 'Other').sum()


# In[14]:


# we can drop other type in gender column
data.drop(data[data['gender'] == 'Other'].index, axis=0, inplace=True)


# In[15]:


sns.kdeplot(data=data, x='bmi', hue='stroke')
plt.title('Stroke vs BMI')
plt.legend(['Stroke', 'No stroke'])


# In[17]:


fig = plt.figure(figsize=(8, 5))

ax = plt.subplot2grid((1, 2), (0, 0))
plt.title('Stroke vs Heart disease')
piedist(data=data[data['stroke'] == 1], column='heart_disease', labels=['Heart disease', 'No heart disease'])

ax = plt.subplot2grid((1, 2), (0, 1))
plt.title('No stroke vs Heart disease')
piedist(data=data[data['stroke'] == 0], column='heart_disease', labels=['Heart disease', 'No heart disease'])

plt.tight_layout()


# In[45]:


sns.boxplot(x=data['bmi'])
plt.show()
sns.scatterplot(data=data, x='bmi', y='stroke')
plt.show()
sns.histplot(data['bmi'])


# In[19]:


data = data.join(pd.get_dummies(data['gender']))
data.drop(columns=['gender'], inplace=True)
data.rename(columns={'Female': 'female', 'Male': 'male'}, inplace=True)

data = data.join(pd.get_dummies(data['work_type']))
data.drop(columns=['work_type'], inplace=True)
data.rename(columns={
    'Private': 'private_work',
    'Self-employed': 'self_employed',
    'Govt_job': 'government_work',
    'children': 'children_work',
    'Never_worked': 'never_worked'
}, inplace=True)

data = data.join(pd.get_dummies(data['Residence_type']))
data.drop(columns=['Residence_type'], inplace=True)
data.rename(columns={'Urban': 'urban_resident',
          'Rural': 'rural_resident'}, inplace=True)

data = data.join(pd.get_dummies(data['smoking_status']))
data.drop(columns=['smoking_status'], inplace=True)
data.rename(columns={
    'formerly smoked': 'formerly_smoked',
    'never smoked': 'never_smoked',
    'Unknown': 'smoking_unknown'
}, inplace=True)

data.head().T


# In[21]:


# Replace null in BMI with the median
data['bmi'].fillna(data['bmi'].median(), inplace=True)


# In[23]:


data.head().T


# In[26]:


scaler = StandardScaler()
continuous_columns = ['avg_glucose_level', 'bmi', 'age']
data[continuous_columns] = scaler.fit_transform(data[continuous_columns])  #scale fit the values

# Should have a mean of ~0 and std of ~1
data[continuous_columns].describe().T


# In[27]:


data.head().T


# In[28]:


data['ever_married'].replace(['Yes', 'No'], [1, 0], inplace=True)
data['ever_married'].dtype


# In[29]:


data.head().T


# In[30]:


data.info()


# In[31]:


X = data.drop(columns=['stroke'])
y = data['stroke']


# In[32]:


all_categorical_features = [
    'hypertension',
    'heart_disease',
    'ever_married',
    'female',
    'male',
    'government_work',
    'never_worked',
    'private_work',
    'self_employed',
    'children_work',
    'rural_resident',
    'urban_resident',
    'smoking_unknown',
    'formerly_smoked',
    'never_smoked',
    'smokes'
]


# In[34]:


smotedata = SMOTENC(categorical_features=[feature in all_categorical_features for feature in X.columns])
X_resampled, y_resampled = smotedata.fit_resample(X, y)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=10)


# In[36]:


# Keeps track of models and their scores
model_names = []
model_scores = []


# In[37]:


def confusion_matrix_plot(matrix, model_name):
  
  #Plots the confusion matrix of a model as a heatmap.
  _, ax = plt.subplots(figsize=(5, 3))
  plt.title(f'{model_name} Confusion Matrix')
  sns.heatmap(matrix, annot=True, fmt='d', cmap='Greens')
  ax.set_xticklabels(['Postitive', 'Negative'])
  ax.set_yticklabels(['Postitive', 'Negative'])
  ax.set_ylabel('Predicted Values')
  ax.set_xlabel('Actual Values')
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')


# In[38]:


#===============================================================================================================================
#Ramdom Forest Tree
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=10)
rf.fit(X_train, y_train)


# In[39]:


print(classification_report(y_test, rf.predict(X_test)))


# In[40]:


confusion_matrix_plot(confusion_matrix(y_test, rf.predict(X_test)), 'Random Forest')


# In[41]:


model_names.append('Random Forest')
model_scores.append(roc_auc_score(y_test, rf.predict(X_test)))


# In[42]:


model_and_score = pd.DataFrame()
model_and_score['name'] = model_names
model_and_score['score'] = model_scores
model_and_score.style.background_gradient(cmap=sns.light_palette('green', as_cmap=True))


# In[ ]:




