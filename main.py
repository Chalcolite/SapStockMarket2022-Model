#%%

#OBJECTIVE: Use a series of tests to substantiate
# or challenge a correlation between the adjusted
# closing price of a stock & volume to it's high
# 
#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# Create function to load any potential dataframes
  
#Load dataframe paths & print
stocks_path = r"C:\Users\kwame\Desktop\TheL\infolimpioavanzadoTarget.csv"
dfstocks = pd.read_csv(stocks_path, index_col=['date'])
dfstocks['Dollar Volume'] = dfstocks['volume'] * dfstocks['open']
dfstocks = dfstocks.drop(columns ='TARGET', axis=1)
pd.to_datetime(dfstocks.index)
dfstocks.head()
 # %%
#Define beginning uppercase letters
dfstocks.columns = dfstocks.columns.str.title()
dfstocks.dtypes
 # %%
#Create variables for testing by splitting the feauture matrix (X)
# the target vector (y)
X = dfstocks.drop(columns='Ticker')
y = dfstocks['Ticker']
dfstocks.drop(columns='Ticker', inplace=True)
numerical = dfstocks.select_dtypes('float64').columns
categorical = dfstocks.select_dtypes('object').columns
dfstocks.head(10)
 # %%
#Split the training test & the validation, with the middle of the yea 
cut_date = '2022-07-02'   
cutoff = pd.to_datetime(cut_date)
#Represents the first half of the dataset
base =  pd.to_datetime(dfstocks.index) <= cutoff
X_train, y_train = X.loc[base], y.loc[base] 
X_val, y_val =   X.loc[~base], y.loc[~base] 
def train_infinite_replace(df):
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], df[col].replace([np.inf, -np.inf], np.nan).max())
    return df
train_infinite_replace(X_train)
train_infinite_replace(X_val)
X_train.head(15)
#%%
#Create transformers
numerical_transformer = make_pipeline(SimpleImputer(strategy = 'median'),
                                      StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy = 'constant', fill_value='missing'),
                                      OrdinalEncoder())
transformer = ColumnTransformer(transformers=[('num', numerical_transformer, numerical), ('cat', categorical_transformer, categorical)])

classifier = make_pipeline(transformer, DecisionTreeClassifier())

 # %%
# Get a baseline accuracy score 
base_acc = y_train.value_counts(normalize=True).max()
base_acc
 # %%
#Fit variables into the pipeline generated above
classifier.fit(X_train, y_train)
 # %%
#Find the training & validation accuracy scores
training_acc = classifier.score(X_train, y_train)
val_acc = classifier.score(X_val, y_val)
print('Training Accuracy Score:', training_acc)
print('Validation Accuracy Score:', val_acc)
 # %%
 #Display the most import features of the model

ordering = numerical
importances = pd.Series(classifier.steps[1][1].feature_importances_, ordering)
n = 5
plt.figure(figsize=(10,n/2))
plt.title(f'Top {n} features')
importances.sort_values()[-n:].plot.barh(color='black')
plt.show()

# %%
#Permute the most significant columns to get their validation score
# The most significant columns would weight down the validation score the most due to their significance being altered
colossal = 'Vwapadjclosevolume'
X_val_permuted  = X_val.copy()
X_val_permuted[colossal].fillna(value = X_val_permuted[colossal].median(), inplace=True)
X_val_permuted[colossal] = np.random.permutation(X_val[colossal])
print('Feature permuted: ', colossal)
print('Validation Accuracy: ', classifier.score(X_val, y_val))
print('Validation Accuracy (permuted)', classifier.score(X_val_permuted, y_val))
print('Difference of Permutation: ', classifier.score(X_val, y_val) - classifier.score(X_val_permuted, y_val))
# %%
#Permute the second largest feature to compare the differences
colossal = 'Atr20'
X_val_permuted  = X_val.copy()
X_val_permuted[colossal].fillna(value = X_val_permuted[colossal].median(), inplace=True)
X_val_permuted[colossal] = np.random.permutation(X_val[colossal])
print('Feature permuted: ', colossal)
print('Validation Accuracy: ', classifier.score(X_val, y_val))
print('Validation Accuracy (permuted)', classifier.score(X_val_permuted, y_val))
print('Difference of Permutation: ', classifier.score(X_val, y_val) - classifier.score(X_val_permuted, y_val))
# %%
#Print confusion matrix

ConfusionMatrixDisplay.from_estimator(classifier, X_val, y_val) 
plt.figure(figsize=(10, 50))
plt.show()
# %%
#Encode labels & implement the XGBClassifier
y_train_encoded = LabelEncoder().fit_transform(y_train)
y_val_encoded = LabelEncoder().fit_transform(y_val)
xg_classifier = XGBClassifier(n_estimators=50, random_state=42, learning_rate = 0.25)
fitted = xg_classifier.fit(X_train, y_train_encoded)
print('Validation Accuracy: XGBoost Classification', xg_classifier.score(X_val, y_val_encoded))
# %%
#How much do the input values partially matter in terms of predicting the model?
#Create a partial dependence plot based on your target column & feature matrix,
# then explain individual predictions through the usage of a Shapely value plot
from pdpbox.pdp import PDPIsolate
X_update = train_infinite_replace(X_val)
isolate = PDPIsolate(model = classifier,
                     df = X_update,
                     model_features=X_val.columns,
                     feature = 'Vwapadjclosevolume',
                     feature_name = 'Vwapadjclosevolume',
                     n_classes = len(y.unique())
                     )
figure, axes = isolate.plot()
figure
#%%
#Create the ROC Curve to plot the false positive 
# & true positive rates
pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'), SimpleImputer(), LogisticRegression(random_state=42))
logreg_classifier = pipeline.fit(X_train, y_train)
y_pred_proba = logreg_classifier.predict_proba(X_val)[::,1]
fpr, tpr, thresholds = roc_curve(y_val,  y_pred_proba)
plt.plot(fpr,tpr)
plt.plot([0,1], ls='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc=4)
plt.show()
#%%
#Now implement shapely
import shap
shaper = shap.TreeExplainer(fitted)
shap_val = shaper.shap_values(X_update)
shap.initjs() 
shap.force_plot(shaper.expected_value, shap_val[25,:], X_update.iloc[25,:])
# %%evg
