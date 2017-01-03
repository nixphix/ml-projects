
# coding: utf-8

# ## Predicting TCS Stock Performace
# 
# In this jupyter notebook we will be training linear models for predicting TCS stock price based on 2016 trading data.

# ### Lets Get Data
# 
# Get the historic data of TCS scrip from [NSE](https://www.nseindia.com/products/content/equities/equities/eq_security.htm)
# - choose Price, Volume and Delivery Position data as it provides insight into investors pulse
# - Enter Symbol 'TCS' and select series equity 'EQ'
# - Get data for past 365 days
# - Click get data and then download full data

# In[1]:

# load ammo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score


# ### Loading data into DataFrame
# Assuming the downloaded data is in your current working directory, the dataset should have name like this **`01-01-2016-TO-30-12-2016TCSEQN.csv`**

# In[2]:

# loading data sec_bhavdata_full.csv
# column name is present in data 
df = pd.read_csv('01-01-2016-TO-30-12-2016TCSEQN.csv')
print(df.shape)
df.dtypes


# In[3]:

# let see what we got here
df.describe()


# TCS being largest company by market capitalization it maintains very high liquidity so we will have data for all the trading days of 2016.

# In[4]:

df.head(2)


# In[5]:

df.tail(2)


# ### Exploring dataset
# Lets see how the scrip has performed over the year

# In[6]:

df['Average Price'].plot(color='black',linewidth=0.7)
df['High Price'].plot(color='green', linewidth=0.5)
df['Low Price'].plot(color='red', linewidth=0.5,)
plt.plot([df['Average Price'].mean()]*247, 'b--')
plt.show()


# We see lot of swing in the price, the stock started & ended the year in the same range.

# In[7]:

df['Average Price'].plot(color='black',linewidth=0.7)
df['Open Price'].plot(color='green', linewidth=0.5)
df['Close Price'].plot(color='red', linewidth=0.5,)
plt.plot([df['Average Price'].mean()]*247, 'b--')
plt.show()


# ### Feature Selection
# 
# Lets build a simple model based on **open**, **high**, **low**, **close** and **average price** features to pridect **next day average price**

# In[8]:

# feature and label selection
features = ['Open Price','High Price','Low Price','Close Price','Average Price']
label = ['Average Price']

# prepare dataset with only required columns
# label is the avg. price shifted to next day
dataset = df[features].assign(label=df[label].shift(-1))


# In[9]:

# lets check head of the dataset, now the label is previous days avg. price
dataset.head(5)


# In[10]:

# the last label is nan cause we shifted the label
dataset.tail(2)


# In[11]:

# drop last datapoint as it lacks label
print(dataset.shape)
dataset.dropna(inplace=True)
print(dataset.shape)


# ### Preprocessing Data
# we will scale the features for obvious performace upgrade

# In[12]:

# separate data and target label
data = dataset[features].values
target = dataset['label'].values

# scale the dataset to ease model
data_scaled = preprocessing.scale(data)


# ### Train and Test Data
# Split data into train and test, with train data representing 70%

# In[13]:

test_size = 0.3 # hold out 30% for prediction
random_state = 42 # answer to the univers could be better choice :P

X_train, X_test, y_train, y_test = train_test_split(data_scaled,target,test_size=test_size,random_state=random_state)

X_train.shape, y_train.shape


# In[14]:

X_test.shape, y_test.shape


# ### Model Selection 
# Let us run this data through few linear models to compare their relative preformace. We will choose top two models

# In[15]:

# lets use multiple models and determine their merit with kfold cross validation
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

models = []
models.append(('Linear Regression',LinearRegression()))
models.append(('SVR Linear',SVR(kernel='linear')))
models.append(('SVR Ploy',SVR(kernel='poly')))
models.append(('SVR RBF',SVR(kernel='rbf')))


# ### Cross Validation 
# I tried cv with shuffling(by explicitly using KFold class) and without shuffling(by just passing cv=10), the models performed significantly worse when shuffled, mainly because this is a time series problem and we are not using the key feature `Date` which I'm saving it for later.

# Model evaluation metric for out problem would be Mean Squard Error 

# In[16]:

# lets evaluate models 
results = []
names = []
scoring = 'neg_mean_squared_error'

# kfold with shuffling
from sklearn.model_selection import KFold
num_splits = 10
kf = KFold(n_splits=num_splits,shuffle=True,random_state=random_state)

# kfold without shuffling
cv_fold = 10

# if you wish to try cv will shuffling pass kf to cv instead of sv_fold

for name, model in models:
    result = cross_val_score(model, X_train, y_train, cv=cv_fold,scoring=scoring)
    names.append(name)
    results.append(result)
    msg = "{:20} : mean = {:.5f}  std = {:.5f}".format(name,result.mean(),result.std())
    print(msg)


# ### Comparing CV Score
# 

# In[17]:

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# As we can see in the comparison `Linear Regressionand` and `SVR` with `Linear Kernel` have higest accuracy and very low deviation.
# `SVR Poly` and `SVR RBF` are bad idea if you gonna put your money on this :P

# ### Choosing Final Models
# I have set MSE threshold to -1000 which shoudl get us LR and SVR-Linear

# In[18]:

# lets predict
accuracy_threshold = -1000
cv_models = []
cv_results = []
cv_names = []

for model,result in zip(models,results):
    if result.mean() > accuracy_threshold:
        cv_models.append(model)
        cv_results.append(result)
        cv_names.append(model[0])
        print('adding model... {}'.format(model[0]))


# In[19]:

# lets have a closer look at the 'relatively' good models
fig = plt.figure()
fig.suptitle('Final Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(cv_results)
ax.set_xticklabels(cv_names)
plt.show()


# there are some outlies but the inter quartail range is short (compared to the one trained on shuffled features)

# ### Predicting Test Set

# In[20]:

# make prediction on test set
from sklearn.metrics import mean_squared_error
predictions = []
mseResult = []
for name,model in cv_models:
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    predictions.append(prediction)
    mse = mean_squared_error(y_test,prediction)
    mseResult.append(mse)
    msg = "{:20} : mae = {:.5f}".format(name,mse)
    print(msg)


# The predicton MSE is in line with what we seen in cross validation 

# In[21]:

df_predictions = pd.DataFrame(np.transpose(predictions),columns=['Linear Reg','SVR Linear'])
df_predictions = df_predictions.assign(y_test=y_test)
df_predictions.head()


# In[22]:

df_predictions['Linear Reg'].plot(color='green',linewidth=0.5)
df_predictions['SVR Linear'].plot(color='red', linewidth=0.5)
df_predictions['y_test'].plot(color='black', linewidth=0.5,)
plt.show()


# We can see the Linear Regression is tracing the acutal value more conservatively while SVR overshooting in most of the time

# #### Thats all folks, will try out enseble later 
