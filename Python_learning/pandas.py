# Practice using library Pandas
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# create a series -- the output would be stored as float
s = pd.Series([1,3,5,np.nan,6,8])

# create date object
dates = pd.date_range("20171116",periods = 6)

# create a dataframe with 6 rows and 4 cols, index is row label
df = pd.Dateframe(np.random.randn(6,4),index = dates, columns = list('ABCD'))
# another way using dict to create a dataframe
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' }

# some useful inspection methods
df2.dtypes
df.index
df.columns
df.values
df.describe()
df.T
df.sort_values(by ='B')

# getting the values
df['A']
df[0:3]
df['row_lable1':"row2"]


df2 = df.copy()
df2['E'] = ['one', 'two']
df2[df2['E'].isin(['two','one'])] #filter

######### Analytics Vidhya
df = pd.read_csv("file_path.csv")
df.head(10)
df.describe()
#for non-numerical values, can print frequency table
df['Property_Area'].value_counts(ascending = True)

#Distribution Analysis
## histogram
df["ApplicationIncome"].hist(bins = 50)

## boxplot
df.boxplot(column = 'ApplicantIncome')
df.boxplot(column = 'ApplicationIncome', by = 'Education')

# frequency table
temp2 = df.pivot_table(values = 'Loan_Status', index=['Credit_History'],
	aggfunc = lambda x: x.map({'Y':1,"N":0}).mean())

# how to plot categorical bar plots
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlable('Credit History')
ax1.set_ylable('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1 = df['Credit_History'].value_counts(ascending = True)
temp1.plot(kind = "bar")

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

# Another way... you have credit_history on the x_axis, and counts on y axis. loan_status each.
temp3 = pd.crosstab(df['Credit_History'],df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

# Checking missing values 
df.apply(lambda x: sum(x.isnull()),axis =0)

# How to fill missing values -- replace with mean
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)

# example:
df['Self_Employed'].value_counts()
#### You will see that 86% is no
df['Self_Employed'].fillna('No',inplace=True)
### Create a pivot table
table = df.pivot_table(values = 'Loan_Amount', index="Self_Employed",columns = "Education"
	aggfunc = np.meadian)
### Define a function that returns the value of the table
def fage(x):
	return table.loc[x["Self_Employed"],x['Education']]

### Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage,axis=1),inplace=True)
### You are filling NA values in LoanAmount. fage() passes the row with LoanAmount NA.

# Transform to log scale -- to subside the effect of extreme values
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)




## Data Manipulation

import pandas as pd
import numpy as np
data = pd.read_csv("train.csv", index_col="Loan_ID")

#1 Boolean Indexing
data.loc[(data["Gender"] == "Female") & (data["Education"] == "Not Graduate"),["Gender","Education"]]

#2 Apply Function
def num_missing(x):
	return sum(s.isnull())

print "Missing values per column: "
print data.apply(num_missing, axis=0) # 0 is column

#3 Inputting missing files
from scipy.stats import mode
data['Gender'].fillna(mode(data['Gender']).mode[0], inplace=True)
print data.apply(num_missing, axis=0) #check if imputing worked

#4 Pivot Table
impute_grps = data.pivot_table(values = ["LoanAmount"], index = ["Gender","Married","Self_Employed"],
	aggfunc = np.mean)
print impute_grps

#5 Multi-indexing -- requires tuple
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['LoanAmount'].isnull(),:].iterrows():
	ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
    data.loc[i, 'LoanAmount'] = impute_grps.loc[ind].values[0]

#6 Crosstab
#absolute vals
pd.crosstab(data["Credit_History"],data["Loan_Status"],margins = True)
#percentage
def percConverter(ser):
	return ser/float(ser[-1])
	pd.crosstab(data["Credit_History"],data["Loan_Status"],margin = True).apply(percConverter,axis = 1)

#7 Merge dataframes(confusing)
prop_rates = pd.DataFrame([1000, 5000, 12000], index=['Rural','Semiurban','Urban'],columns=['rates'])
data_merged = data.merge(right=prop_rates, how='inner',left_on='Property_Area',right_index=True, sort=False)
data_merged.pivot_table(values='Credit_History',index=['Property_Area','rates'], aggfunc=len) #confirm

#8 Sorting dataframes -- can be based on multiple columns
data_sorted = data.sort_values(['ApplicationIncome', "CoapplicantIncome"], ascending = False)
data_sorted[['ApplicantIncome','CoapplicantIncome']].head(10)

#9 Plot
import matplotlib.pyplot as plt
%matplotlib inline
data.boxplot(column="ApplicantIncome",by="Loan_Status")

data.hist(column="ApplicantIncome",by="Loan_Status",bins=30)

#10 Binning
#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

cut_points = [90,140,190]
labels = ["low","medium","high","very high"]
data["LoanAmount_Bin"] = binning(data["LoanAmount"], cut_points, labels)
print pd.value_counts(data["LoanAmount_Bin"], sort=False)

#11 Coding Categorical Data
def coding(col, codeDict):
	colCoded = pd.series(col, copy=True)
	for key,value in codeDict.items():
		colCoded.replace(key,value,inplace = True)
return colCoded

#example coding loanStatus as Y=1, N=0
print 'Before Coding:'
print pd.value_counts(data["Loan_Status"])
data["Loan_Status_Coded"] = coding(data["Loan_Status"],{'N':0,'Y':1})
print '\nAfter Coding:'
print pd.value_counts(data["Loan_Status_Coded"])

#12 Iterating rows of dataframe
for i, row in colTypes.iterrows() # i: dataframe index, row: series
    if row['type'] == "categorical":
    	data[row["feature"]] = data[row["feature"]].astype(np.object)
    elif row['type']=="continuous":
        data[row['feature']]=data[row['feature']].astype(np.float)
print data.dtypes




# Building a predictive model 
## We use sklearn to do this job! This requires all input to be numeric

from sklearn.preprocessing import LableEncoder
var_mod = ['Gender','Married'...]
le = LableEncoder()
for i in var_mode:
	df[i] = le.fit_transform(df[i])

# Code Skeleton
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

# Generic function for making a classification model and accessing performance:
# Can try different methods in model. Eg, model = LogisticRegression()
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])








## Tree Based Modeling
### Note that sklearn does not provide pruning, check xgboost!
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion = 'gini')
model.fit(X,y)
model.score(X,y)
predicted = model.predict(x_test)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 1000)
model.fit(X,y)
predicted = model.predict(x_test)






### Analytics Vidhya Workshop: Python STATS










