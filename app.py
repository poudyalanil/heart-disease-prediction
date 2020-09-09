#importing necessary libraries

import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

# Importing libraries for classifiers
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# getting data set
data = pd.read_csv("heart.csv")
print(data.head)
print(data.info())
print(data.describe())


# Feature Selection
import seaborn as sns

# getting correlation
corr = data.corr()
top_corr_features = corr.index
plt.figure(figsize=(20,20))

# getting heatmap
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
