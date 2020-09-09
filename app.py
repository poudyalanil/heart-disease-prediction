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