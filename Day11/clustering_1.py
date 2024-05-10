from sklearn.cluster.hierarchy import linkage , dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns



scaler = StandardScaler().set_output(transform='pandas')
scaler.fit_transform(X)