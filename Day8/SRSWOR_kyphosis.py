import numpy as np
import pandas as pd

kyp = pd.read_csv("C:/Users/Administrator.DAI-PC2/Desktop/MachineLearning/Cases/Kyphosis/Kyphosis.csv")

kyp_ind = list(kyp.index)
#Simple Random Sampling Without Replacement : SRSWOR
samp_ind = np.random.choice(kyp_ind,size=60,replace=False)

samp_kyp = kyp.iloc[samp_ind, :]

#Simple Random Sampling With Replacement : SRSWOR
samp_ind = np.random.choice(kyp_ind,size=60,replace=True)


# Bootstrap Sample
samp_kyp = kyp.iloc[samp_ind, :]