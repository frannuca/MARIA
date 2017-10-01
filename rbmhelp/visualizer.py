import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates

url = "E:/crbm_input.csv"
data = pd.read_csv(url)

x= np.array(data.values[:,0])
y = np.array(data.values[:,1])
plt.scatter(x,y,color="red")

url2 = "E:/crbm.csv"
data2 = pd.read_csv(url2)

x2= np.array(data2.values[:,0])
y2 = np.array(data2.values[:,1])


plt.scatter(x2,y2,color="yellow")
plt.show()