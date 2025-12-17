#open C:\Users\colin\Documents\ETUDE\MAIN\UTC semestre 5  PK\Neural Networks\final_project\data\metadata\dataset_split.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\colin\Documents\ETUDE\MAIN\UTC semestre 5  PK\Neural Networks\final_project\data\metadata\dataset_split.csv")


sns.countplot(data=df, x='split')
plt.title('Distribution of Data Splits')
plt.xlabel('Data Split')
plt.ylabel('Count')
plt.show()
