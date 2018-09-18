#!/usr/bin/python3

import  matplotlib.pyplot as plt
import  pandas  as pd

df=pd.read_csv('new.csv')

dff=df.head(6)
print(dff)
#plt.xlim(19,400)
#plt.ylim(19,400)
plt.scatter(dff['X'],dff['Y'])
plt.show()
