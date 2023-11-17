
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats
#Donnees, correlation, regression lineaire de WMC/TASSERT
data = pd.read_csv("jfreechart-test-stats.csv")
X = data[' WMC'].values.reshape(-1, 1)
y = data[' TASSERT'].values

model = LinearRegression()
model.fit(X, y)
a = model.coef_[0]
b = model.intercept_
y_pred = model.predict(X)
sp, p = scipy.stats.spearmanr(X, y)


plt.scatter(X, y, label='Data', color='blue')
plt.plot(X, y_pred, label='Linear Regression', color='red')
plt.xlabel('WMC')
plt.ylabel('TASSERT')
plt.legend()
plt.title('WMC-TASSERT LR')
plt.show()


print(f"Spearman: {sp}")
print(f'a: {a}')
print(f'b: {b}')

#Donnees, correlation, regression lineaire de TLOC/TASSERT
X = data['TLOC'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
sp, p = scipy.stats.spearmanr(X, y)

y_pred = model.predict(X)
a = model.coef_[0]
b = model.intercept_

plt.scatter(X, y, label='Data', color='blue')
plt.plot(X, y_pred, label='Linear Regression', color='red')
plt.xlabel('TLOC')
plt.ylabel('TASSERT')
plt.legend()
plt.title('WMC-TASSERT LR')
plt.show()

print(f"Spearman: {sp}")
print(f'a: {a}')
print(f'b: {b}')

#Info important
x = data['TLOC'].values
y = data[' TASSERT'].values
z = data[' WMC'].values



q1X = np.percentile(x, 25)
medianX = np.percentile(x, 50)
q3X = np.percentile(x, 75)

q1Y = np.percentile(y, 25)
medianY = np.percentile(y, 50)
q3Y = np.percentile(y, 75)

q1Z = np.percentile(z, 25)
medianZ = np.percentile(z, 50)
q3Z = np.percentile(z, 75)

meanX = np.mean(x)
meanY = np.mean(y)
meanZ = np.mean(z)

iqrX = q3X - q1X
lower_limitX = q1X - 1.5 * iqrX
upper_limitX = q3X + 1.5 * iqrX


iqrY = q3Y - q1Y
lower_limitY = q1Y - 1.5 * iqrY
upper_limitY = q3Y + 1.5 * iqrY


iqrZ = q3Z - q1Z
lower_limitZ = q1Z - 1.5 * iqrZ
upper_limitZ = q3Z + 1.5 * iqrZ

print(f"Q1 TLOC: {q1X}")
print(f"Median TLOC: {medianX}")
print(f"Q3 TLOC: {q3X}")
print(f"Mean TLOC: {meanX}")
print(f"Lower Limit TLOC: {lower_limitX}")
print(f"Upper Limit TLOC: {upper_limitX}")
print(f"------------------------------------")

print(f"Q1 TASSERT: {q1Y}")
print(f"Median TASSERT: {medianY}")
print(f"Q3 TASSERT: {q3Y}")
print(f"Mean TASSERT: {meanY}")
print(f"Lower Limit TASSERT: {lower_limitY}")
print(f"Upper Limit TASSERT: {upper_limitY}")
print(f"------------------------------------")

print(f"Q1 WMC: {q1Z}")
print(f"Median WMC: {medianZ}")
print(f"Q3 WMC: {q3Z}")
print(f"Mean WMC: {meanZ}")
print(f"Lower Limit WMC: {lower_limitZ}")
print(f"Upper Limit WMC: {upper_limitZ}")

#boites de moustache et graphe pour hypothese

plus20 = []
plus20A = []
moins20 = []
moins20A = []
i = 0
for i in range(len(x) - 1):
  if y[i]>20:
    plus20.append(z[i])
    plus20A.append(y[i])
  else:
    moins20.append(z[i])
    moins20A.append(y[i])

tout = [plus20, moins20]

meanPlus = np.mean(plus20)
meanMoins = np.mean(moins20)

medPlus = np.percentile(plus20, 50)
medMoins = np.percentile(moins20, 50)



print(f"Median > 20: {medPlus}")
print(f"Mean > 20: {meanPlus}")
print(f"Min > 20: {min(plus20)}")
print(f"------------------------------------")

print(f"Median <= 20: {medMoins}")
print(f"Mean <= 20: {meanMoins}")
print(f"Min < 20: {min(moins20)}")
print(f"------------------------------------")

#boite a moustache TLOC
plt.boxplot(x)

plt.xlabel('TLOC')
plt.ylabel('Value')

plt.show()


#boite a moustache TASSERT
plt.boxplot(y)

plt.xlabel('Tassert')
plt.ylabel('Value')

plt.show()


#boite a moustache WMC
plt.boxplot(z)

plt.xlabel('WMC')
plt.ylabel('Value')

plt.show()


#boite a moustache 2 partie
plt.boxplot(tout)

plt.xlabel('TASSERT')
plt.ylabel('WMC')

plt.show()


#Histogramme partie >20
plt.hist(plus20, bins=15, edgecolor='k', alpha=0.5)

plt.title('Class avec plus de 20 assertions')
plt.xlabel('WMC')
plt.ylabel('nombre')

plt.show()


#Histogramme partie <20
plt.hist(moins20, bins=15, edgecolor='k', alpha=0.5)

plt.title('Class avec moins de 20 assertions')
plt.xlabel('WMC')
plt.ylabel('nombre')

plt.show()