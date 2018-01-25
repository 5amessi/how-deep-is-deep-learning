import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
train = pd.read_csv('train.csv')
y = np.asarray(train['Survived'])
train['Sex'] = train['Sex'].replace(['male','female'] , (1,0))
train['Embarked'] = train['Embarked'].replace(['C','S','Q'] , (1,2,3))
#train = train.drop(['Survived','Name','Ticket','Cabin','Embarked','Pclass','Age','Fare','Parch','SibSp'] , axis=1)
train = train.drop(['PassengerId','Survived','Name','Ticket','Cabin'] , axis=1)

train = np.nan_to_num(train)
X = np.asarray(train)
X_norm = (X - X.min())/(X.max() - X.min())


pca = sklearnPCA(n_components=2) #2-dimensional PCA

transformed = pd.DataFrame(pca.fit_transform(X_norm))

plt.scatter(transformed[y==0][0], transformed[y==0][1], label='Class 1', c='red')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Class 2', c='blue')
plt.legend()
plt.show()


lda = LDA(n_components=2) #2-dimensional LDA
lda_transformed = pd.DataFrame(lda.fit_transform(X_norm, y))
print(X_norm)
print(X_norm)

# Plot all three series
plt.scatter(lda_transformed[y==0][0], lda_transformed[y==0][1], label='Class 1', c='red')
plt.scatter(lda_transformed[y==1][0], lda_transformed[y==1][1], label='Class 2', c='blue')

# Display legend and show plot
plt.legend(loc=2)
plt.show()
