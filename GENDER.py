import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('gender.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['gender']=lab.fit_transform(data[' Gender'])
data['occupation']=lab.fit_transform(data[' Occupation'])
data['education']=lab.fit_transform(data[' Education Level'])
data['marry']=lab.fit_transform(data[' Marital Status'])
data['color']=lab.fit_transform(data[' Favorite Color'])
x=data[['occupation','education','marry','color',' Age',' Height (cm)',' Weight (kg)',' Income (USD)']]
y=data['gender']

plt.plot(data[' Age'],marker='o',label='age',color='red')
plt.plot(data['marry'],marker='o',label='marrital status',color='blue')
plt.title('age vs marry')
plt.legend()
plt.show()

plt.plot(data[' Age'],marker='o',label='age',color='red')
plt.plot(data['education'],marker='o',label='education',color='blue')
plt.title('age vs education')
plt.legend()
plt.show()

plt.plot(data[' Age'],marker='o',label='age',color='red')
plt.plot(data['occupation'],marker='o',label='occupation',color='blue')
plt.title('age vs occupation')
plt.legend()
plt.show()

plt.plot(data[' Age'],marker='o',label='age',color='red')
plt.plot(data[' Height (cm)'],marker='o',label='Height',color='blue')
plt.title('age vs Height')
plt.legend()
plt.show()

df=data[['gender','occupation','education','marry','color',' Age',' Height (cm)',' Weight (kg)',' Income (USD)']]
for i in df:
        for j in df:
            plt.plot(data[i],marker='o',label=f'{i}')
            plt.plot(data[j],marker='o',label=f'{j}')
            plt.title(f"The information is regarding {i} vs {j}")
            plt.legend()
            plt.show()
for i in df:
    if len(data[i].values) <=5:
        sn.countplot(data[i])
        plt.show()

sn.heatmap(df.corr())
plt.show()

for i in df.columns.values:
    sn.boxplot(df[i])
    plt.show()

sn.pairplot(df)
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
print(tree.score(x_test,y_test))
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print(xgb.score(x_test,y_test))