
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

iris=datasets.load_iris()
#print(iris)
X=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(X,y)
svc_model=SVC()
svc_model=svc_model.fit(x_train,y_train)

pickle.dump(svc_model,open('svc_model.pkl','wb'))
