import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.modal_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

col_names = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Survived']
df = pd.read_csv('titanic.csv', names=col_names).iloc[1:]

features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
x = df[features]
y = df.Survived
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

dot_data = StringIO()
export_graphviz(clf, out_file = dot_data, filled = True, rounded = True, special_characters = True, feature_names = features, class_names = ['0','1'])
print(dot_data.getvalue()) 

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Titanic.png')
Image(graph.create_png())

