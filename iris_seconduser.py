import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,completeness_score

import matplotlib.pyplot as plt
import seaborn as sns

#use the code from dagshub
import dagshub
dagshub.init(repo_owner='MuktiKsinha', repo_name='mlflow-dagshub-demo', mlflow=True) 

#mlflow.set_tracking_uri('http://localhost:5000') # change it to dagshub url

mlflow.set_tracking_uri('https://dagshub.com/MuktiKsinha/mlflow-dagshub-demo.mlflow')


# load iris dataset
iris=load_iris()
X=iris.data
y=iris.target

#split train test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#define the parameter of random forest
max_depth=10
n_estimators=100


#apply ml_flow
mlflow.set_experiment('iris-rf')

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    #create confusion matrix

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap= 'Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.xlabel('Actual')
    plt.ylabel('predicted')
    plt.title('confusion matrix')

    #save the confusion matrix
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    # log file
    mlflow.log_artifact(__file__)

    #log model
    mlflow.sklearn.log_model(rf,"Randomforest")
    
    #set tags 
    mlflow.set_tag('name','vicky')
    mlflow.set_tag('model','Randomforest')

    print('accuracy',accuracy)


