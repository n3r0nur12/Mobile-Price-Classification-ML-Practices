"""
   ____ ____  _____ _  _    ___   ___   ___  
  / ___/ ___|| ____| || |  / _ \ ( _ ) ( _ ) 
 | |   \___ \|  _| | || |_| | | |/ _ \ / _ \ 
 | |___ ___) | |___|__   _| |_| | (_) | (_) |
  \____|____/|_____|  |_| _\___/_\___/_\___/ 
 |  _ \|  _ \ / _ \    | | ____/ ___|_   _|  
 | |_) | |_) | | | |_  | |  _|| |     | |    
 |  __/|  _ <| |_| | |_| | |__| |___  | |    
 |_|   |_| \_\\___/ \___/|_____\____| |_|    
                                             
        ONURCAN ISLER 150120825
        ERKAM KARACA 150118021
        BERK KIRTAY 150118043

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier


def visualize_data():
    df = pd.read_csv('train.csv')
    #plt.figure(figsize = (17,10))
    #corrmat = df.corr()
    #sns.heatmap(corrmat, square = True)
    #plt.title('Correlation', fontsize = 20)
    #plt.show()

    #g = sns.FacetGrid(df,hue = 'price_range', height = 10)
    #g.map(plt.scatter, 'ram','battery_power',alpha = 0.6)
    #g.add_legend()
    #g.savefig("BatteryPower.png")


def plot_results():
    algs = ["Logistic Regression", "Random Forest", "SVM", "K-neighbors", "Decision Tree"]
    accs = [0.634, 0.872, 0.964, 0.916, 0.86]
    fig = plt.figure(figsize=(10,5))
    plt.bar(algs, accs, color='maroon', width=0.4)
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracty Rate")
    plt.title("Accuracty Rates Algorithms")
    #plt.show()
    plt.savefig("results.png")


def print_my_confusion_matrix(array):
    return
    df_cm = pd.DataFrame(array, range(4), range(4))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.show()


def logistic_regression():
    df = pd.read_csv('train.csv')
    y = df['price_range']
    X = df.drop('price_range', axis = 1)

    # Split the data into test and train by using sklearn.
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

    # 42 is the Answer to the Ultimate Question of Life...
    lr = LogisticRegression(random_state = 42, max_iter=100000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred_lr)
    print("Logistic Regression has Etest accuracy:" + str(accuracy))
    print_my_confusion_matrix(metrics.confusion_matrix(y_test, y_pred_lr))


def random_forest():
    df = pd.read_csv('train.csv')
    y = df['price_range']
    X = df.drop('price_range', axis = 1)

    # Split the data into test and train by using sklearn.
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

    # 42 is the Answer to the Ultimate Question of Life...
    rc = RandomForestClassifier(random_state=42)
    rc.fit(X_train,y_train)
    y_pred_rc = rc.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred_rc)
    print("Random Forest has Etest accuracy:" + str(accuracy))
    print_my_confusion_matrix(metrics.confusion_matrix(y_test, y_pred_rc))
    


def support_vector_machines():
    df = pd.read_csv('train.csv')
    y = df['price_range']
    X = df.drop('price_range', axis = 1)

    # Split the data into test and train by using sklearn.
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    accuracy_svc = metrics.accuracy_score(y_test, y_pred_svc)

    print("SVM has Etest accuracy:" + str(accuracy_svc))
    print_my_confusion_matrix(metrics.confusion_matrix(y_test, y_pred_svc))


def k_neighbors():
    df = pd.read_csv('train.csv')
    y = df['price_range']
    X = df.drop('price_range', axis = 1)

    # Split the data into test and train by using sklearn.
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

    model = KNeighborsClassifier()
    model.fit(X_train,y_train)
    predicted= model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test,predicted)
    print("K-neighbors has Etest accuracy:" + str(accuracy))
    print_my_confusion_matrix(metrics.confusion_matrix(y_test, predicted))


def decision_tree():
    df = pd.read_csv('train.csv')
    y = df['price_range']
    X = df.drop('price_range', axis = 1)

    # Split the data into test and train by using sklearn.
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

    first_tree = DecisionTreeClassifier()
    first_tree.fit(X_train, y_train)
    y_pred=first_tree.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    print("Decision Tree has Etest accuracy:" + str(accuracy))
    print_my_confusion_matrix(metrics.confusion_matrix(y_test, y_pred))


#visualize_data()
#logistic_regression()
#random_forest()
#support_vector_machines()
#k_neighbors()
#decision_tree()
#plot_results()

