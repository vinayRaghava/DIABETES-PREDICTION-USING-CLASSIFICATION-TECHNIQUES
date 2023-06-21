import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from flask import Flask, render_template, request, flash
from flask import *


app= Flask(__name__)

app.config['UPLOAD_FOLDER']= r'C:\Users\YMTS0519\Documents\MAY-2022\DIABETES PREDICTION\CODE\uploads'
app.config['SECRET_KEYS']='python'

global df,x_train, x_test, y_train, y_test

df= pd.read_csv('C:\\Users\\YMTS0519\\Documents\\MAY-2022\\DIABETES PREDICTION\\CODE\\DATASET\\diabetes.csv')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load', methods= ['POST', 'GET'])
def load():
    global df, dataset
    if request.method =='POST':
        file= request.files['file']
        df= pd.read_csv(file)
        dataset= df.head(100)
        msg= 'DATA LOADED SUCCESSFULLY'

        return render_template('load.html', msg=msg)
    return render_template('load.html')



@app.route('/view')
def view():

    print(dataset)
    print(dataset.head())
    print(dataset.columns)
    return render_template('view.html', columns= dataset.columns.values, rows= dataset.values.tolist())
    return render_template('view.html')



@app.route('/preprocess', methods= ['POST', 'GET'])
def preprocess():
    global x,y, size, x_train, x_test, y_train, y_test
    if request.method== 'POST':
        size= int(request.form['split'])
        size= size/100

        le = LabelEncoder()

        df['Gender'] = le.fit_transform(df['Gender'])
        df['Polyuria'] = le.fit_transform(df['Polyuria'])
        df['Polydipsia'] = le.fit_transform(df['Polydipsia'])
        df['sudden weight loss'] = le.fit_transform(df['sudden weight loss'])
        df['weakness'] = le.fit_transform(df['weakness'])
        df['Polyphagia'] = le.fit_transform(df['Polyphagia'])
        df['Genital thrush'] = le.fit_transform(df['Genital thrush'])
        df['visual blurring'] = le.fit_transform(df['visual blurring'])
        df['Itching'] = le.fit_transform(df['Itching'])
        df['Irritability'] = le.fit_transform(df['Irritability'])
        df['delayed healing'] = le.fit_transform(df['delayed healing'])
        df['partial paresis'] = le.fit_transform(df['partial paresis'])
        df['muscle stiffness'] = le.fit_transform(df['muscle stiffness'])
        df['Alopecia'] = le.fit_transform(df['Alopecia'])
        df['Obesity'] = le.fit_transform(df['Obesity'])
        df['class'] = le.fit_transform(df['class'])


        x= df.iloc[:,:-1]
        y= df.iloc[:,-1]

        x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state= 42)

        return render_template('preprocess.html', msg= 'DATA PREPROCESSED AND SPLITTED SUCCESSFULLY')
    return render_template('preprocess.html')


@app.route('/model', methods= ['POST', 'GET'])
def model():
    global model_dt
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    if request.method== 'POST':
        model_no = int(request.form['algo'])

        if model_no == 0:
            msg= "You've Not Selected Any Model"


        elif model_no == 1:
            model = XGBClassifier()
            model.fit(x_train, y_train)
            pred= model.predict(x_test)
            xgb= accuracy_score(pred, y_test)
            cr= classification_report(pred, y_test)
            print(cr)

            msg= "THE ACCURACY OF XGBOOST CLASSIFIER IS: " + str(xgb*100) + str('%')

            return render_template('model.html', msg = msg)


        elif model_no ==2:
            model = KNeighborsClassifier()
            model.fit(x_train, y_train)
            pred= model.predict(x_test)
            knn= accuracy_score(pred, y_test)
            cr= classification_report(pred, y_test)
            print(cr)

            msg= "THE ACCURACY OF KNEIGHBORS CLASSIFIER IS: " + str(knn*100) + str('%')

            return render_template('model.html', msg=msg)

        elif model_no == 3:
            model = SVC(kernel='linear')
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            svm = accuracy_score(pred, y_test)
            cr = classification_report(pred, y_test)
            print(cr)

            msg = "THE ACCURACY OF SUPPORT VECTOR CLASSIFIER IS: " + str(svm*100) + str('%')

            return render_template('model.html', msg=msg)


        elif model_no == 4:
            model_dt =  DecisionTreeClassifier()
            model_dt.fit(x_train, y_train)
            pred = model_dt.predict(x_test)
            dt = accuracy_score(pred, y_test)
            cr = classification_report(pred, y_test)
            print(cr)

            msg = "THE ACCURACY OF DECISION TREE CLASSIFIER IS: " + str(dt*100) + str('%')

            return render_template('model.html', msg=msg)


        elif model_no == 5:
            clf =AdaBoostClassifier(base_estimator= model_dt,
                                    n_estimators=100,
                                    learning_rate=0.0005,
                                    algorithm = 'SAMME',
                                    random_state=1)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            adb = accuracy_score(pred, y_test)
            cr = classification_report(pred, y_test)
            print(cr)

            msg = "THE ACCURACY OF ADABOOST CLASSIFIER IS: " + str(adb*100) + str('%')

            return render_template('model.html', msg=msg)


        elif model_no == 6:
            model =  Sequential()
            model.add(Dense(12, input_shape=(x_train.shape), activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=100, batch_size=50)
            predictions = model.predict(x_test)
            acc = model.evaluate(x_test, y_test)

            msg = "Accuracy of Artificial Neural Network is: "+ str(round(acc[1]*100, 2)) + str('%')

            return render_template('model.html', msg=msg)

    return render_template('model.html')

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    global df, x, y, size, x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    if request.method== 'POST':
        Age = request.form['Age']
        print(Age)
        Gender = request.form['Gender']
        print(Gender)
        Polyuria = request.form['Polyuria']
        print(Polyuria)
        Polydipsia = request.form['Polydipsia']
        print(Polydipsia)
        sudden_weight_loss = request.form['sudden weight loss']
        print(sudden_weight_loss)
        weakness = request.form['weakness']
        print(weakness)
        Polyphagia = request.form['Polyphagia']
        print(Polyphagia)
        Genital_thrush = request.form['Genital thrush']
        print(Genital_thrush)
        visual_blurring = request.form['visual blurring']
        print(visual_blurring)
        Itching = request.form['Itching']
        print(Itching)
        Irritability = request.form['Irritability']
        print(Irritability)
        delayed_healing = request.form['delayed healing']
        print(delayed_healing)
        partial_paresis = request.form['partial paresis']
        print(partial_paresis)
        muscle_stiffness = request.form['muscle stiffness']
        print(muscle_stiffness)
        Alopecia = request.form['Alopecia']
        print(Alopecia)
        Obesity = request.form['Obesity']
        print(Obesity)


        di= {'Age': [Age], 'Gender': [Gender],
              'Polyuria': [Polyuria], 'Polydipsia': [Polydipsia],
              'sudden weight loss': [sudden_weight_loss], 'weakness': [weakness],
              'Polyphagia': [Polyphagia], 'Genital thrush': [Genital_thrush],
              'visual blurring': [visual_blurring], 'Itching': [Itching],
              'Irritability': [Irritability], 'delayed healing': [delayed_healing], 'partial paresis': [partial_paresis],
              'muscle stiffness': [muscle_stiffness],
              'Alopecia': [Alopecia], 'Obesity': [Obesity]}


        test= pd.DataFrame.from_dict(di)
        print(test)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        model = XGBClassifier()
        model.fit(x_train, y_train)
        output = model.predict(test)
        print(output)

        if output[0] == 1:
            msg = 'THE PERSON IS HAVING DIABETES'

        else:
            msg = 'THE PERSON IS NOT HAVING DIABETES'

        return render_template('prediction.html', msg=msg)
    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug = True)






















