# DIABETES-PREDICTION-USING-CLASSIFICATION-TECHNIQUES
This model proposes a diabetes prediction algorithm based on XGBoost algorithm with the numerical features being separated while some important features are extracted from the text features of experiment data. Experiment results show that accuracy of diabetes prediction based the improved XGBoost Classifier, Decision Tree, KNN Classifier, SVM Classifier, DF Classifier, Ada Boost, ANN with features combination is 80.2%, which is feasible and effective method for diabetes prediction.

IMPORTANT !
we are performing the classification on the diabetes identification using XGBoost Classifier, Decision Tree, KNN Classifier, SVM Classifier, DF Classifier, Ada Boost, ANN of deep learning along with the machine learning methods. As health analysis based approaches for diabetic detection. Hence, proper classification is important for the proper diabetes .

This code is purely for research and statistics.

HOW TO USE !
Download the csv.dataset file and extract into the repository folder.
You need to install some packages to execute your project in a proper way.
10. Open the command prompt/ anaconda prompt or terminal as administrator. 
11.  The prompt will get open, with specified path, type “pip install package name” which you want to install (like NumPy, pandas, sea born, scikit-learn, Matplotlib, Pyplot)
Ex: Pip install NumPy
You need to train the each and evrey classification model like
model = XGBClassifier()
            model.fit(x_train, y_train)
            pred= model.predict(x_test)
            xgb= accuracy_score(pred, y_test)
            
To run the .py file use the command line in the console python(app.py) .

ACKNOWLEDGED/CITED IN:

Ren, Q., Cheng, H., Han, H.: Research on machine learning framework based on random forest algorithm, AIP Conference Proceedings, vol. 1820, 2017.

M. M. Islam, H. Iqbal, M. R. Haque and M. K. Hasan, “Prediction of Breast Cancer Using Support Vector Machine and K-Nearest Neighbors,” IEEE Region 10 Humanitarian Technology Conference (R10-HTC), 2017.

S. H. Ripon, “Rule induction and prediction of chronic kidney dis-ease using boosting classifiers, Ant-Miner and J48 Decision Tree,” inProc. Int. Conf. Elect., Comput. Commun. Eng. (ECCE), Cox’s Bazar, Bangladesh, 2019, pp. 1–6.

T. Chen and C. Guestrin, “XGBOOST: A scalable tree boosting system,”inProc. 22nd ACMSIGKDD Int. Conf. Knowl. Discovery Data Mining, 2016, pp. 785–794.
