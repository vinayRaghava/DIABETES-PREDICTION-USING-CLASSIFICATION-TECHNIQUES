# DIABETES-PREDICTION-USING-CLASSIFICATION-TECHNIQUES
This paper proposes a diabetes prediction algorithm based on XGBoost algorithm with the numerical features being separated while some important features are extracted from the text features of experiment data. Experiment results show that accuracy of diabetes prediction based the improved XGBoost Classifier, Decision Tree, KNN Classifier, SVM Classifier, DF Classifier, Ada Boost, ANN with features combination is (80.2%) which is feasible and effective method for diabetes prediction.
IMPORTANT
performing the classification on the diabetes identification using XGBoost Classifier, Decision Tree, KNN Classifier, SVM Classifier, DF Classifier, Ada Boost, ANN of deep learning along with the machine learning methods. As health analysis based approaches for diabetic detection. Hence, proper classification is important for the proper diabetes .

This Code is Purely for research and statistics.
Results will shows the values of particular symptoms to know the diabetes and to take precautions.


HOW TO USE!

*Download the csv.dataset and extract into the repository folder.
You need to install some packages to execute your project in a proper way.
10. Open the command prompt/ anaconda prompt or terminal as administrator. 
11.  The prompt will get open, with specified path, type “pip install package name” which you want to install (like NumPy, pandas, sea born, scikit-learn, Matplotlib, Pyplot)
Ex: Pip install NumPy
To run the model .
 model_dt =  DecisionTreeClassifier()
            model_dt.fit(x_train, y_train)
            pred = model_dt.predict(x_test)
            dt = accuracy_score(pred, y_test)
            cr = classification_report(pred, y_test)
            print(cr)
           
FUTURE SCOPE!
This can be utilized in future to classify the types of different Deficiencies easily that which can tend to easy to Predicated the diabetes in early stages and can take the initial curing of human and take measures to not affect.

ACKNOWLEDGEED/CITED IN
P. M. S. Sai, G. Anuradha, P. kumar, “Survey on Type 2 Diabetes Prediction Using Machine Learning,” Proceedings of the Fourth International Conference on Computing Methodologies and Communication (ICCMC), 2020.

D. Sisodia, D.S. Sisodia, “Prediction of Diabetes using Classification Algorithms”, International Conference on Computational Intelligence and Data Science (ICCIDS), Procedia Computer Science, Vol. 132, pp. 1578–1585, 2018.

S Das, A Mishra, P Roy – 2019, “Automatic Diabetes Prediction Using Tree Based Ensemble Learners”, International Conference on Computational Intelligence &IoT(ICCIIoT), 2018.

Wei S, Zhao X, Miao C. A comprehensive exploration to the machine learning techniques for diabetes identification. In Internet of Things (WF-IoT), 2018 IEEE 4th World Forum, pp. 291-295, 5 Feb, 2018.

            
            

