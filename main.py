import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import SimpleImputer

if __name__ == '__main__':

    train = pd.read_csv('training_set.csv')
    test = pd.read_csv('testing_set.csv')
    #dTrain = pd.DataFrame(train)
    #dTest = pd.DataFrame(test)
    trainCat = pd.DataFrame(train(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'property_Area']))

    testCat = pd.DataFrame(test(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'property_Area']))

    # print(dTrain.head(), trainCat.head())
    # print(dTrain.isna().sum(), trainCat.isna().sum())

    imputerCat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    trainCat = pd.DataFrame(imputerCat.fit(trainCat), columns=imputerCat.get_feature_names_out())
    trainCat
    #print(trainCat.isna().sum(), trainCat.head())

    imputerNum = SimpleImputer(missing_values=np.nan, strategy='mean')
    #trainNum = pd.DataFrame(imputerNum.fit_transform(trainNum), columns=imputerNum.get_feature_names_out())
    # print(trainNum.isna().sum(), trainNum.head())

    dFTrain = pd.concat([trainCat, trainNum], axis=1)
    #print(dFTrain.head())



    # target_row = loan_status

    # 1. Allgemeine Information
    # Datenset: training_set.csv, testing_set.csv

    # 2. Geschäftsverständnis
    # Ziel: Vorhersage ob Kreditnehmer kreditwürdig ist oder nicht

    # 3. Datenverständnis
    trainC = dTrain.columns
    # print(trainC)
    # print(dTrain.Loan_Status.value_counts())

    countplot = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "property_Area"]
    plt.figure(figsize=(12, 6))  # Width, height in inches.
    x = 1
    for i in countplot:
        plt.subplot(2, 3, x)
        #
        sb.countplot(x=i, hue="Loan_Status", data=dTrain)
        x += 1
        # plt.show()

    # Fehlende Daten (Anzahl je Spalte)
    # print(dTrain.isnull().sum())

    # trainCat = dTrain.drop(["ApplicantIncome", "CoapplicantIncome", "LoanAmount"], axis=1)
    # print(trainCat.head())

    # imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # trainCat = imputer.fit_transform(trainCat)

    # print(trainCat.Gender.value_counts())

# 4. Datenaufbereitung
