""" Joshua Tran 
    ITP-449
    H08
    This code finds the attribute that is best correlated with Progression 
    and runs a linear regression algorithim and plots a line of best fit for it.
"""
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def main():
    file_path = "diabetes.csv"
    diabetes_df = pd.read_csv(file_path, skiprows = 1)

    # column_names = diabetes_df.columns
    # fig, ax = plt.subplots(len(column_names), len(column_names), figsize = (16,9))
    
    # rows, col = 0, 0
    # for row in range(len(column_names)):
    #     for col in range(len(column_names)):
    #         if row == col:
    #             ax[row, col].hist(diabetes_df.iloc[:, row])
    #         else:
    #             # scatterplot the current row, column
    #             ax[row, col].scatter(diabetes_df.iloc[:, row], diabetes_df.iloc[:, col], s=12)#, edgecolors='white')
    # plt.savefig("diabetes_scatterplot.png")
    # print(column_names)
    diabetes_df = diabetes_df.drop_duplicates()
    corr_matrix = diabetes_df.corr(numeric_only=True)
    diabetes_df = diabetes_df[["BMI", "Y"]]

    print(corr_matrix)
    print(diabetes_df)

    y = diabetes_df["Y"]
    x = diabetes_df["BMI"]
    X = x.values.reshape(-1, 1)
    X = pd.DataFrame(X, columns=[x.name])

    model_linreg = LinearRegression()
    model_linreg.fit(X, y)

    X_trend = np.array([[x.min()], [x.max()]])
    y_pred = model_linreg.intercept_ + model_linreg.coef_[0]*X_trend
    y_pred = model_linreg.predict(X_trend)

    # model_linreg.

    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    ax.scatter(X, y, label= "Diabetes data")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Progression")
    ax.set_title("Diabetes data: Progression vs BMI (Linear Regression)")
    ax.plot(X_trend, y_pred, label='Line of best fit', color='orange')
    ax.legend()

    plt.savefig('Diabetes_data_progression_vs_BMI')

if __name__ == '__main__':
    main()