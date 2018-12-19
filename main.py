import pandas as panda
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt


def linearPredictionPokemon():
    df = panda.read_csv("Pokemonedit.csv", sep=';')
    print(df)
    reg = linear_model.LinearRegression()
    reg.fit(df[['HP', 'Attack', 'Defense']], df.CP)
    print(reg.coef_)
    print("\n")
    print(reg.intercept_)
    y_pred = reg.predict(df[['HP', 'Attack', 'Defense']])
    meanerror = mean_squared_error(df.CP, y_pred)
    print('Variance score: %.2f' % r2_score(df.CP, y_pred))
    # plt.scatter(df['#'], df.CP, color='black')
    # plt.plot(df['#'], y_pred, color = 'red')
    #
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()


def main():
    linearPredictionPokemon()


if __name__ == "__main__":
    main()
