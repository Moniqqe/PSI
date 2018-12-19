import pandas as pd
from sklearn import linear_model


def singleregresion():
    df = pd.read_csv('Pokemontree.csv')
    HP = df[['HP']]
    CP = df.CP
    reg = linear_model.LinearRegression()
    reg.fit(HP, CP)
    hp_df = pd.read_csv("hp.csv")
    p = reg.predict(hp_df)
    hp_df['CP'] = p


def main():
    singleregresion()


if __name__ == "__main__":
    main()

