import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


def decision_tree():
    df = pd.read_csv("Pokemontree.csv", sep=';')
    # print(df)
    inputs = df.drop('Over', axis='columns')
    target = df['Over']
    # print(target)
    le_type1 = LabelEncoder()
    le_type2 = LabelEncoder()
    inputs['type1_new'] = le_type1.fit_transform(inputs['Type_1'])
    inputs['type2_new'] = le_type2.fit_transform(inputs['Type_2'])
    inputs_n = inputs.drop(['Type_1', 'Type_2'], axis='columns')
    model = tree.DecisionTreeClassifier()
    model.fit(inputs_n, target)
    model.score(inputs_n, target)
    model.predict([3, 3]) #grass, poison


def main():
    decision_tree()


if __name__ == "__main__":
    main()