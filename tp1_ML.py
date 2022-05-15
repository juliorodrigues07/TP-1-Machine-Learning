from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import os


def main():

    pokemon_dataset = pd.read_csv('pokemon.csv')

    model_predictors = pokemon_dataset.drop(['type1', 'type2'], axis='columns')
    model_class = pokemon_dataset['type1']

    # Discretiza os valores atributos qualitativos ('abilities, 'classification', 'japanese_name', 'name')
    l_encoder_predictors = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 24, 29, 30])], remainder='passthrough')
    model_predictors = l_encoder_predictors.fit_transform(model_predictors).toarray()

    # Preenche missing values com a média
    fill_vm = SimpleImputer(strategy='mean')
    model_predictors = fill_vm.fit_transform(model_predictors)

    # Discretiza os valores da classe
    l_encoder_classes = LabelEncoder()
    model_class = l_encoder_classes.fit_transform(model_class)

    # "Pré-processamento" dos dados
    scaler = StandardScaler()
    model_predictors = scaler.fit_transform(model_predictors)

    # Divisão do conjunto de treinamento e teste (80% e 20%)
    training_predictors, test_predictors, training_class, test_class = train_test_split(model_predictors, model_class, test_size=0.2, random_state=0)

    # Construção da árvore de decisão e treinamento do classificador
    classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(training_predictors, training_class)
    predictions = classifier.predict(test_predictors)

    # Resultado
    result = accuracy_score(test_class, predictions)
    print(result)

    #fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
    #tree.plot_tree(classifier, filled=True)
    #fig.savefig('tree.png')


if __name__ == '__main__':
    main()
