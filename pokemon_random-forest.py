from sklearn.ensemble import RandomForestClassifier
from ml_methods import plot_learning_curve
from ml_methods import dataset_divisor
from ml_methods import pre_processing
from ml_methods import random_forest
from ml_methods import cv_tests


def main():

    class_names = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire', 'ghost', 'grass',
                   'ground', 'ice', 'normal', 'poison', 'psyquic', 'rock', 'steel', 'water']

    # Coleta dos atributos e da classe na base de dados
    model_predictors, model_class = dataset_divisor()

    # Pré-Processamento
    model_predictors, model_class = pre_processing(model_predictors.copy(), model_class.copy())

    # Treinamento normal
    random_forest(model_predictors, model_class, class_names)

    # Treinamento com validação cruzada
    algorithm = RandomForestClassifier(criterion='entropy')
    cv_tests(model_predictors, model_class, algorithm)

    # Plot das curvas de aprendizado (Análise de overfitting e underfitting)
    # plot_learning_curve(model_predictors, model_class, algorithm, 40)


if __name__ == '__main__':
    main()
