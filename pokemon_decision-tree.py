from ml_methods import dataset_divisor
from ml_methods import pre_processing
from ml_methods import ordinary_tests
from ml_methods import info_plots
from ml_methods import cv_tests


def main():

    class_names = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire', 'flying', 'ghost',
                   'grass', 'ground', 'ice', 'normal', 'poison', 'psyquic', 'rock', 'steel', 'water']

    # Divisão da base de dados entre atributos e classe
    model_predictors, model_class = dataset_divisor()

    # Pré-Processamento
    model_predictors, model_class = pre_processing(model_predictors.copy(), model_class.copy())

    # Treinamento normal
    classifier = ordinary_tests(model_predictors, model_class, class_names)

    # Treinamento com validação cruzada
    cv_tests(model_predictors, model_class)

    # Plots da árvore de decisão, matriz de correlação e afins
    # info_plots(classifier)


if __name__ == '__main__':
    main()
