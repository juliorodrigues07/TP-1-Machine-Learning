from sklearn.tree import DecisionTreeClassifier
from ml_methods import plot_learning_curve
from ml_methods import dataset_divisor
from ml_methods import pre_processing
from ml_methods import ordinary_tests
from ml_methods import info_plots
from ml_methods import cv_tests


def main():

    class_names = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire', 'flying', 'ghost',
                   'grass', 'ground', 'ice', 'normal', 'poison', 'psyquic', 'rock', 'steel', 'water']

    # Coleta dos atributos e da classe na base de dados
    model_predictors, model_class = dataset_divisor()

    # Pré-Processamento
    model_predictors, model_class = pre_processing(model_predictors.copy(), model_class.copy())

    # Treinamento normal
    classifier = ordinary_tests(model_predictors, model_class, class_names)

    # Treinamento com validação cruzada
    algorithm = DecisionTreeClassifier(criterion='entropy')
    cv_tests(model_predictors, model_class, algorithm)

    # Plots da árvore de decisão, matriz de correlação e afins
    # info_plots(classifier)

    # Plot das curvas de aprendizado (Análise de overfitting e underfitting)
    # plot_learning_curve(model_predictors, model_class, algorithm, 40)


if __name__ == '__main__':
    main()
