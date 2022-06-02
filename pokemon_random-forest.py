from ml_methods import dataset_divisor
from ml_methods import pre_processing
from ml_methods import random_forest


def main():

    class_names = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire', 'ghost',
                   'grass', 'ground', 'ice', 'normal', 'poison', 'psyquic', 'rock', 'steel', 'water']

    # Divisão da base de dados entre atributos e classe
    model_predictors, model_class = dataset_divisor()

    # Pré-Processamento
    model_predictors, model_class = pre_processing(model_predictors.copy(), model_class.copy())

    # Treinamento do modelo
    random_forest(model_predictors, model_class, class_names)


if __name__ == '__main__':
    main()
