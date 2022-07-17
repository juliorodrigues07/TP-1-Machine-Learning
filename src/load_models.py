from ml_methods import train_test_split
from ml_methods import dataset_divisor
from ml_methods import accuracy_score
from ml_methods import pre_processing
from ml_methods import model_dir
from ml_methods import summary
import joblib
import os


model_name = 'RandomForest_model_2022-07-17_0_28_9.sav'


def main():

    class_names = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire', 'ghost', 'grass',
                   'ground', 'ice', 'normal', 'poison', 'psyquic', 'rock', 'steel', 'water']

    # Coleta dos atributos e da classe na base de dados
    model_predictors, model_class = dataset_divisor()

    # Pré-Processamento
    attributes, classes = pre_processing(model_predictors, model_class)

    _, test_attributes, _, test_classes = train_test_split(attributes, classes, test_size=0.2, random_state=1)

    models = os.listdir(os.getcwd() + model_dir)

    for model in models:
        clf = joblib.load(os.getcwd() + model_dir + model)
        predictions = clf.predict(test_attributes)
        result = accuracy_score(test_classes, predictions)
        summary(class_names, test_classes, predictions)
        print('\nAcurácia: ' + str(result) + '\n')


if __name__ == '__main__':
    main()
