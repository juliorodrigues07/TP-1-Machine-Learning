
from ml_methods import plot_learning_curve
from ml_methods import dataset_divisor
from ml_methods import pre_processing
from ml_methods import grandient_boosting
from ml_methods import cv_tests
from sklearn.ensemble import GradientBoostingClassifier





def main():

    class_names = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire', 'ghost', 'grass',
                   'ground', 'ice', 'normal', 'poison', 'psyquic', 'rock', 'steel', 'water']

    # Coleta dos atributos e da classe na base de dados
    model_predictors, model_class = dataset_divisor()

    # Pré-Processamento
    model_predictors, model_class = pre_processing(model_predictors.copy(), model_class.copy())
    
    #Treinamento normal
    grandient_boosting(model_predictors, model_class)

    #Validação cruzada
    algorithm = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    cv_tests(model_predictors, model_class, algorithm)



if __name__ == '__main__':
    main()
