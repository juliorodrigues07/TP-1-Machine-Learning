from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import os


def main():

    pokemon_dataset = pd.read_csv('pokemon.csv')

    model_predictors = pokemon_dataset.drop(['type1', 'type2'], axis='columns')
    model_class = pokemon_dataset['type1']

    l_encoder_predictors = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 24, 29, 30])], remainder='passthrough')
    model_predictors = l_encoder_predictors.fit_transform(model_predictors).toarray()

    imp = SimpleImputer(strategy="most_frequent")
    model_predictors = imp.fit_transform(model_predictors)

    l_encoder_classes = LabelEncoder()
    model_class = l_encoder_classes.fit_transform(model_class)

    scaler = StandardScaler()
    model_predictors = scaler.fit_transform(model_predictors)

    training_predictors, test_predictors, training_class, test_class = train_test_split(model_predictors, model_class, test_size=0.2, random_state=0)

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(training_predictors, training_class)
    predictions = classifier.predict(test_predictors)

    result = accuracy_score(test_class, predictions)
    #matrix = confusion_matrix(test_class, predictions)
    print(result)
    #print(matrix)


if __name__ == '__main__':
    main()
