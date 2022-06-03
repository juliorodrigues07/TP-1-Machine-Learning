from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore")


def dataset_divisor():

    pokemon_dateset = pd.read_csv('pokemon.csv')

    model_predictors = pokemon_dateset.drop(['type1', 'type2'], axis='columns')
    model_class = pokemon_dateset['type1']

    return model_predictors, model_class


def pre_processing(attributes, classes):

    # Discretiza os valores atributos qualitativos ('abilities, 'classification', 'japanese_name', 'name')
    l_encoder_attributes = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 24, 29, 30])],
                                             remainder='passthrough')
    attributes = l_encoder_attributes.fit_transform(attributes.copy()).toarray()

    # TODO: Ajustar estratégia de preenchimento de valores ausentes ('mean', 'median', 'most_frequent', 'constant')

    # Preenche missing values com a média
    fill_mv = SimpleImputer(strategy='mean')
    attributes = fill_mv.fit_transform(attributes.copy())

    # Descarta atributos irrelevantes (Ganho de informação pequeno ou nulo)
    # classifier = ExtraTreesClassifier(n_estimators=100)
    # classifier = classifier.fit(attributes, classes)

    # discard = SelectFromModel(classifier, prefit=True)
    # attributes = discard.transform(attributes.copy())

    # Discretiza os valores da classe
    l_encoder_classes = LabelEncoder()
    classes = l_encoder_classes.fit_transform(classes.copy())

    # Redistribuição dos dados
    scaler = StandardScaler()
    attributes = scaler.fit_transform(attributes.copy())

    return attributes, classes


def decision_tree(classifier, training_attributes, training_classes, test_attributes, test_classes, class_names):

    # Construção da árvore de decisão e treinamento do classificador
    classifier.fit(training_attributes, training_classes)
    predictions = classifier.predict(test_attributes)

    summary(class_names, test_classes, predictions)

    # Resultado
    result = accuracy_score(test_classes, predictions)
    return result


def ordinary_tests(attributes, classes, class_names):

    # TODO: Ajustar a proporção dos conjuntos de treinamento e teste para aplicar o modelo de aprendizado

    # Divisão do conjunto de treinamento e teste (80% e 20%)
    training_attributes, test_attributes, training_classes, test_classes = train_test_split(attributes,
                                                                                            classes,
                                                                                            test_size=0.2,
                                                                                            random_state=0)

    classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)

    result = decision_tree(classifier, training_attributes, training_classes, test_attributes, test_classes, class_names)
    print('\nTestes normais: ' + str(result))

    return classifier


def cv_tests(attributes, classes):

    classifier = tree.DecisionTreeClassifier(criterion='entropy')

    # TODO: Ajustar escolha do valor de k para aplicação de testes com validação cruzada
    scores = cross_val_score(estimator=classifier, X=attributes, y=classes, cv=10, n_jobs=-1)

    result = np.mean(scores)
    print('\nValidação Cruzada: ' + str(result) + '\n')


def random_forest(attributes, classes, class_names):

    training_attributes, test_attributes, training_classes, test_classes = train_test_split(attributes,
                                                                                            classes,
                                                                                            test_size=0.2,
                                                                                            random_state=0)

    # Construção da floresta aleatória e treinamento do classificador
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(training_attributes, training_classes)
    predictions = classifier.predict(test_attributes)

    summary(class_names, test_classes, predictions)

    result = accuracy_score(test_classes, predictions)
    print('\nFloresta Aleatória: ' + str(result) + '\n')


def summary(class_names, test_classes, predictions):

    # Exibe um resumo dos resultados da predição para cada rótulo e imprime a matriz de confusão
    print(classification_report(test_classes, predictions, target_names=class_names))
    print(confusion_matrix(test_classes, predictions))


def plot_per_column_distribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):

        plt.subplot(int(nGraphRow), int(nGraphPerRow), i + 1)
        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()

        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_correlation_matrix(df, graphWidth):

    filename = df.dataframeName
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]]

    if df.shape[1] < 2:
        return

    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Matriz de correlação ({filename})', fontsize=15)
    plt.show()


def info_plots(classifier):

    # TODO: Plotar gráfico das curvas de treinamento e teste
    pokemon_dataset = pd.read_csv('pokemon.csv')
    pokemon_dataset.dataframeName = 'pokemon.csv'

    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
    tree.plot_tree(classifier, filled=True)
    fig.savefig('tree.png')

    # plot_per_column_distribution(pokemon_dataset, 24, 4)
    plot_correlation_matrix(pokemon_dataset, 8)
