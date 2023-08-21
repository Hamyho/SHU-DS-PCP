import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.transform import rotation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import ClusterCentroids
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from yellowbrick.classifier import ROCAUC
from sklearn.tree import DecisionTreeClassifier

import pickle


# print(source_data.info())


# print(source_data[source_data['activity'].isnull() == False])

class Decorator:
    @classmethod
    def blank_ranges(self, func):
        def wrapper(self, ranges=None):
            if ranges is None:
                return func(self, self.source_data.select_dtypes(include='number').columns)
            else:
                return func(self, ranges)

        return wrapper


class EDA:

    def __init__(self, source_data=None):
        self.source_data = source_data
        self.X = self.source_data.drop('activity', axis=1)
        self.y = self.source_data['activity']
        self.y_train = None
        self.X_train = None
        self.y_test = None
        self.X_test = None

    def drop_none(self):
        self.source_data.dropna(how='any', inplace=True)

    @Decorator.blank_ranges
    def mean(self, ranges):
        return self.source_data[ranges].mean(axis=0)

    @Decorator.blank_ranges
    def median(self, ranges):
        return self.source_data[ranges].median(axis=0)

    @Decorator.blank_ranges
    def std(self, ranges):
        return self.source_data[ranges].std(axis=0)

    @Decorator.blank_ranges
    def var(self, ranges):
        return self.source_data[ranges].var(axis=0)

    @Decorator.blank_ranges
    def min(self, ranges):
        return self.source_data[ranges].min(axis=0)

    @Decorator.blank_ranges
    def max(self, ranges):
        return self.source_data[ranges].max(axis=0)

    @Decorator.blank_ranges
    def skew(self, ranges):
        return self.source_data[ranges].skew(axis=0)

    @Decorator.blank_ranges
    def kurt(self, ranges):
        return self.source_data[ranges].kurt(axis=0)

    def frequency(self):

        frequencies = self.y.value_counts()

        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=200)
        frequencies.plot(kind='bar', ax=ax)
        ax.set_title('Frequency')
        plt.xticks(fontproperties=FontProperties(size=4), rotation=90)
        plt.yticks(fontproperties=FontProperties(size=4))
        plt.tight_layout()

        return ax

    def dependencies(self):
        cor_matrix = self.source_data.corr(numeric_only=True).round(2)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        heatmap = sns.heatmap(cor_matrix, annot=True, annot_kws={'size': 4}, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        plt.xticks(fontproperties=FontProperties(size=4), rotation=90)
        plt.yticks(fontproperties=FontProperties(size=4))
        plt.tight_layout()
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=4)
        return fig

    def rms(self, ranges):
        rows = self.X[ranges].shape[0]
        s = self.X[ranges].pow(2)
        s = s.sum(axis=0).div(rows)
        return s.apply(np.sqrt)

    @Decorator.blank_ranges
    def zero_crossing(self, ranges=None):
        zc = np.where(np.sign(self.X[ranges].values[:-1]) != np.sign(self.X[ranges].values[1:]))[0]

        return ' '.join(str(x) if (i + 1) % 20 != 0 else str(x) + '\n' for i, x in enumerate(zc))

    def sos(self):
        s = (self.X - self.X.mean().T) ** 2
        return s.sum(axis=0)

    def cov(self, ranges):
        return self.source_data[ranges].cov()

    def info(self, ranges):
        statstic_data = {
            'Mean': self.mean(ranges),
            'Standard Deviation': self.std(ranges),
            'Median': self.median(ranges),
            'Variance': self.var(ranges),
            'Minimum': self.min(ranges),
            'Maximum': self.max(ranges),
            'Skewness': self.skew(ranges),
            'Kurtosis': self.kurt(ranges),
            # 'rms': self.rms(),
            # 'sos': self.sos(),
        }
        return pd.DataFrame(statstic_data)

    def pre_unbalanced_classification(self, max_numbers_sample=1500):
        label_encoder = LabelEncoder()
        self.source_data['activity'] = label_encoder.fit_transform(self.source_data['activity'])
        self.y = self.source_data['activity']
        numbers_activities = self.source_data['activity'].value_counts()
        oversampled_data = []
        undersampled_data = []
        print(numbers_activities)
        for label, number in numbers_activities.items():
            if number < max_numbers_sample:
                oversampled_sample = np.random.choice(self.X[self.y == label].index, size=max_numbers_sample,
                                                      replace=True)

                oversampled_data.append(self.source_data.loc[oversampled_sample])
            elif number > max_numbers_sample:
                undersampled_sample = np.random.choice(self.X[self.y == label].index, size=max_numbers_sample,
                                                       replace=False)
                undersampled_data.append(self.source_data.loc[undersampled_sample])
            else:
                oversampled_data.append(self.source_data[self.y == label])
                undersampled_data.append(self.source_data[self.y == label])
        oversampled_df = pd.concat(oversampled_data)
        undersampled_df = pd.concat(undersampled_data)
        df = pd.concat([oversampled_df, undersampled_df])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.source_data = df
        self.source_data['activity'] = label_encoder.inverse_transform(self.source_data['activity'])
        self.X = self.source_data.drop(['activity'], axis=1)
        self.y = self.source_data['activity']

    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25,
                                                                                random_state=101)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def model_knn(self):
        knn = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': list(range(1, 20))
        }

        grid = GridSearchCV(knn, param_grid=param_grid)
        grid.fit(self.X_train, self.y_train)
        print(f'Best parameters for knn model is: \n{grid.best_params_}')

        # save to file
        with open('model_knn.pkl', 'wb') as file:
            pickle.dump(grid, file)

        return f'Best parameters for knn model is: \n{grid.best_params_} \n'

    def model_logistic(self):
        log_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=10)
        penalty = ['l1', 'l2', 'elasticnet']

        l1_ratio = np.linspace(0, 1, 10)
        C = np.logspace(0, 10, 10)
        param_grid = {
            'penalty': penalty,
            'l1_ratio': l1_ratio,
            'C': C
        }
        grid = GridSearchCV(log_model, param_grid=param_grid)
        grid.fit(self.X_train, self.y_train)
        # check best model
        print(f'Best parameters for logistic model is: \n{grid.best_params_}')
        # predict

        # save to file
        with open('model_logistic.pkl', 'wb') as file:
            pickle.dump(grid, file)

        # plot roc curve
        visualizer = ROCAUC(grid, classes=grid.classes_)
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        visualizer.show()

        return f'Best parameters for logistic model is: \n{grid.best_params_}\n'

    def model_svm(self):
        svm = SVC()
        param_grid = {
            'C': [1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        grid = GridSearchCV(svm, param_grid)
        grid.fit(self.X_train, self.y_train)

        with open('model_svm.pkl', 'wb') as file:
            pickle.dump(grid, file)
        return f'Best parameters for svm model is: \n{grid.best_params_}\n'

    def model_random_forest(self):
        param_grid = {
            'n_estimators': [80, 100, 120],
            'max_features': ['auto', 'log2'],
            'bootstrap': [True, False],
            'oob_score': [True, False],
            'random_state': [101]
        }
        rfc = RandomForestClassifier()
        grid = GridSearchCV(rfc, param_grid)
        grid.fit(self.X_train, self.y_train)
        print(f'Best parameters for Random Forest is \n {grid.best_params_}\n')

        # save file
        with open('model_randomforest.pkl', 'wb') as file:
            pickle.dump(grid, file)
        return f'Best parameters for Random Forest model is: \n{grid.best_params_}\n'

    def model_multi_layer_perception_neural_network(self):
        mlp = MLPClassifier()
        param_grid = {
            'hidden_layer_sizes': [50, 100, 150],
            'activation': ['identity', 'relu', 'tanh', 'logistic'],
            'max_iter': [100, 150, 200]
        }
        grid = GridSearchCV(mlp, param_grid)
        grid.fit(self.X_train, self.y_train)
        print(f'Best parameters for MLP Neural Network is {grid.best_params_}')

        # save file
        with open('model_mlp.pkl', 'wb') as file:
            pickle.dump(grid, file)
        return f'Best parameters for MLP model is: \n{grid.best_params_}\n'

    def model_decision_tree(self):
        decision_tree = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ["best", "random"],
            'max_depth': [2, 10, 13, 16]
        }
        grid = GridSearchCV(decision_tree, param_grid)
        grid.fit(self.X_train, self.y_train)
        print(f'Best parameters for Decision Tree is {grid.best_params_}')

        # save file
        with open('model_decisiontree.pkl', 'wb') as file:
            pickle.dump(grid, file)
        return f'Best parameters for Decision Tree model is: \n{grid.best_params_}\n'

    def evaluation(self, model_name, grid, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        fig = plt.figure(figsize=(4, 2), dpi=200)
        ax = fig.add_subplot(111)
        sns.heatmap(cm, annot=True,
                    fmt='d',
                    cmap='OrRd',
                    cbar=False,
                    xticklabels=grid.classes_,
                    yticklabels=grid.classes_[::-1],
                    annot_kws={"fontproperties": FontProperties(size=5)}
                    )

        plt.xticks(fontproperties=FontProperties(size=4), rotation=90)
        plt.yticks(fontproperties=FontProperties(size=4))
        plt.tight_layout()
        return fig

    def report(self, Y_true, Y_predict):
        return pd.DataFrame(classification_report(y_true=Y_true, y_pred=Y_predict, output_dict=True)).transpose().round(
            3)

    def clean(self):
        self.source_data.dropna()
        self.source_data.drop_duplicates()
        self.pre_unbalanced_classification()
        self.split_train_test()

# source_data = pd.read_csv('activity_context_tracking_data.csv')
# source_data = source_data.drop('_id', axis=1)
#
# X = source_data.drop('activity', axis=1)
# y = source_data['activity']
# eda = EDA(source_data)
# print(eda.cov(['orX','orY']))
# eda.frequency()
# eda.dependencies()
# eda.pre_unbalanced_classification()
# eda.split_train_test()
# print(eda.y.value_counts())
# eda.model_multi_layer_perception_neural_network()
