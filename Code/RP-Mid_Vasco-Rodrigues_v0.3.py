import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import kruskal
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from scipy.spatial.distance import mahalanobis

def MainMenu():
    print("----------Main Menu----------")
    print("1 - Minimum Distance Classifier - Euclidean\n"
    "2 - Minimum Distance Classifier - Mahalanobis\n"
    "3 - Fisher LDA\n"
    "0 - Exit")

def MainMenuOptions(option, X, T):
    if option == "1":
        x_mdce = X

        x_mdce = chooseNormalize(x_mdce)

        x_mdce = chooseKruskal(x_mdce)

        x_mdce = choosePCA(x_mdce)

        print("Calculating predictions with cross-validation...")
        classifier = MinimumDistanceClassifier()
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1 = cross_validate_model(x_mdce, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")
    elif option =="2":
        x_mdcm = X

        x_mdcm = chooseNormalize(x_mdcm)

        x_mdcm = chooseKruskal(x_mdcm)

        x_mdcm = choosePCA(x_mdcm)

        print("Calculating predictions with cross-validation...")
        classifier = MahalanobisMinimumDistanceClassifier()
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1 = cross_validate_model(x_mdcm, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")
    elif option == "3":
        x_fisher = X

        x_fisher = chooseNormalize(x_fisher)

        x_fisher = chooseKruskal(x_fisher)
        
        x_fisher = choosePCA(x_fisher)

        print("Calculating predictions...")
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1 = cross_validate_fld(x_fisher, T, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")
    elif option == "0":
        print("Exiting...")
    else:
        print("Invalid Option")

def chooseNormalize(x):
    optionNormalized = input("Do you want the data to be normalized? (y/n): ")
    if optionNormalized == "y":
        x = normalize_data(X, method='z-score')
        print("Data normalized")
    elif optionNormalized == "n":
        print("Data not normalized")
    else:
        print("Invalid option. By default, data won't be normalized.")

    return x

def chooseKruskal(x):
    optionKruskal = input("Do you want to apply the Kruskal-Walis test? (y/n)")
    if optionKruskal == "y":

        print("Applying Kruskal-Wallis test to select features...")
        h_statistics = {}
        for column in range(x.shape[1]):  # Itera sobre as colunas de x
            grouped_data = [x[T == group, column] for group in np.unique(T)]
            h_stat, p_value = kruskal(*grouped_data)
            h_statistics[column] = h_stat
        
        ranked_features = sorted(h_statistics.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFeatures ranked by Kruskal-Wallis H statistic:")
        for feature, h_stat in ranked_features:
            print(f"Feature {feature}: H = {h_stat:.4f}")
        
        top_n = int(input("Enter the number of top features to keep: "))
        selected_features = [feature for feature, h_stat in ranked_features[:top_n]]
        x = x[:, selected_features]
        print(f"\nSelected features: {selected_features}")
    else:
        if optionKruskal == "n":
            print("Kruskal-Wallis test wont be applied")
        else:
            print("Invalid option. By default, Kruskal-Wallis test won't be performed.")
    
    return x

def choosePCA(x):
    optionPca = input("Do you want to apply PCA? (y/n): ")
    if optionPca == "y":
        pca, x = apply_pca(x, n_components=None)  # Calcula todos os componentes
        print("PCA applied")
        
        test_option = input("Choose a test to decide the number of components (kaiser/scree): ")
        if test_option == "kaiser":
            n_components = kaiser_test(pca)
            x = apply_pca(x, n_components=n_components)[1]
        elif test_option == "scree":
            scree_test(pca)
            n_components = int(input("Based on the Scree Plot, enter the number of components to keep: "))
            x = apply_pca(x, n_components=n_components)[1]
        else:
            print("Invalid test option. No further PCA reduction will be applied.")
    else:
        print("PCA not applied")
    
    return x

def cross_validate_model(X, T, classifier, cv=5):
    T = T.astype(int)
    
    accuracies, sensitivities, specificities, f1_scores = [], [], [], []
    
    skf = StratifiedKFold(n_splits=cv)
    
    for train_index, test_index in skf.split(X, T):
        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        
        classifier.fit(X_train, T_train)
        
        predictions = classifier.predict(X_test)
        
        accuracy, sensitivity, specificity, f1 = calculate_metrics(T_test, predictions)
        
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1)
    
    mean_accuracy = np.mean(accuracies)
    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)
    mean_f1 = np.mean(f1_scores)
    
    return mean_accuracy, mean_sensitivity, mean_specificity, mean_f1

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

class MahalanobisMinimumDistanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.mean_C1 = None
        self.mean_C2 = None
        self.cov_inv_C1 = None
        self.cov_inv_C2 = None

    def fit(self, X, y):
        # Treinar o classificador
        class_C1 = X[y == 0]
        class_C2 = X[y == 1]
        
        # Calcular as médias das classes
        self.mean_C1 = np.mean(class_C1, axis=0)
        self.mean_C2 = np.mean(class_C2, axis=0)
        
        # Calcular as matrizes de covariância e suas inversas
        cov_C1 = np.cov(class_C1, rowvar=False)
        cov_C2 = np.cov(class_C2, rowvar=False)
        
        # Adicionar uma pequena constante à diagonal para evitar singularidade
        epsilon = 1e-6
        self.cov_inv_C1 = np.linalg.inv(cov_C1 + epsilon * np.eye(cov_C1.shape[0]))
        self.cov_inv_C2 = np.linalg.inv(cov_C2 + epsilon * np.eye(cov_C2.shape[0]))
        
        return self

    def predict(self, X):
        # Fazer previsões
        predictions = []
        for point in X:
            distance_C1 = mahalanobis(point, self.mean_C1, self.cov_inv_C1)
            distance_C2 = mahalanobis(point, self.mean_C2, self.cov_inv_C2)
            if distance_C1 < distance_C2:
                predictions.append(0)  # Pertence à classe C1
            else:
                predictions.append(1)  # Pertence à classe C2
        return np.array(predictions)

class MinimumDistanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.mean_C1 = None
        self.mean_C2 = None

    def fit(self, X, y):
        # Treinar o classificador
        class_C1 = X[y == 0]
        class_C2 = X[y == 1]
        self.mean_C1 = np.mean(class_C1, axis=0)
        self.mean_C2 = np.mean(class_C2, axis=0)
        return self

    def predict(self, X):
        # Fazer previsões
        predictions = []
        for point in X:
            distance_C1 = np.linalg.norm(point - self.mean_C1)
            distance_C2 = np.linalg.norm(point - self.mean_C2)
            if distance_C1 < distance_C2:
                predictions.append(0)  # Pertence à classe C1
            else:
                predictions.append(1)  # Pertence à classe C2
        return np.array(predictions)

def cross_validate_fld(X, T, cv=5):
    T = T.astype(int)
    
    unique_labels = np.unique(T)
    if len(unique_labels) != 2:
        raise ValueError("O FLD requer exatamente duas classes. Verifique os rótulos.")
    
    accuracies, sensitivities, specificities, f1_scores = [], [], [], []
    
    skf = StratifiedKFold(n_splits=cv)
    
    for train_index, test_index in skf.split(X, T):
        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        
        lda = LDA()
        lda.fit(X_train, T_train)
        
        predictions = lda.predict(X_test)
        
        accuracy, sensitivity, specificity, f1 = calculate_metrics(T_test, predictions)
        
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1)
    
    mean_accuracy = np.mean(accuracies)
    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)
    mean_f1 = np.mean(f1_scores)
    
    return mean_accuracy, mean_sensitivity, mean_specificity, mean_f1

def apply_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    #print("\nPCA Results:")
    #print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    #print("Principal Components (Eigen Vectors):\n", pca.components_)
    
    return pca, X_pca

def kaiser_test(pca):
    eigenvalues = pca.explained_variance_
    n_components = np.sum(eigenvalues > 1)
    print(f"Kaiser Test: {n_components} should be retained.")
    return n_components

def scree_test(pca):
    eigenvalues = pca.explained_variance_
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b', label='Eigenvalues')
    
    plt.axhline(y=1, color='r', linestyle='--', label='Kaiser limit (eigenvalue = 1)')
    
    plt.title("Scree Plot")
    plt.xlabel("Number of Components")
    plt.ylabel("Eigenvalues")
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
    print("Scree Test: Inspect the graph to identify the optimal number of components.")

def normalize_data(X, method='min-max'):
    if X.dtype != float:
        print("Converting X to float.")
        X = X.astype(float)
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("X has NaN or inf values. NaN will be substituted by the mean of the column and inf for big values.")
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        X = np.nan_to_num(X, posinf=1e10, neginf=-1e10)
    
    if method == 'min-max':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_normalized = (X - X_min) / (X_max - X_min)
    elif method == 'z-score':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    else:
        raise ValueError("Invalid normalization method")
    
    return X_normalized

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, sensitivity, specificity, f1

df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
D = df.values
X = D[:, 0:54]
T = D[:,55]

col_remove = [0, 1, 3, 6, 29]
X = np.delete(X, col_remove, axis=1)

option = -1
while option != "0":
    MainMenu()
    option = input("Choose an option: ")
    MainMenuOptions(option, X, T)
