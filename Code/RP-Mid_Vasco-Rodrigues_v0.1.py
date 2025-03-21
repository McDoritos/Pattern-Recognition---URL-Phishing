import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import kruskal

def MainMenu():
    print("----------Main Menu----------")
    print("1 - Minimum Distance Classifier\n"
    "2 - Fisher LDA\n"
    "0 - Exit")

def MainMenuOptions(option, X, T):
    if option == "1":
        x_mdc = X
        
        x_mdc = chooseNormalize(x_mdc)

        x_mdc = chooseKruskal(x_mdc)
        
        x_mdc = choosePCA(x_mdc)
        
        print("Calculating predictions...")
        predictions = minimum_distance_classifier(x_mdc, T) # usar ou X ou X_normalized ou X_pca

        evaluation(predictions, T)
    elif option == "2":
        x_fisher = X

        x_fisher = chooseNormalize(x_fisher)

        x_fisher = chooseKruskal(x_fisher)
        
        x_fisher = choosePCA(x_fisher)

        print("Calculating predictions...")
        predictions = fisher_discriminant(x_fisher, T)

        evaluation(predictions, T)
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


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def minimum_distance_classifier(X, T):
    print(X)
    class_C1 = X[T == 0]
    class_C2 = X[T == 1]
    
    mean_C1 = np.mean(class_C1, axis=0)
    mean_C2 = np.mean(class_C2, axis=0)
    
    predictions = []
    for point in X:
        distance_C1 = euclidean_distance(point, mean_C1)
        distance_C2 = euclidean_distance(point, mean_C2)
        if distance_C1 < distance_C2:
            predictions.append(0)  # Pertence à classe C1
        else:
            predictions.append(1)  # Pertence à classe C2
    
    return np.array(predictions)
    #errors_C1 = np.sum((np.array(predictions) == 1) & (T == 0))
    #errors_C2 = np.sum((np.array(predictions) == 0) & (T == 1))
    #
    #total_C1 = np.sum(T == 0)
    #total_C2 = np.sum(T == 1)
    #
    #error_rate_C1 = errors_C1 / total_C1
    #error_rate_C2 = errors_C2 / total_C2
    #
    #return error_rate_C1, error_rate_C2

def fisher_discriminant(X, T):

    T = T.astype(int)
    
    unique_labels = np.unique(T)
    if len(unique_labels) != 2:
        raise ValueError("O LDA requer exatamente duas classes. Verifique os rótulos.")
    
    lda = LDA()
    lda.fit(X, T)
    
    predictions = lda.predict(X)
    
    return predictions

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

def evaluation(predictions, T):
    TP = np.sum((predictions == 1) & (T == 1))
    TN = np.sum((predictions == 0) & (T == 0))
    FP = np.sum((predictions == 1) & (T == 0))
    FN = np.sum((predictions == 0) & (T == 1))
    
    sensitivity = TP / (TP + FN)
    
    specificity = TN / (TN + FP)
    
    precision = TP / (TP + FP)
    
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print("\nEvaluation Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-measure (F1-score): {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

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