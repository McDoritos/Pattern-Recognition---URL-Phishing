import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import kruskal
from sklearn.model_selection import cross_val_score, StratifiedKFold,RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from sklearn import mixture, svm
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

def MainMenu():
    print("----------Main Menu----------")
    print("1 - Minimum Distance Classifier - Euclidean\n"
    "2 - Minimum Distance Classifier - Mahalanobis\n"
    "3 - Fisher LDA\n"
    "4 - Bayes Classifier\n"
    "5 - KNN Classifier\n"
    "6 - SVM Classifier\n"
    "7 - Validate Best classifier\n"
    "8 - Grid Search C (SVM)\n"
    "9 - Choose best K for KNN\n"
    "10 - Visualize kruskal wallis\n"
    "11 - Visualize correlation matrix\n"
    "0 - Exit")

def MainMenuOptions(option, X, T):
    if option == "1":
        x_mdce = X.copy()

        x_mdce = chooseNormalize(x_mdce)

        x_mdce = chooseFeatureReduction(x_mdce)

        x_mdce = choosePCA(x_mdce)

        print("Calculating predictions with cross-validation...")
        classifier = MinimumDistanceClassifier()
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, _, fpr, tpr = cross_validate_model(x_mdce, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")

        plot_ROC(fpr, tpr)
    elif option =="2":
        x_mdcm = X.copy()

        x_mdcm = chooseNormalize(x_mdcm)

        x_mdcm = chooseFeatureReduction(x_mdcm)

        x_mdcm = choosePCA(x_mdcm)

        print("Calculating predictions with cross-validation...")
        classifier = MahalanobisMinimumDistanceClassifier()
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, _, fpr, tpr = cross_validate_model(x_mdcm, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")

        plot_ROC(fpr, tpr)
    elif option == "3":
        x_fisher = X.copy()

        x_fisher = chooseNormalize(x_fisher)

        x_fisher = chooseFeatureReduction(x_fisher)
        
        x_fisher = choosePCA(x_fisher)

        print("Calculating predictions...")
        classifier = LDA()
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, _, fpr, tpr = cross_validate_model(x_fisher, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")

        plot_ROC(fpr, tpr)
    elif option == "4":
        x_bayes = X.copy()

        x_bayes = chooseNormalize(x_bayes,force=True)

        x_bayes = chooseFeatureReduction(x_bayes)

        x_bayes = choosePCA(x_bayes)

        print("Calculating predictions...")
        classifier = BayesClassifier()
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, _, fpr, tpr = cross_validate_model(x_bayes, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")

        plot_ROC(fpr, tpr)

    elif option == "5":
        x_knn = X.copy()

        x_knn = chooseNormalize(x_knn,force=True)

        x_knn = chooseFeatureReduction(x_knn)

        x_knn = choosePCA(x_knn)

        print("Calculating predictions...")
        classifier = KNeighborsClassifier(n_neighbors=3)
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, _, fpr, tpr = cross_validate_model(x_knn, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")

        plot_ROC(fpr, tpr)

    elif option == "6":
        x_svm = X.copy()

        x_svm = chooseNormalize(x_svm,force=True)

        x_svm = chooseFeatureReduction(x_svm)

        x_svm = choosePCA(x_svm)

        print("Calculating predictions...")
        classifier = chooseKernelSVM()
        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, _, fpr, tpr = cross_validate_model(x_svm, T, classifier, cv=5)

        print("\nMetrics\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")

        plot_ROC(fpr, tpr)

    elif option == "7":

        X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3, random_state=42)

        x_best_model = X_train.copy()

        x_best_model = chooseNormalize(x_best_model,force=True)

        print("Calculating predictions...")
        classifier = chooseKernelSVM(force_linear=True)

        mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, classifier, fpr, tpr = cross_validate_model(x_best_model, T_train, classifier, cv=5)
        print("\nMetrics on trainig data\n")
        print(f"Mean accuracy (cross-validation): {mean_accuracy:.4f}")
        print(f"Sensitivity (Recall): {mean_sensitivity:.4f}")
        print(f"Specificity: {mean_specificity:.4f}")
        print(f"F-measure (F1-score): {mean_f1:.4f}")
        
        print("Doing validation...")
        
        X_test = chooseNormalize(X_test,force=True)

        predictions = classifier.predict(X_test)

        T_test = T_test.astype(int)

        accuracy, sensitivity, specificity, f1 = calculate_metrics(T_test, predictions)
        print("\nMetrics on validating data\n")
        print(f"Mean accuracy (cross-validation): {accuracy:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F-measure (F1-score): {f1:.4f}")

    elif option == "8":

        optionSVM = input("Do you want to test for linear or rbf? (0/1): ")

        n_samples = 5000
        random_indices = np.random.choice(len(X), n_samples, replace=False)
        X_reduced = X[random_indices]
        T_reduced = T[random_indices].astype(int)

        x_svm = chooseNormalize(X_reduced, force=True)
        x_svm = chooseFeatureReduction(x_svm, T_reduced)
        x_svm = choosePCA(x_svm)

        C_range = [2**i for i in range(-5, 13)]
        n_repeats = 10

        if optionSVM == "0":
            print("Grid Search for linear SVM...")
            param_grid = {'C': C_range}
            mean_scores = np.zeros(len(C_range))
            classifier = svm.SVC(kernel='linear')
        else:
            print("Grid Search for RBF SVM...")
            gamma_range = [2**i for i in range(-30, 6)]
            param_grid = {'C': C_range, 'gamma': gamma_range}
            mean_scores = np.zeros((len(gamma_range), len(C_range)))
            classifier = svm.SVC(kernel='rbf')

        for _ in tqdm(range(n_repeats), desc="Grid Search Progress"):
            X_train, X_test, T_train, T_test = train_test_split(x_svm, T_reduced, test_size=0.5, random_state=None)
            
            grid = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
            grid.fit(X_train, T_train)
            
            if optionSVM == "0":
                for j, C in enumerate(C_range):
                    mean_scores[j] += 1 - grid.cv_results_['mean_test_score'][grid.cv_results_['params'].index({'C': C})]
            else:
                for i, gamma in enumerate(gamma_range):
                    for j, C in enumerate(C_range):
                        mean_scores[i, j] += 1 - grid.cv_results_['mean_test_score'][grid.cv_results_['params'].index({'C': C, 'gamma': gamma})]

        mean_scores /= n_repeats

        if optionSVM == "0":
            # Resultados para SVM linear
            best_idx = np.argmin(mean_scores)
            best_C = C_range[best_idx]
            best_error = mean_scores[best_idx]
            
            print(f"\n--- Best Parameters (Linear SVM) ---")
            print(f"Best C: {best_C:.6f}")
            print(f"Best error: {best_error:.6f}")
            
            # Plot para linear (apenas C)
            plt.figure(figsize=(10, 6))
            plt.plot(np.log2(C_range), mean_scores, 'b-o')
            plt.scatter(np.log2(best_C), best_error, color='red', s=100, label=f'Best C=2^{np.log2(best_C):.0f}')
            plt.xlabel('log2(C)')
            plt.ylabel('Error Rate')
            plt.title('Linear SVM Performance')
            plt.legend()
            plt.grid(True)
            plt.show()
    
        else:
            # Resultados para SVM RBF
            best_idx = np.unravel_index(np.argmin(mean_scores), mean_scores.shape)
            best_C = C_range[best_idx[1]]
            best_gamma = gamma_range[best_idx[0]]
            best_error = mean_scores[best_idx]
            
            print(f"\n--- Best Parameters (RBF SVM) ---")
            print(f"Best C: {best_C:.6f}")
            print(f"Best gamma: {best_gamma:.6f}")
            print(f"Best error: {best_error:.6f}")
            
            # Plot 3D para RBF
            C_log = np.log2(C_range)
            gamma_log = np.log2(gamma_range)
            C_grid, gamma_grid = np.meshgrid(C_log, gamma_log)
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(C_grid, gamma_grid, mean_scores, cmap='viridis', alpha=0.8)
            ax.scatter(np.log2(best_C), np.log2(best_gamma), best_error, color='red', s=100, 
                    label=f'Best (C=2^{np.log2(best_C):.0f}, γ=2^{np.log2(best_gamma):.0f})')
            ax.set_xlabel('log2(C)')
            ax.set_ylabel('log2(gamma)')
            ax.set_zlabel('Error Rate')
            ax.legend()
            plt.title('RBF SVM Performance')
            plt.tight_layout()
            plt.show()

    elif option == "9":
        x_knn = X.copy()

        x_knn = chooseNormalize(x_knn,force=True)

        x_knn = chooseFeatureReduction(x_knn)

        x_knn = choosePCA(x_knn)

        Xtrain, Xtest, Ttrain, Ttest = train_test_split(x_knn, T, test_size=0.5, random_state=42)

        Ttrain = Ttrain.astype(int)
        # Cross-validation to find best k, using the error
        k_values = range(1, 40)
        mean_errors = []
        std_errors = []

        for k in tqdm(k_values, desc="Testando valores de k"):
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, Xtrain, Ttrain, cv=5, scoring='accuracy', n_jobs=-1)
            errors = 1 - scores
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))


        plt.figure(figsize=(12, 6))
        plt.errorbar(k_values, mean_errors, yerr=std_errors, fmt='-o', capsize=5)
        plt.title('Cross-Validation Testing Error vs. Number of Neighbors')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Mean Testing Error (± 1 SD)')
        plt.grid(True)
        plt.show()

        best_k = k_values[np.argmin(mean_errors)]
        print(f"Optimal number of neighbors (based on error): k = {best_k}")
    elif option == "10":
        x = normalize_data(X)

        h_statistics = {}
        p_values = {}
        for column in range(x.shape[1]):
            grouped_data = [x[T == group, column] for group in np.unique(T)]
            h_stat, p_value = kruskal(*grouped_data)
            h_statistics[column] = h_stat
            p_values[column] = p_value

        plot_kruskal_results(x, T, h_statistics)
    elif option == "11":
        n_features = X.shape[1]
        x_df = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(n_features)])

        corr_matrix = x_df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Matriz de Correlação")
        plt.show()
    elif option == "0":
        print("Exiting...")
    else:
        print("Invalid Option")
    

def chooseNormalize(x, force=False):
    if not force:
        optionNormalized = input("Do you want the data to be normalized? (y/n): ")
        if optionNormalized == "y":
            x = normalize_data(x, method='z-score')
            print("Data normalized")
        elif optionNormalized == "n":
            print("Data not normalized")
        else:
            print("Invalid option. By default, data won't be normalized.")
    else:
        x = normalize_data(x, method='z-score')
        print("Data normalized")

    return x

def chooseFeatureReduction(x):
    option = input("Do you want to use for feature reduction?(0/1)")
    if option == "1":
        option_feature_reduction = input("Kruskal wallis or ROC? (0/1)")
        if option_feature_reduction == "0":
            return chooseKruskal(x)
        elif option_feature_reduction == "1":
            return chooseROCFeatureReduction(x)
        else:
            print("Invalid option, not applying feature reduction")
            return x
    elif option == "0":
        print("Not applying feature reduction")
        return x
    else:
        print("Invalid option, not applying feature reduction")
        return x

def chooseKruskal(x,t=None):
    if t is None:
        t=T.copy()
    
    print("Applying Kruskal-Wallis test to select features...")
    h_statistics = {}
    for column in range(x.shape[1]):  # Itera sobre as colunas de x
        grouped_data = [x[t == group, column] for group in np.unique(T)]
        h_stat, p_value = kruskal(*grouped_data)
        h_statistics[column] = h_stat
    
    ranked_features = sorted(h_statistics.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeatures ranked by Kruskal-Wallis H statistic:")
    counter = 0
    for feature, h_stat in ranked_features:
        counter+=1
        print(f"{counter} - Feature {feature}: H = {h_stat:.4f}")
    
    top_n = int(input("Enter the number of top features to keep: "))
    selected_features = [feature for feature, h_stat in ranked_features[:top_n]]
    x = x[:, selected_features]
    print(f"\nSelected features: {selected_features}")
    
    return x

def chooseROCFeatureReduction(x,t=None):
    if t is None:
            t=T.copy()

    t = t.astype(int)

    print("\nApplying ROC AUC feature selection...")
    auc_scores = {}
    
    for column in range(x.shape[1]):
        auc = roc_auc_score(t, x[:, column])
        auc_scores[column] = auc
    
    ranked_features = sorted(auc_scores.items(), key=lambda item: item[1], reverse=True)
    
    print("\nFeatures ranked by ROC AUC score:")
    counter = 0
    for feature, auc in ranked_features:
        counter+=1
        print(f"{counter} - Feature {feature}: AUC = {auc:.4f}")
    
    top_n = int(input("\nEnter the number of top features to keep: "))
    selected_features = [feature for feature, auc in ranked_features[:top_n]]
    x_selected = x[:, selected_features]
    
    print(f"\nSelected features: {selected_features}")
    return x_selected

def plot_kruskal_results(x, T, h_statistics):
    ranked_features = sorted(h_statistics.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeatures ranked by Kruskal-Wallis H statistic:")
    for feature, h_stat in ranked_features:
        print(f"Feature {feature}: H = {h_stat:.4f}")
    
    n_features = x.shape[1]
    
    n_cols = 6 
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(20, 3 * n_rows)) 
    
    for i, (feature, h_stat) in enumerate(ranked_features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.violinplot(x=T, y=x[:, feature], legend=False) 
        plt.title(f"Feature {feature}")
        plt.xlabel("")
        plt.ylabel("")
    plt.tight_layout()
    plt.show()


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

def chooseKernelSVM(force_linear=False):
    if force_linear:
        print("Using best model - svm with linear kernel...")
        classifier = svm.SVC(kernel='linear', C=16)
    else:
        optionSvm = input("Do you want a linear or non-linear kernel? (1/2): ")
        if optionSvm == "1":
            print("Using svm with linear kernel...")
            classifier = svm.SVC(kernel='linear', C=16) # DO GRID SEARCH FOR C
        else:
            print("Using svm with Non-linear kernel...")
            classifier = svm.SVC(kernel='rbf', C=4096, gamma=0.000977) 
    
    return classifier
        

def cross_validate_model(X, T, classifier, cv=5):
    T = T.astype(int)
    
    accuracies, sensitivities, specificities, f1_scores = [], [], [], []
    all_fpr, all_tpr = [], []

    skf = StratifiedKFold(n_splits=cv)
    
    fold_iter = tqdm(skf.split(X, T), 
                    total=cv,
                    desc=f"Cross-validating {classifier.__class__.__name__}",
                    unit="fold")
    
    for train_index, test_index in fold_iter:
        X_train, X_test = X[train_index], X[test_index]
        T_train, T_test = T[train_index], T[test_index]
        
        classifier.fit(X_train, T_train)

        if hasattr(classifier, "decision_function"):
            y_scores = classifier.decision_function(X_test)
        elif hasattr(classifier, "predict_proba"):
            y_scores = classifier.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(T_test, y_scores)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        predictions = classifier.predict(X_test)
        accuracy, sensitivity, specificity, f1 = calculate_metrics(T_test, predictions)
        
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1)
        
        # Atualiza a descrição da barra de progresso com métricas atuais
        fold_iter.set_postfix({
            'Acc': f"{accuracy:.3f}",
            'Sens': f"{sensitivity:.3f}",
            'Spec': f"{specificity:.3f}",
            'F1': f"{f1:.3f}"
        })
    
    mean_accuracy = np.mean(accuracies)
    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)
    mean_f1 = np.mean(f1_scores)
    
    return mean_accuracy, mean_sensitivity, mean_specificity, mean_f1, classifier, all_fpr, all_tpr

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

class MahalanobisMinimumDistanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.mean_C1 = None
        self.mean_C2 = None
        self.cov_inv_C1 = None
        self.cov_inv_C2 = None

    def fit(self, X, y):
        class_C1 = X[y == 0]
        class_C2 = X[y == 1]
        
        self.mean_C1 = np.mean(class_C1, axis=0)
        self.mean_C2 = np.mean(class_C2, axis=0)
        
        cov_C1 = np.cov(class_C1, rowvar=False)
        cov_C2 = np.cov(class_C2, rowvar=False)
        
        epsilon = 1e-6  # Evita matriz singular
        self.cov_inv_C1 = np.linalg.inv(cov_C1 + epsilon * np.eye(cov_C1.shape[0]))
        self.cov_inv_C2 = np.linalg.inv(cov_C2 + epsilon * np.eye(cov_C2.shape[0]))
        
        return self

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, 0, 1)

    def decision_function(self, X):

        distances_C1 = np.array([mahalanobis(x, self.mean_C1, self.cov_inv_C1) for x in X])
        distances_C2 = np.array([mahalanobis(x, self.mean_C2, self.cov_inv_C2) for x in X])
        
        return distances_C1 - distances_C2

class MinimumDistanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.mean_C1 = None
        self.mean_C2 = None

    def fit(self, X, y):
        class_C1 = X[y == 0]
        class_C2 = X[y == 1]
        self.mean_C1 = np.mean(class_C1, axis=0)
        self.mean_C2 = np.mean(class_C2, axis=0)
        return self

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, 0, 1)

    def decision_function(self, X):
        distances_C1 = np.linalg.norm(X - self.mean_C1, axis=1)
        distances_C2 = np.linalg.norm(X - self.mean_C2, axis=1)
        return distances_C1 - distances_C2

class BayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.mean1 = None
        self.mean2 = None
        self.cov1 = None
        self.cov2 = None
        self.Pw1 = None
        self.Pw2 = None
        self.epsilon = 1e-10

    def pdfGauss(self, x, mean, cov):
        covInv = np.linalg.inv(cov)
        dim = cov.shape[0]
        val = np.array([])
        for i in range(x.shape[0]):
            dist = ((np.array([x[i,:] - mean])) @ covInv @ (np.array([x[i,:] - mean])).T.squeeze())
            val = np.append(val, np.exp(-0.5*dist) / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**0.5))
        return np.array([val]).T

    def fit(self, x, y):
        ix1 = np.where(y==0)[0]
        ix2 = np.where(y==1)[0]

        n_classe1 = ix1.shape[0]
        n_classe2 = ix2.shape[0]
        total = n_classe1 + n_classe2

        self.Pw1 = n_classe1 / total
        self.Pw2 = n_classe2 / total

        clf1 = mixture.GaussianMixture(n_components=1)
        clf2 = mixture.GaussianMixture(n_components=1)
        mod1 = clf1.fit(x[ix1,:])
        mod2 = clf2.fit(x[ix2,:])
        
        self.mean1 = mod1.means_.squeeze()
        self.mean2 = mod2.means_.squeeze()
        self.cov1 = mod1.covariances_[0]
        self.cov2 = mod2.covariances_[0]
        
        return self

    def predict(self, xtest):
        Pw1X = self.pdfGauss(xtest, self.mean1, self.cov1) * self.Pw1
        Pw2X = self.pdfGauss(xtest, self.mean2, self.cov2) * self.Pw2
        
        predictions = ((-np.sign(Pw1X - Pw2X)) * 0.5 + 1.5).squeeze()
        return np.where(predictions > 1, 1, 0).astype(int)
    
    def predict_proba(self, xtest):
        """Only used for ROC Curve"""
        Pw1X = self.pdfGauss(xtest, self.mean1, self.cov1) * self.Pw1
        Pw2X = self.pdfGauss(xtest, self.mean2, self.cov2) * self.Pw2
        
        # Adds epsilon to avoid zero division
        total = Pw1X + Pw2X + 2*self.epsilon
        proba1 = (Pw1X + self.epsilon) / total
        proba2 = (Pw2X + self.epsilon) / total
        
        return np.hstack([proba1, proba2])

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

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

def plot_ROC(all_fpr, all_tpr):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    for fpr, tpr in zip(all_fpr, all_tpr):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        tprs.append(interp_tpr)
    
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc = np.trapz(mean_tpr, mean_fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', 
             label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)
    
    plt.fill_between(mean_fpr, 
                     mean_tpr - std_tpr, 
                     mean_tpr + std_tpr, 
                     color='grey', alpha=0.2,
                     label='±1 std. dev.')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Mean ROC Curve (Cross-Validation)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
D = df.values
X = D[:, 0:54]
T = D[:,55]

col_remove = [0, 1, 3, 6, 16, 18, 20, 25, 28, 29, 31, 32, 33, 34, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48]
X = np.delete(X, col_remove, axis=1)

option = -1
while option != "0":
    MainMenu()
    option = input("Choose an option: ")
    MainMenuOptions(option, X, T)
