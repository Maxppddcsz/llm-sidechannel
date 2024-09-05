import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
import sys


def read_and_label_data(optimized_file, non_optimized_file, label_opt, label_non_opt):
    print(f"Reading and labeling data from {optimized_file} and {non_optimized_file}")
    optimized_data = np.loadtxt(optimized_file)
    non_optimized_data = np.loadtxt(non_optimized_file)

    optimized_df = pd.DataFrame(optimized_data, columns=['response_time'])
    non_optimized_df = pd.DataFrame(non_optimized_data, columns=['response_time'])

    optimized_df['label'] = label_opt
    non_optimized_df['label'] = label_non_opt

    return optimized_df, non_optimized_df

def optimize_xgboost(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    
    clf = xgb.XGBClassifier(
        **params,
        tree_method='gpu_hist',
        eval_metric='logloss',
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        clf.fit(X_train_cv, y_train_cv)
        y_prob = clf.predict_proba(X_val_cv)[:, 1]
        fpr, tpr, _ = roc_curve(y_val_cv, y_prob)
        roc_auc = auc(fpr, tpr)
        scores.append(roc_auc)

    return np.mean(scores)

def find_threshold_for_tpr(fpr, tpr, thresholds, target_tpr=0.7):
    tpr_diff = np.abs(tpr - target_tpr)
    best_index = np.argmin(tpr_diff)
    
    if best_index >= len(thresholds):
        best_index = len(thresholds) - 1
    
    optimal_threshold = thresholds[best_index]
    optimal_fpr = fpr[best_index]
    
    return optimal_threshold, optimal_fpr

def train_and_evaluate(X_train, X_test, y_train, y_test, prefix_length):
    print(f"Starting training and evaluation for prefix length {prefix_length}")

    print("Preprocessing data")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Applying SMOTE to handle imbalanced data")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("Applying feature selection")
    selector = SelectKBest(f_classif, k=20) # NO MEANING
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    print("Starting Bayesian Optimization")
    optimizer = BayesianOptimization(
        f=optimize_xgboost,
        pbounds={
            'n_estimators': (50, 500),
            'max_depth': (3, 15),
            'learning_rate': (0.001, 0.3),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        },
        random_state=42
    )

    optimizer.maximize(init_points=10, n_iter=30)

    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    print(f"Training individual classifiers with best parameters: {best_params}")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        tree_method='gpu_hist',
        eval_metric='logloss',
        random_state=42
    )

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('gb', gb_clf)],
        voting='soft',
        weights=[2, 1, 1]
    )

    voting_clf.fit(X_train, y_train)
    y_prob = voting_clf.predict_proba(X_test)[:, 1]

    print("Calculating ROC curve and AUC")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    optimal_threshold, optimal_fpr = find_threshold_for_tpr(fpr, tpr, thresholds, target_tpr=0.55)

    y_pred_new_threshold = (y_prob >= optimal_threshold).astype(int)

    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(y_test, y_pred_new_threshold)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_new_threshold).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    # 保存模型、scaler和selector
    model_filename = f'data/voting_model_prefix_{prefix_length}.pkl'
    scaler_filename = f'data/scaler_prefix_{prefix_length}.pkl'
    selector_filename = f'data/selector_prefix_{prefix_length}.pkl'
    threshold_filename = f'data/best_threshold_prefix_{prefix_length}.pkl'

    print(f"Saving model to {model_filename}")
    joblib.dump(voting_clf, model_filename)

    print(f"Saving scaler to {scaler_filename}")
    joblib.dump(scaler, scaler_filename)

    print(f"Saving selector to {selector_filename}")
    joblib.dump(selector, selector_filename)

    print(f"Saving best threshold to {threshold_filename}")
    joblib.dump(optimal_threshold, threshold_filename)

    # 保存测试结果
    results_filename = f'data/results_prefix_{prefix_length}.txt'
    print(f"Saving results to {results_filename}")
    with open(results_filename, 'w') as f:
        f.write(f'ROC AUC: {roc_auc}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'False Positive Rate: {false_positive_rate}\n')
        f.write(f'False Negative Rate: {false_negative_rate}\n')
        f.write(f'Optimal Threshold: {optimal_threshold}\n')
        f.write(f'Optimal FPR: {optimal_fpr}\n')

    # 绘制ROC曲线并保存
    print("Plotting ROC curve")
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(optimal_fpr, 0.55, color='red', label=f'Threshold at TPR=0.55, FPR={optimal_fpr}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_filename = f'model/roc_curve_prefix_{prefix_length}.pdf'
    plt.savefig(roc_filename)
    plt.close()

    return fpr, tpr, roc_auc, accuracy, false_positive_rate, false_negative_rate, optimal_threshold

if __name__ == "__main__":
    for prefix_length in [3,4,5,6,7]: 
        optimized_file = f"data/prefix_{prefix_length+1}.txt"
        non_optimized_file = f'data/prefix_{prefix_length}.txt'

        optimized_df, non_optimized_df = read_and_label_data(optimized_file, non_optimized_file, label_opt=1, label_non_opt=0)

        data = pd.concat([optimized_df, non_optimized_df], ignore_index=True)
        X = data[['response_time']].values
        y = data['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        fpr, tpr, roc_auc, accuracy, false_positive_rate, false_negative_rate, optimal_threshold = train_and_evaluate(X_train, X_test, y_train, y_test, prefix_length)
        print(f"Evaluation complete for prefix length {prefix_length}")
