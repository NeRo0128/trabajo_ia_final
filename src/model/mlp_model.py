import os
import joblib
import pandas as pd
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)


def train_mlp(data_dir, results_dir, results_file, model_params):
    os.makedirs(results_dir, exist_ok=True)
    x_train = joblib.load(os.path.join(data_dir, 'X_train.pkl'))
    x_test = joblib.load(os.path.join(data_dir, 'X_test.pkl'))
    y_train = joblib.load(os.path.join(data_dir, 'y_train.pkl'))
    y_test = joblib.load(os.path.join(data_dir, 'y_test.pkl'))

    results = []

    for size in model_params['hidden_layer_sizes']:
        for a in model_params['alpha']:
            for lr in model_params['learning_rate_init']:
                for mi in model_params['max_iter']:
                    model = MLPClassifier(
                        hidden_layer_sizes=size,
                        alpha=a,
                        learning_rate_init=lr,
                        max_iter=mi,
                        early_stopping=model_params['early_stopping'],
                        random_state=model_params['random_state']
                    )
                    start_time = time.time()
                    model.fit(x_train, y_train)
                    train_time = time.time() - start_time

                    preds = model.predict(x_test)
                    proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, zero_division=0)
                    rec = recall_score(y_test, preds, zero_division=0)
                    f1 = f1_score(y_test, preds, zero_division=0)
                    roc = roc_auc_score(y_test, proba) if proba is not None else None

                    cm = confusion_matrix(y_test, preds)
                    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None, None, None, None)

                    results.append({
                        'hidden_layer_sizes': str(size),
                        'alpha': a,
                        'learning_rate_init': lr,
                        'max_iter': mi,
                        'early_stopping': model_params['early_stopping'],
                        'train_time': train_time,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1,
                        'roc_auc': roc,
                        'true_positive': tp,
                        'false_positive': fp,
                        'false_negative': fn,
                        'true_negative': tn
                    })

    # Guardar resultados en un archivo CSV
    df_mlp = pd.DataFrame(results)
    output_path:  str | bytes = os.path.join(results_dir, results_file)
    df_mlp.to_csv(output_path, index=False)
    print(f"MLP: resultados guardados en {output_path}")

