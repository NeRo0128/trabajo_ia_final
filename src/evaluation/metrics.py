import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

def save_results(results_dir, output_file, y_test_path, X_test_path):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_mlp = pd.read_csv(os.path.join(results_dir, 'mlp_results.csv'))
    df_mlp.to_excel(output_file, sheet_name='MLP', index=False)

    y_test = joblib.load(y_test_path)
    best_model = df_mlp.loc[df_mlp['accuracy'].idxmax()]
    model_path = os.path.join(results_dir, f"mlp_model_{best_model['accuracy']:.4f}.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        X_test = joblib.load(X_test_path)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión - Mejor Modelo MLP')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        plt.savefig(os.path.join('reports/img', 'confusion_matrix.png'))
        plt.close()
    print(f"Resultados guardados en {output_file}")

if __name__ == "__main__":
    import sys
    save_results(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])