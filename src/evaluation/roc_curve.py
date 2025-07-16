import argparse
import os
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

# Usar backend no interactivo para evitar errores al generar gráficos en entornos sin GUI
matplotlib.use('Agg')  # Evita problemas al ejecutar el script como proceso batch


def roc_curve_generate():
    """
    Genera la curva ROC de un clasificador MLP entrenado sobre un conjunto de datos.
    El gráfico se guarda como una imagen PNG y también se muestra visualmente al finalizar.

    Flujo del proceso:
    1. Cargar los datos de prueba y entrenamiento desde archivos .pkl.
    2. Configurar y entrenar un MLPClassifier con los parámetros óptimos.
    3. Generar las predicciones y calcular la curva ROC.
    4. Graficar y guardar la curva ROC como imagen.
    """

    data_dir = '../../data/preprocessed'
    results_dir = '../../reports/figures'

    try:
        X_test = joblib.load(os.path.join(data_dir, 'x_test.pkl'))
        y_test = joblib.load(os.path.join(data_dir, 'y_test.pkl'))
        X_train = joblib.load(os.path.join(data_dir, 'x_train.pkl'))
        y_train = joblib.load(os.path.join(data_dir, 'y_train.pkl'))
    except FileNotFoundError as e:
        print(f"Error cargando archivos: {e}")
        return

    best_params = {
        'hidden_layer_sizes': [64, 32],
        'alpha': 0.0001,
        'learning_rate_init': 0.001,
        'max_iter': 300,
        'early_stopping': True,
        'random_state': 42
    }

    model = MLPClassifier(**best_params)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curva ROC: Predicción de Complicaciones en Cirugías de Cadera', fontsize=14)


    idx = np.argmax(tpr - fpr)
    plt.scatter(fpr[idx], tpr[idx], marker='o', color='red', s=100,
                label=f'Umbral óptimo: {thresholds[idx]:.2f}')

    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)


    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'roc_curve_mlp.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Curva ROC guardada en: {output_path}")


    plt.close()  # Cierra la figura actual
    matplotlib.use('TkAgg')  # Cambia a backend compatible
    plt.imshow(plt.imread(output_path))
    plt.axis('off')
    plt.title('Curva ROC Generada')
    plt.show()


if __name__ == '__main__':
    roc_curve_generate()