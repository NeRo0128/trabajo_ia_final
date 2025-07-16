import yaml
"""
Script principal para el entrenamiento y evaluación de un modelo MLP (Perceptrón Multicapa)
para la predicción de complicaciones médicas. El flujo general incluye:

1. Preprocesamiento de datos.
2. Entrenamiento del modelo MLP con búsqueda de hiperparámetros.
3. Evaluación del modelo y generación de reportes.

La configuración del pipeline (rutas de datos y parámetros del modelo)
se controla mediante el archivo YAML de configuración.
"""
from src.preprocessing.processing import preprocessed
from src.model.mlp_model import train_mlp
from src.evaluation.metrics import evaluate_models

with open('config/mlp_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Definir rutas y parámetros a partir del archivo de configuración
raw_path = config['data_paths']['raw']                     # Ruta a los datos crudos
preprocessed_path = config['data_paths']['preprocessed']   # Ruta a los datos preprocesados
results_path = config['data_paths']['results']             # Carpeta para guardar resultados del modelo
results_dir = 'mlp_result.csv'                             # Nombre del archivo CSV de resultados
model_params = config['model_params']                      # Parámetros del modelo MLP
confusion_matrix_path = config['report_paths']['confusion_matrix']  # Ruta para guardar matriz de confusión
report_excel__path = config['report_paths']['report_excel']         # Ruta para guardar el reporte general en Excel


if __name__ == '__main__':
    print("Starting preprocessing...")

    preprocessed(
        input_path=raw_path,
        out_path=preprocessed_path,
    )
    train_mlp(
        data_dir=preprocessed_path,
        results_dir=results_path,
        results_file=results_dir,
        model_params=model_params,
    )
    evaluate_models(
        results_csv=f"{results_path}/{results_dir}",
        output_dir=confusion_matrix_path,
        report_dir=report_excel__path
    )
