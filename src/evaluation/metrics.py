import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(
        results_csv: str,
        output_dir: str,
        report_dir: str,
):
    """
      Evalúa modelos basados en los resultados guardados en un archivo CSV.
      Genera un reporte en formato Excel con las métricas y visualiza matrices de confusión.

        Params
      -----------
      results_csv : str
          Ruta al archivo CSV que contiene las métricas de los modelos evaluados.
          Debe contener las columnas: 'true_positive', 'false_positive', 'false_negative', 'true_negative'.

      output_dir : str
          Directorio donde se guardarán las imágenes de las matrices de confusión.

      report_dir : str
          Directorio donde se almacenará el reporte Excel comparativo.
      """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df = pd.read_csv(results_csv)

    # Guardar todas las métricas como un archivo Excel
    report_path = os.path.join(report_dir, 'reporte_comparativo_modelos.xlsx')
    df.to_excel(report_path, index=False)
    print(f"Reporte comparativo guardado en {report_path}")

    # Generar matrices de confusión
    for idx, row in df.iterrows():
        tn = int(row['true_negative'])
        fp = int(row['false_positive'])
        fn = int(row['false_negative'])
        tp = int(row['true_positive'])

        cm = [[tn, fp],
              [fn, tp]]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión\nModelo {idx+1}')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')

        model_name = f"modelo_{idx+1}.png"
        plt.savefig(os.path.join(output_dir, model_name))
        plt.close()

    print(f"Generadas {len(df)} matrices de confusión en {output_dir}")
