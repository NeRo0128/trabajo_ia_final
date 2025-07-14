import os

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OrdinalEncoder
from sklearn.utils import resample

import src.data.preprocessing.mappings as maps

"""
#todo:
revisar q el xlsx se genero de la forma correcta q necesito
"""


def map_value(col: str, val) -> str:
    """
    Mapea un valor según la especificación en mappings.py

    Args:
        col: nombre de la columna
        val: valor a mapear

    Returns:
        valor mapeado o  una string sin espacios y todo a miusculas
    """
    if pd.isna(val):
        return val

    txt = str(val).strip().lower()
    if col in maps.value_mappings:
        mapping = maps.value_mappings[col]
        if txt in mapping:
            return mapping[txt]
        # para complications: todo lo no mapeado es 'yes' 
        if col == "complications":
            return 'yes'
    return txt


def parse_time(val: str) -> int | float:
    # todo: document
    # todo: no se analizando correctamente algunos casos
    if val == "" or pd.isna(val):
        return 0
    if '-' in val:
        val = val.replace("mint", "min")
        val = val.replace("minut", "min")
        lo, hi = val.replace("min", "").split('-')
        return (float(lo) + float(hi)) / 2
    try:
        return float(val.replace("min", ""))
    except:
        return 0


def preprocessed(input_path: str, out_path: str):
    """
    Carga la base limpia desde excel, aplica:
      - renombrado de columnas
      - transformaciones de cada campo según especificación
      - sobremuestreo de la clase positiva de 'complications'
      - guarda los datos preprocesados para entrenamiento
      - exporta un Excel de inspección

    Args:
        input_path: ruta al .xlsx con la base de datos
        out_path: carpeta donde guardar los outputs
    """

    os.makedirs(out_path, exist_ok=True)
    df = pd.read_excel(input_path)

    # * Rename or drop columns
    rename_map = {orig: new for orig, new in maps.column_mappings.items() if new}
    drop_cols = [orig for orig, new in maps.column_mappings.items() if new is None]
    df: DataFrame = df.drop(columns=drop_cols).rename(columns=rename_map)

    # * Normalize using value_mappings
    for col in df.columns:
        df[col] = df[col].apply(lambda v, c=col: map_value(c, v))

    # * Specific transformations
    # age:
    scaler = MinMaxScaler()
    df['age'] = scaler.fit_transform(df[['age']])

    # sex
    df['sex'] = (df['sex'] == 'f').astype(int)

    # surgical_time
    df['surgical_time'] = df['surgical_time'].apply(parse_time)

    # treatment_type & classification → one-hot
    df = pd.get_dummies(
        df,
        columns=["treatment_type", "classification"],
        prefix=["treat", "class"],
        dtype=int
    )

    # medical_history → multi-hot
    mlb = MultiLabelBinarizer()
    df["medical_history"] = df["medical_history"].fillna("").apply(
        lambda s: s if isinstance(s, list) else [x.strip() for x in s.split(",") if x.strip()]
    )

    mh_arr = mlb.fit_transform(df["medical_history"])
    mh = pd.DataFrame(mh_arr, columns=[f"mh_{c}" for c in mlb.classes_], index=df.index)
    df = pd.concat([df.drop(columns=["medical_history"]), mh], axis=1)

    # # evolutions -> ordinal encode
    # oe = OrdinalEncoder(categories=[['good', 'regular', 'bad']])
    # for col in ["evolution_1_month", "evolution_3_months", "evolution_6_months"]:
    #     df[[col]] = oe.fit_transform(df[[col]])

    # complications -> binary
    df['complications'] = df["complications"].replace(
        {"no": 0, "yes": 1})  # todo: ver esto para q no levante mas el warn

    # * Features / Target
    x = df.drop(columns=["complications"])
    y = df["complications"]

    df.to_excel(os.path.join(out_path, "df_preprocessed.xlsx"), index=False)

    # 80% train, 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # * Oversampling
    df_train = pd.concat([x_train, y_train], axis=1)

    # Sobremuestreo de la clase positiva
    df_pos = df[df['complications'] == 1]
    df_neg = df[df['complications'] == 0]

    # Aquí igualas las cantidades (sobremuestreo verdadero)
    df_pos_up = resample(df_pos, replace=True, n_samples=len(df_neg), random_state=42)

    df_bal = pd.concat([df_neg, df_pos_up]).sample(frac=1, random_state=42)

    # separa de nuevo X_train e y_train balanceados
    x_train = df_bal.drop(columns=["complications"])
    y_train = df_bal["complications"]

    # *Save Results
    x_train.to_pickle(os.path.join(out_path, "x_train.pkl"))
    y_train.to_pickle(os.path.join(out_path, "y_train.pkl"))
    x_test.to_pickle(os.path.join(out_path, "x_test.pkl"))
    y_test.to_pickle(os.path.join(out_path, "y_test.pkl"))

    print(f"Preprocesado completado. Datos guardados en {out_path}")


if __name__ == "__main__":
    import yaml

    with open('config/mlp_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    out_dir = config['data']['preprocessed']
    input_path = config['data']['raw']

    preprocessed(input_path, out_dir)
