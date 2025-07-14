# Column name mappings (original -> new)
column_mappings = {
    "No": None,
    "Nombre y Apellidos": None,  # None indicates column to drop
    "Edad": "age",
    "Sexo": "sex",
    "Dirrección": None,
    "Municipio": None,
    "Fecha": None,
    "Hora": None,
    "Antecedentes Patológicos": "medical_history",
    "Causa": None,
    "Clasificación": "classification",
    "Localización": None,
    "T max (ºC)": None,
    "T min (ºC)": None,
    "T med (ºC)": None,
    "Hr med (%)": None,
    "precipitaciones (mm)": None,
    "Pe med (hPa)": None,
    "tiempo quirúrgico": "surgical_time",
    "Tipo de tratamiento": "treatment_type",
    "Evolucion al mes": None, #"evolution_1_month",
    "Evolucion a los 3 meses": None, # "evolution_3_months",
    "Evolucion a los 6 meses": None, #"evolution_6_months",
    "Complicaciones": "complications",
}

# Value mappings for specific columns
value_mappings = {
    "complications": {
        "bien": "no",
        "no": "no",
        # Cualquier otro valor se convertirá a "si"
    },
    'treatment_type': {
        'osteosintesis': 'osteosynthesis',
        'no operado': 'untreated',
        'protesis': 'prosthesis',
        'tornillos': 'screws',
        'mipo': 'mipo'
    },
    'classification': {
        'basicervical': 'basicervical_fracture',
        'intertrocanterica': 'intertrochanteric_fracture',
        'subcapital': 'subcapital_fracture',
        'transcervical': 'transcervical_fracture',
        'subtrocanterica': 'subtrochanteric_fracture'
    },
    "evolution_1_month": {
        "bien": "good",
        "mal": "bad",
        "sepsis": "bad",
        "regular": "regular",
    },
    # "evolution_3_months": {
    #     "bien": "good",
    #     "mal": "bad",
    #     "sepsis": "bad",
    #     "regular": "regular",
    # },
    # "evolution_6_months": {
    #     "bien": "good",
    #     "mal": "bad",
    #     "sepsis": "bad",
    #     "regular": "regular",
    # },
    "medical_history": {
        "ninguna": "none",
    },
    "surgical_time": {
        "": "none",
    }

}
