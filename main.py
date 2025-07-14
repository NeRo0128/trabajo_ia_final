import yaml

from src.data.preprocessing.processing import preprocessed
from src.model.mlp_model import train_mlp

if __name__ == '__main__':
    print("Starting preprocessing...")
    with open('config/mlp_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    preprocessed(
        config['data_paths']['raw'],
        config['data_paths']['preprocessed'],
    )

    train_mlp(
        config['data_paths']['preprocessed'],
        config['data_paths']['results'],
        'mlp_result.csv',
        config['model_params'],
    )

