import ssl

from mlp.cli.cli import cli

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    cli()

"""
cd super_resolution_sub_pixel_cnn
poetry run python main.py \
model train \
--training_class super_resolution_sub_pixel.SuperResolution \
--trainer_config_path configs/params.yaml \
--data_access_class datasets.BSRData
"""