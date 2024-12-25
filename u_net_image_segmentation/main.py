import ssl

from mlp.cli.cli import cli

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    cli()

"""
cd u_net_image_segmentation
poetry run python main.py \
model train \
--training_class u_net.UNet \
--trainer_config_path configs/params.yaml \
--data_access_class datasets.AnnotationsData
"""