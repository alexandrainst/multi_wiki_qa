"""Generate the dataset for all the languages.

Usage:
    uv run src/scripts/generate_dataset.py <config_key>=<config_value> ...
"""

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from multi_wiki_qa.dataset_generation import build_dataset

load_dotenv()


@hydra.main(config_path="../../config", config_name="generation", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    build_dataset(config=config)


if __name__ == "__main__":
    main()
