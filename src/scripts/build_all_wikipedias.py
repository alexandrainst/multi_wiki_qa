"""Build all Wikipedia datasets.

Usage:
    uv run src/scripts/build_all_wikipedias.py <config_key>=<config_value> ...
"""

import hydra
from huggingface_hub import HfApi
from omegaconf import DictConfig
from tqdm.auto import tqdm

from multi_wiki_qa.wikipedia_building import build_wikipedia_dataset


@hydra.main(config_path="../../config", config_name="wikipedia", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    api = HfApi()
    card_data = api.repo_info(repo_id=config.repo_id, repo_type="dataset").card_data
    languages_to_skip = [dct["config_name"] for dct in card_data.dataset_info]
    languages = [
        language for language in config.languages if language not in languages_to_skip
    ]

    for language in tqdm(languages, desc="Building Wikipedia datasets"):
        build_wikipedia_dataset(
            language=language, date_str=config.dump_date, repo_id=config.repo_id
        )


if __name__ == "__main__":
    main()
