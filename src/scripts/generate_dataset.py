"""Generate the dataset for all the languages.

Usage:
    uv run src/scripts/generate_dataset.py <config_key>=<config_value> ...
"""

import logging

import hydra
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from omegaconf import DictConfig

from multi_wiki_qa.constants import LANGUAGE_MAPPING
from multi_wiki_qa.dataset_generation import build_dataset

load_dotenv()

logger = logging.getLogger("generate_dataset")


@hydra.main(config_path="../../config", config_name="generation", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    if config.language_code == "all":
        api = HfApi()
        wikipedia_languages = [
            cfg["config_name"].split(".")[-1]
            for cfg in api.repo_info(
                repo_id="wikimedia/wikipedia", repo_type="dataset"
            ).card_data.configs
        ]
        languages_in_mapping = list(LANGUAGE_MAPPING.keys())
        try:
            already_generated_languages = [
                cfg["config_name"].split(".")[-1]
                for cfg in api.repo_info(
                    repo_id=config.hub_id, repo_type="dataset"
                ).card_data.configs
            ]
        except RepositoryNotFoundError:
            already_generated_languages = []
        if already_generated_languages:
            logger.info(
                f"Skipping already generated languages: "
                f"{', '.join(already_generated_languages)}"
            )
        else:
            logger.info(
                "No languages have been generated yet. Generating all languages."
            )
        language_codes = sorted(
            (set(wikipedia_languages) & set(languages_in_mapping))
            - set(already_generated_languages)
        )

        # Special case for Mandarin
        if "zh" in language_codes:
            language_codes.remove("zh")
            language_codes.append("zh-cn")
            language_codes.append("zh-tw")

        for idx, language_code in enumerate(language_codes):
            config.language_code = language_code
            try:
                build_dataset(config=config)
            except Exception as e:
                logger.error(
                    f"An error occurred while generating the dataset for "
                    f"{language_code}: {e}. Skipping this language."
                )
                continue
            logger.info(
                f"Finished generating dataset for {language_code} "
                f"({idx + 1}/{len(language_codes)})"
            )
    else:
        build_dataset(config=config)


if __name__ == "__main__":
    main()
