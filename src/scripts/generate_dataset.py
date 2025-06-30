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

        # Get the list of available Wikipedia languages
        wikipedia_languages = [
            cfg["config_name"].split(".")[-1]
            for cfg in api.repo_info(
                repo_id="wikimedia/wikipedia", repo_type="dataset"
            ).card_data.configs
        ]
        # Special case for Chinese, as we want to split it into Simplified and
        # Traditional Chinese
        if "zh" in wikipedia_languages:
            wikipedia_languages.remove("zh")
            wikipedia_languages.append("zh-cn")
            wikipedia_languages.append("zh-tw")

        # Skip already generated languages
        try:
            already_generated_languages = [
                cfg["config_name"].split(".")[-1]
                for cfg in api.repo_info(
                    repo_id=config.hub_id, repo_type="dataset"
                ).card_data.configs
            ]
        except RepositoryNotFoundError:
            logger.warning(f"Repository {config.hub_id} not found.")
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

        # Filter languages to only keep those in the mapping and not already generated
        languages_in_mapping = list(LANGUAGE_MAPPING.keys())
        language_codes = sorted(
            (set(wikipedia_languages) & set(languages_in_mapping))
            - set(already_generated_languages)
        )

        # Build the dataset for each language
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
