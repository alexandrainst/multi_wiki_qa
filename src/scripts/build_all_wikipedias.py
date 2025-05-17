"""Build all Wikipedia datasets.

Usage:
    uv run src/scripts/build_all_wikipedias.py <config_key>=<config_value> ...
"""

import hydra
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
    for language in tqdm(config.languages, desc="Building Wikipedia datasets"):
        build_wikipedia_dataset(
            language=language, date_str=config.dump_date, repo_id=config.repo_id
        )


if __name__ == "__main__":
    main()
