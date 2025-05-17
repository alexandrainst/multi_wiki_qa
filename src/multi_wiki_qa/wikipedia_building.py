"""Building Wikipedia datasets."""

import logging
from shutil import rmtree

from datasets import Dataset, load_dataset
from nlp_dedup import Deduper

logger = logging.getLogger(__name__)


def build_wikipedia_dataset(language: str, date_str: str, repo_id: str) -> Dataset:
    """Build Wikipedia dataset for a given language and date.

    Args:
        language:
            The language code for the Wikipedia dataset (e.g., 'en' for English).
        date_str:
            The date string in the format 'YYYYMMDD' for the Wikipedia dump.
        repo_id:
            The repository ID for the Hugging Face Hub where the dataset will be pushed.

    Returns:
        The Wikipedia dataset for the specified language and date.
    """
    dataset = load_dataset(
        "wikipedia",
        language=language,
        date=date_str,
        trust_remote_code=True,
        split="train",
        cache_dir=".cache",
    )
    assert isinstance(dataset, Dataset)

    # Deduplicate the dataset
    deduper = Deduper(return_generator=True)
    indices_to_keep = [
        idx
        for idx, dct in enumerate(
            deduper.deduplicate(
                corpus=dataset, output_dir="deduplicated", overwrite=True
            )
        )
        if not dct["duplicate"]
    ]
    dataset = dataset.select(indices_to_keep)

    # Save the dataset to the Hugging Face Hub
    logger.info(f"Pushing {language} Wikipedia dataset to the Hugging Face Hub...")
    dataset.push_to_hub(
        repo_id=repo_id,
        config_name=language,
        commit_message=f"Add {language} Wikipedia dataset",
        max_shard_size="1GB",
        private=True,
    )

    # Clean up temporary directories
    rmtree(".cache", ignore_errors=True)
    rmtree("deduplicated", ignore_errors=True)

    return dataset
