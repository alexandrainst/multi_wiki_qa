"""Generating the dataset."""

import json
import logging
from pathlib import Path
from shutil import rmtree
from time import sleep

import pandas as pd
from datasets import Dataset, load_dataset
from litellm.exceptions import InternalServerError
from nlp_dedup import Deduper
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .constants import LANGUAGE_MAPPING
from .litellm import generate_samples_from_context

logger = logging.getLogger(__name__)


def build_dataset(config: DictConfig) -> None:
    """Generate the question answering dataset in a given language.

    Args:
        config:
            The Hydra configuration object.
    """
    try:
        language = LANGUAGE_MAPPING[config.language_code]
    except KeyError:
        logger.error(
            f"The language code {config.language_code!r} is not supported. "
            "Please check the configuration."
        )
        return

    logger.info(f"Loading the {language} Wikipedia dataset...")
    try:
        dataset = (
            load_dataset(
                "wikimedia/wikipedia",
                name=f"20231101.{config.language_code}",
                split="train",
            )
            .shuffle(seed=config.seed)
            .filter(lambda x: len(x["text"]) > config.min_article_length)
        )
        assert isinstance(dataset, Dataset)
    except ValueError:
        logger.error(f"The {language!r} Wikipedia dataset is not available. Skipping.")
        return

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
    rmtree("deduplicated")

    if len(dataset) < config.min_num_articles:
        logger.error(
            f"The {language!r} Wikipedia dataset has only {len(dataset):,} "
            f"articles. Skipping."
        )
        return

    records_path = Path("data", "processed", f"{config.language_code}-records.jsonl")
    if records_path.exists():
        with records_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = list()

    # Remove the existing records
    existing_urls = {record["id"] for record in records}
    if existing_urls:
        dataset = dataset.filter(
            lambda sample: sample["url"] not in existing_urls,
            desc=f"Removing samples from {len(existing_urls):,} Wikipedia articles "
            "that we already have generated samples for",
        )

    with tqdm(
        desc=f"Generating samples with {config.model}", total=config.num_samples
    ) as pbar:
        pbar.update(len(records))
        for sample in dataset:
            assert isinstance(sample, dict)
            if len(records) >= config.num_samples:
                logger.info(
                    f"Reached the target number of samples ({config.num_samples:,}). "
                    "Stopping."
                )
                break

            num_attempts = 5
            errors: list[Exception] = list()
            for _ in range(num_attempts):
                try:
                    generated_samples = generate_samples_from_context(
                        article=sample["text"],
                        language=language,
                        model=config.model,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        system_prompt=config.system_prompt,
                        prompt=config.prompt,
                        follow_up_prompt=config.follow_up_prompt,
                    )
                    break
                except InternalServerError as e:
                    errors.append(e)
                    if "try again later" in str(e):
                        sleep(60)
                        continue
                except Exception as e:
                    errors.append(e)
                    continue
            else:
                logger.info(
                    f"Failed to generate samples for {sample['url']} after "
                    f"{num_attempts} attempts. The errors were: {errors}"
                )
                continue

            with records_path.open("a") as f:
                for generated_sample in generated_samples:
                    record = dict(
                        id=sample["url"],
                        title=sample["title"],
                        context=sample["text"],
                        question=generated_sample["question"],
                        answers=dict(
                            text=[generated_sample["answer"]],
                            answer_start=sample["text"].find(
                                generated_sample["answer"]
                            ),
                        ),
                    )
                    records.append(record)
                    f.write(json.dumps(record) + "\n")
                    pbar.update()

    logger.info("Converting the records to a Hugging Face dataset...")
    df = pd.DataFrame.from_records(records)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    logger.info("Saving the dataset to disk...")
    dataset_path = Path("data", "final", config.language_code)
    dataset.save_to_disk(dataset_path)
    logger.info(f"Dataset saved to {dataset_path}.")

    if config.push_to_hub:
        logger.info("Pushing the dataset to the Hugging Face Hub...")
        dataset.push_to_hub(
            config.hub_id, config_name=config.language_code, private=True
        )

    logger.info("All done!")
