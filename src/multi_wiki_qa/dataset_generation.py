"""Generating the dataset."""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from shutil import rmtree
from time import sleep

import hanzidentifier as hanz
import pandas as pd
from datasets import Dataset, load_dataset
from litellm.exceptions import InternalServerError
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers.pipelines import pipeline

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
                name=get_wikipedia_subset(language_code=config.language_code),
                split="train",
            )
            .shuffle(seed=config.seed)
            .filter(
                function=lambda x: len(x["text"]) > config.min_article_length,
                num_proc=mp.cpu_count(),
                desc="Filtering articles by length",
            )
        )
        assert isinstance(dataset, Dataset)
    except ValueError:
        logger.error(f"The {language!r} Wikipedia dataset is not available. Skipping.")
        return

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

    # Special case for Mandarin
    if config.language_code == "zh-cn":
        dataset = dataset.filter(
            lambda sample: hanz.identify(sample["text"]) == hanz.SIMPLIFIED,
            desc="Filtering out traditional Chinese articles (zh-cn only)",
        )
    elif config.language_code == "zh-tw":
        dataset = dataset.filter(
            lambda sample: hanz.identify(sample["text"]) == hanz.TRADITIONAL,
            desc="Filtering out simplified Chinese articles (zh-tw only)",
        )

    # Special case for Portuguese
    if config.language_code in {"pt-pt", "pt-br"}:
        logger.info("Loading the Portuguese language classifier...")
        classifier = pipeline(task="text-classification", model="liaad/PtVId")
        indices_to_keep: list[int] = []
        with tqdm(
            total=config.num_samples,
            desc=f"Selecting {config.language_code.upper()} articles",
        ) as pbar:
            for i, sample in enumerate(dataset):
                prediction = classifier(
                    inputs=sample["text"], truncation=True, max_length=512
                )[0]["label"]
                if prediction == config.language_code.upper():
                    indices_to_keep.append(i)
                    pbar.update()
                if len(indices_to_keep) >= config.num_samples:
                    break
        dataset = dataset.select(indices_to_keep)
        logger.info(
            f"Selected {len(indices_to_keep):,} articles in "
            f"{config.language_code.upper()}."
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

            num_attempts = 10
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
                        sleep(5 * 60)
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
                    question_invalid = not isinstance(generated_sample["question"], str)
                    answer_invalid = not isinstance(generated_sample["answer"], str)
                    if question_invalid or answer_invalid:
                        continue
                    record = dict(
                        id=sample["url"],
                        title=sample["title"],
                        context=sample["text"],
                        question=generated_sample["question"],
                        answers=dict(
                            text=[generated_sample["answer"]],
                            answer_start=[
                                sample["text"].find(generated_sample["answer"])
                            ],
                        ),
                    )
                    records.append(record)
                    f.write(json.dumps(record) + "\n")
                    pbar.update()

    logger.info("Converting the records to a Hugging Face dataset...")
    df = pd.DataFrame.from_records(records)
    df = df[df.question.map(lambda x: isinstance(x, str))]
    df = df[df.answers.map(lambda x: isinstance(x["text"][0], str))]
    assert isinstance(df, pd.DataFrame)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    logger.info("Saving the dataset to disk...")
    dataset_path = Path("data", "final", config.language_code)
    dataset.save_to_disk(dataset_path)
    logger.info(f"Dataset saved to {dataset_path}.")

    logger.info("Removing the temporary records file...")
    records_path.unlink(missing_ok=True)

    if config.push_to_hub:
        logger.info("Pushing the dataset to the Hugging Face Hub...")
        dataset.push_to_hub(
            config.hub_id, config_name=config.language_code, private=True
        )

        logger.info("Removing the local dataset directory...")
        rmtree(dataset_path, ignore_errors=True)

    logger.info("All done!")


def get_wikipedia_subset(language_code: str) -> str:
    """Get the name of the Wikipedia subset for a given language code.

    Args:
        language_code:
            The language code to get the Wikipedia subset for.

    Returns:
        The name of the Wikipedia subset.
    """
    if language_code == "zh-cn" or language_code == "zh-tw":
        language_code = "zh"
    elif language_code == "pt-pt" or language_code == "pt-br":
        language_code = "pt"
    elif language_code == "yue":
        language_code = "zh-yue"
    return f"20231101.{language_code}"
