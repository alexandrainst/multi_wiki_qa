"""LLM generation with LiteLLM."""

import json
import logging

import litellm
from litellm.types.completion import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from litellm.types.utils import Choices, ModelResponse

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


def generate_samples_from_context(
    article: str,
    language: str,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    prompt: str,
    follow_up_prompt: str,
) -> list[dict[str, str]]:
    """Generate a list of (context, question, answer) dictionaries from an article.

    Args:
        article:
            The article to generate questions from.
        language:
            The name of the language that the article is written in.
        model:
            The model to use for generation.
        max_tokens:
            The maximum number of tokens to generate.
        temperature:
            The temperature to use for generation.
        system_prompt:
            The system prompt to use for generation.
        prompt:
            The prompt to use for generation.
        follow_up_prompt:
            The follow-up prompt to use for generation.

    Returns:
        A list of dictionaries containing the generated text.
    """
    client = litellm.LiteLLM()

    model_output = client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(
                role="system", content=system_prompt.format(language=language)
            ),
            ChatCompletionUserMessageParam(
                role="user", content=prompt.format(article=article, language=language)
            ),
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=dict(type="json_object"),
    )
    assert isinstance(model_output, ModelResponse)

    choices = model_output.choices[0]
    assert isinstance(choices, Choices)
    generation_output = choices.message.content
    assert isinstance(generation_output, str)
    json_obj = json.loads(generation_output)
    assert isinstance(json_obj, dict) and "results" in json_obj

    generated_samples = list()
    for generated_sample in json_obj["results"]:
        if (
            all(key in generated_sample for key in ["question", "answer"])
            and generated_sample["answer"] in article
        ):
            generated_samples.append(generated_sample)

    # Re-phrase the generated questions
    for generated_sample in generated_samples:
        model_output = client.chat.completions.create(
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system", content=system_prompt.format(language=language)
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=follow_up_prompt.format(
                        question=generated_sample["question"], language=language
                    ),
                ),
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=dict(type="json_object"),
        )
        assert isinstance(model_output, ModelResponse)
        choices = model_output.choices[0]
        assert isinstance(choices, Choices)
        generation_output = choices.message.content
        assert isinstance(generation_output, str)
        json_obj = json.loads(generation_output)
        assert isinstance(json_obj, dict) and "question" in json_obj
        generated_sample["original_question"] = generated_sample["question"]
        generated_sample["question"] = json_obj["question"]

    return generated_samples
