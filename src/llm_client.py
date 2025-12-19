from openai import OpenAI

import os


_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def openai_llm(prompt, model="gpt-4.1-mini", temperature=0.2):
    """
    Calls OpenAI ChatCompletion API
    Can be swapped with other LLMs as required
    """
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a senior data scientist."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content
