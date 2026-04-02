import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HYDE_PROMPT = """You are a regulatory document analyst.
Given a compliance question, write a short paragraph (3-5 sentences) that would 
appear in an authoritative regulatory document directly answering this question.
Use formal regulatory language, be specific, and include relevant section references 
if applicable. Do not answer conversationally — write as if you are the document itself.

Question: {query}

Regulatory excerpt:"""


def expand_query_hyde(query: str, n_hypothetical: int = 2) -> list[str]:
    """
    Generates n hypothetical document excerpts for a query.
    These are used to create richer embedding vectors for retrieval —
    never shown to the user directly.

    Generates multiple hypotheticals and returns all of them so the
    caller can embed each one and merge the retrieved results.
    """
    hypotheticals = []

    for i in range(n_hypothetical):
        response = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": HYDE_PROMPT.format(query=query)
                }
            ],
            # Low temperature keeps it focused on plausible regulatory language
            # rather than creative variations
            temperature=0.3,
            max_tokens=200,
        )
        hypotheticals.append(response.choices[0].message.content.strip())

    return hypotheticals
