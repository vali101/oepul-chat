from .rag import rag_query

import json
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.prompts import (
    BasePromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)

from llama_index.llms.openai import OpenAI


def generate_responses():
    # open eval set
    with open('data/eval_set_new.json', 'r', encoding='utf-8') as f:
        eval_set = json.load(f)

    eval_set_with_answers = []

    for item in eval_set:
        query = item['query']
        expected_header_paths = item['header_paths']
        expected_answer = item['expected_answer']

        response = rag_query(query, 'indices/oepul_index/')
        eval_set_with_answers.append({
            'query': query,
            'header_paths': expected_header_paths,
            'expected_answer': expected_answer,
            'response': str(response)
        })

    with open('tmp/eval_set_with_answers.json', 'w', encoding='utf-8') as f:
        json.dump(eval_set_with_answers, f, indent=4)

    return eval_set_with_answers


def evaluate_responses():
    # responses = generate_responses()

    llm = OpenAI("gpt-4")

    evaluator = CorrectnessEvaluator(llm=llm)


print(OpenAI().complete("Paul Graham is "))
