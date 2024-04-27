from .rag import rag_query

import json
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core import BasePromptTemplate, ChatPromptTemplate, PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI


DEFAULT_SYSTEM_TEMPLATE = """
Sie sind ein Expertenbewertungssystem für einen Chatbot, der Fragen beantwortet.

Sie erhalten die folgenden Informationen:
- eine Benutzeranfrage und
- eine generierte Antwort

Sie können auch eine Referenzantwort erhalten, die Sie als Referenz für Ihre Bewertung verwenden können.

Ihre Aufgabe ist es, die Relevanz und Korrektheit der generierten Antwort zu beurteilen.
Geben Sie eine einzelne Punktzahl aus, die eine ganzheitliche Bewertung darstellt.
Sie müssen Ihre Antwort in einer Zeile zurückgeben, die nur die Punktzahl enthält.
Geben Sie keine Antworten in einem anderen Format zurück.
Geben Sie in einer separaten Zeile auch Ihre Begründung für die Bewertung an.

Befolgen Sie die folgenden Richtlinien für die Punktevergabe:
- Ihre Punktzahl muss zwischen 1 und 5 liegen, wobei 1 die schlechteste und 5 die beste Note ist.
- Wenn die generierte Antwort für die Benutzeranfrage nicht relevant ist, \
sollten Sie eine Punktzahl von 1 vergeben.
- Wenn die generierte Antwort relevant ist, aber Fehler enthält, \
sollten Sie eine Punktzahl zwischen 2 und 3 vergeben.
- Wenn die generierte Antwort relevant und vollständig korrekt ist, \
sollten Sie eine Punktzahl zwischen 4 und 5 vergeben.

Beispielantwort:
4.0
Die generierte Antwort hat genau dieselben Metriken wie die Referenzantwort, \
    aber sie ist nicht so prägnant.

Übersetzt mit DeepL.com (kostenlose Version)
"""

DEFAULT_USER_TEMPLATE = """
## Benutzerabfrage
{query}

## Referenz-Antwort
{reference_answer}

## Generierte Antwort
{generated_answer}
"""

CUSTOM_EVAL_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
    ]
)


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

    with open('tmp/eval_set_with_answers.json', 'r', encoding='utf-8') as f:
        eval_set = json.load(f)

    ele = eval_set[0]
    query = ele['query']
    response = ele['response']
    reference = ele['expected_answer']

    llm = OpenAI("gpt-4")

    evaluator = CorrectnessEvaluator(llm=llm, eval_template=CUSTOM_EVAL_TEMPLATE)

    results = []
    for ele in eval_set:
        query = ele['query']
        response = ele['response']
        reference = ele['expected_answer']

        result = evaluator.evaluate(
            query=query,
            response=response,
            reference=reference,
        )

        result = {
            'query': query,
            'response': response,
            'reference': reference,
            'score': result.score,
            'feedback': result.feedback

        }

        results.append(result)

    with open('tmp/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
