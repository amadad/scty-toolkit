import os
import getpass
from openai import OpenAI
from tqdm.auto import tqdm
import json

TOPIC_GENERATION_PROMPT_TEMPLATE = """Given the following topic, generate a list of {n_subtopics} subtopics that are related to the topic.
The topic is: {topic}
The list must be without numbers, and without any description of the subtopics. The subtopics should be separated by a comma. There must be no other text than the list.
"""

QUESTION_PROMPT_TEMPLATE = """Given the following topic, generate {n_questions} questions that could be asked about that topic. Your response should be in a list format.
The topic is: {sub_topic}
The list must be without numbers. The questions should be separated by a newline character. There must be no other text than the list.
"""

RESPONSE_PROMPT_TEMPLATE = """Given a question, generate 2 responses that could be given to that question. Your response should be in a list format.
The question is: {question}

The list must be in the format:

RESPONSE A: Response A text here
RESPONSE B: Response B text here
"""

# Prompt for NVIDIA API key if not set as an environment variable
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_api_key)

topic = "Computer Science"
n_subtopics = 3
n_questions = 2

def generate_subtopics(client, topic, n_subtopics):
    prompt = TOPIC_GENERATION_PROMPT_TEMPLATE.format(topic=topic, n_subtopics=n_subtopics)
    response = client.chat.completions.create(
        model="nvidia/nemotron-4-340b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response

responses = generate_subtopics(client, topic=topic, n_subtopics=n_subtopics)
print(responses.choices[0].message.content)

subtopic_list = [x.strip() for x in responses.choices[0].message.content.split(",")]
print(subtopic_list)

def generate_questions(client, sub_topic, n_questions):
    prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
    response = client.chat.completions.create(
        model="nvidia/nemotron-4-340b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def question_generator(client, subtopic_list, n_question):
    question_list = [generate_questions(client, subtopic, n_question) for subtopic in tqdm(subtopic_list)]
    return question_list

question_list = question_generator(client, subtopic_list, n_questions)
print(question_list)

question_list_formatted = []
for question_set in question_list:
    question_list_formatted += question_set.split("\n")

question_list_formatted = [x for x in question_list_formatted if x]
print(question_list_formatted)
print(len(question_list_formatted))

def generate_responses(client, question):
    prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
    response = client.chat.completions.create(
        model="nvidia/nemotron-4-340b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def response_generator(client, question_list):
    response_list = [generate_responses(client, question) for question in tqdm(question_list)]
    return response_list

question_response_list = response_generator(client, question_list_formatted)

question_response_pair_list = []
for question, response_set in zip(question_list_formatted, question_response_list):
    question_response_pair_list.append(
        {
            "question": question,
            "responses": {
                "response_a": {"response": response_set.split("RESPONSE B:")[0].replace("RESPONSE A:", "").strip()},
                "response_b": {"response": response_set.split("RESPONSE B:")[-1].split("\n\n")[0].strip()}
            },
        }
    )

with open('synthetic_data.jsonl', 'w') as f:
    for item in question_response_pair_list:
        f.write(json.dumps(item))
        f.write('\n')

print("Data saved to synthetic_data.jsonl")