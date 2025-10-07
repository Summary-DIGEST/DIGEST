import json
from concurrent.futures import ThreadPoolExecutor, as_completed

llm = None

questions = {
    "topic": "What is the main topic or focus of the content?",
    "key_pts": "What are the key points or arguments presented?",
    "entities": "Who are the three main entities or individuals involved, and what roles do they play?",
    # "timeline": "Which timeline, if any, is being discussed here?",
    # "details": "What are the supporting details, examples, or evidence provided?",
    "conclude": "What conclusions, recommendations, impacts, or implications are mentioned, if any?",
    # "tone": "What is the overall tone or sentiment (e.g., objective, critical, positive, negative, etc.)?",
    # "challenges": "What questions or challenges does the content raise?",
    # "insights": "What unique insights or perspectives are offered?",
    # "audience": "What audience is the content aimed at, and how does this affect its presentation?"
}


def process(line):
    key = line['id']

    question_prompt = "Before you make summary, please first answer the following question precisely without any additional detail: \n"
    for q in questions.keys():
        question_prompt+=f"{questions[q]}\n"
    question_prompt += "Given the following article, generate a summary of the article."

    try:
        print(key)
        PROMPT = \
"""You are an AI assistant. Your task is to create a summary of the given dialogue. 
Please only give the summary and never output any other information.

dialogue:
%s

%s
Please follow the given format:
{
    "QA_answers": "give the answer of the questions",
    "summary": "your summary"
}
"""
        content = line['dialogue']

        prompt = PROMPT % (content, question_prompt)

        print(prompt)

        response = llm.predict(prompt)
        print(response)
        ret = json.loads(response)
        response = ret["summary"]
        print(response)
        print("="*50)

        line['response'] = response

        return line

    except Exception as e:
        print(e)
        print(f"{key} error")
        return line



with open("../data/datas.jsonl", "r") as f:
    datas = json.load(f)

results = []

with ThreadPoolExecutor(max_workers=30) as executor:
    future_to_line = {executor.submit(process, line): line for line in datas}

    for future in as_completed(future_to_line):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print("Future error:", e)

with open("../result/QA_Sum.json", "w+") as f:
    json.dump(results, f, indent=4)


