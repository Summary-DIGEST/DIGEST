import json
from concurrent.futures import ThreadPoolExecutor, as_completed

llm = None

def process(line):
    key = line['id']
    print(key)
    try:

        Element_aware_extraction = """
You are an AI assistant, you task is to extract key elements from a given dialogue by answer some questions.
Dialogue:
%s

What are the important entities in this document?
What are the important dates in this document?
What events are happening in this document?
What is the result of these events?
Please answer the above questions:
""" % line['dialogue']
        elements = llm.predict(Element_aware_extraction)

        PROMPT = \
"""You are an AI assistant. Your task is to create a summary of the given dialogue. 
Please only give the summary and never output any other information.

Key information:
%s

Dialogue:
%s

Let's integrate the above information, and make a summary for the given dialogue.
"""
        content = line['dialogue']

        prompt = PROMPT % (elements ,content)

        print(prompt)

        response = llm.predict(prompt)

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

with open("../result/Sum_CoT.json", "w+") as f:
    json.dump(results, f, indent=4)


