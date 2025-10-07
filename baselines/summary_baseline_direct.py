
llm = None

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def process(line):
    key = line['id']
    try:
        print(key)
        PROMPT = \
"""You are an AI assistant. Your task is to create a summary of the given dialogue.
Please only give the summary and never output any other information.
dialogue:
%s
"""
        content = line['dialogue']
        prompt = PROMPT % content

        response = llm.predict(prompt)

        print(response)
        print("="*50)

        line['response'] = response
        return line

    except Exception as e:
        print(e)
        print(f"{key} error")
        return line



if __name__ == "__main__":
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

    with open("../result/summary_only_LLM.json", "w+") as f:
        json.dump(results, f, indent=4)

