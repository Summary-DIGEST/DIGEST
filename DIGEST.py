
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

llm = None


def summary_dialogue(content):
    PROMPT = \
        """You are an AI assistant. Your task is to create a summary of the given dialogue. 
        The summary should focus on the main topic of the dialogue, and don't output other details.
        Please only give the summary and never output any other information.

        dialogue:
        %s
        """
    prompt = PROMPT % content
    response = llm.predict(prompt)
    summary_part = response.split('[Summary]:')[-1].strip()
    return summary_part




def merge_subsumaries(sub_sumarizes, tag=False):
    if (not tag):
        sub_sumarizes_text = "\n".join(sub_sumarizes)
        prompt = """
    You are an AI assistant. Your task is to determine whether the given sub_summaries are absolutely different scene.
    Different scene show the following features:
      - Time has clearly passed or the location changes or a new setting is introduced.  
      - The set of main characters in the conversation changes significantly.
    Output only one word:  
    - "True" if the conversation contains multiple scenes.  
    - "False" if the conversation contains only one scenes.

    Sub_summaries:
    %s
    """ % (sub_sumarizes_text)
        response = llm.predict(prompt)
    else:
        response = "true"
    # print(response)
    if ("true" in response.lower()):
        return connect_subsumaries(sub_sumarizes)
    else:
        return fusion_sub_summaries(sub_sumarizes)


def fusion_sub_summaries(sub_sumarizes):
    sub_summarize_text = "\n".join(sub_sumarizes)

    prompt = """You are an AI assistant. Your task is to create a summary of the given sub_summaries. 
The summary should focus on the main topics, and don't output other details.
Please only give the summary and never output any other information.
Please also try to keep the summary logical, if some content comes out of nowhere and has no logical connection to the context before or after it, you can delete it.
Please remove the unnecessary details and maintain that related to core topics.
Sub_summaries:
%s
""" % (sub_summarize_text)
    summary_str = llm.predict(prompt)
    # print(sub_sumarizes)
    return summary_str


def connect_subsumaries(sub_sumarizes):
    sub_summarize_text = "\n".join(sub_sumarizes)
    return sub_summarize_text


def determine_topics(sentences, relations):
    prompt = """
You are a dialogue topic analyzer.  
Your task is to determine whether a given conversation contains multiple topics (True) or only one topic (False).  
Focus on semantic meaning rather than surface word overlap.  

You are given two pieces of information:  
1. The conversation content.  
2. The reply relations between utterances (e.g., 2 → 1, 3 → None, 4 → 1).  

Judgment rules:  
1. Core meaning shift: If the conversation introduces ideas that belong to fundamentally different domains of knowledge or activities, mark as True.  
2. Contextual focus shift: If the same entities appear but the focus of discussion changes significantly, mark as True.  
3. If the reply relations show that several utterances map to "None" (not connected to earlier ones), this strongly suggests a new topic → lean towards True.  
4. Otherwise, if all utterances revolve around the same semantic field or topic domain, even with small sub-questions or details, mark as False.  
5. Time has clearly passed, the location changes, or a new set of participants/characters is introduced → mark as True.  

Output only one word:  
- "True" if the conversation contains multiple topics.  
- "False" if the conversation contains only one topic.

Dialogue:
%s

Reply relations:
%s
""" % (sentences, relations)
    response = llm.predict(prompt)
    # print(response)
    if ("true" in response.lower()):
        return True
    else:
        return False


def determine_dependent(sentences, relations, sub_topics):
    dialogue = "\n".join([f"{s['index']}" for s in sentences.values()])
    prompt = """
You are an AI assistant.  
Your task is to determine whether the given dialogue contains absolutely independent scenes.  

You are given three pieces of information:  
1. The original dialogue content.  
2. The reply relations between utterances (e.g., 2 → 1, 3 → None, 4 → 1).  
3. The detected subtopics of the dialogue.  
An "independent scene" shows the following features:  
- It is self-contained: time has clearly passed, the location changes, or a new setting is introduced.  
- The set of main characters in the conversation changes significantly.  
- Reply relations show many utterances mapped to "None" or forming disconnected branches, suggesting separation.  
- Subtopics belong to fundamentally different domains of knowledge or activities.  
Output only one word:  
- "True" if the conversation contains multiple independent scenes.  
- "False" if the conversation contains only one scene.  
Dialogue:  
%s  

Reply relations:  
%s  

Subtopics:  
%s  
""" % (dialogue, relations, sub_topics)
    response = llm.predict(prompt)
    # print(response)
    if ("false" in response.lower()):
        return False
    else:
        return True


def segment_dialogue(sentences, relations):
    dialogue = "\n".join([f"{s['index']}" for s in sentences.values()])
    PROMPT_TOPIC = """
Group the dialogue lines into major topics based on story content.
Please perform segmentation at a coarser granularity.
If they are discussing the same object or topic, please regard them as one topic.
If you believe that further segmentation will be too fragmented, please return null.

Output format:
{
    "<topic_name>": [line_id, line_id, ...],
    "<topic_name>": [line_id, line_id, ...]
}
or
null

Dialogue:
%s

Relations (for reference):
%s
""" % (dialogue, relations)
    response = llm.predict(PROMPT_TOPIC)
    print(response)
    try:
        ret = json.loads(response)
    except:
        ret = None

    if (ret is None or "null" in ret or "None" in ret or "none" in ret):
        return None
    else:
        if (len(ret.keys()) == 1):
            return None

        visited = []
        for value in ret.values():
            for v in value:
                if(v in visited):
                    return None
                visited.append(v)
        return ret


def get_relation(sentences):
    dialogue = "\n".join([f"{s['index']}" for s in sentences.values()])

    prompt = """
You will be given a dialogue in the following format:
1. [message content]  
2. [message content]  
3. [message content]  
...

Your task is to determine, for each message (except the first one), which earlier message it is replying to.  

Output the mapping in this format:  
2 → 1 (short explanation)  
3 → 2 (short explanation)  
4 → None (short explanation)  

Requirements:  
- If a message directly responds to the immediately previous one, map it as current → previous and briefly explain why.  
- If it replies to an earlier message, map it as current → earlier number and explain why.  
- If it does not relate to any previous message, output current → None and explain why.  
- If the relation is unclear, make the most reasonable assumption and explain your reasoning.
- Keep explanations short and concise (one sentence less than 10 words).

Dialogue:
%s
""" % (dialogue)
    response = llm.predict(prompt)
    return response



def re_check(children):
    summary_map = {}
    for child in children:
        summary_map[len(summary_map.keys())] = child

    summary_text = "\n".join([f"{idx}. {summary_map[idx].summary}" for idx in summary_map.keys()])

    prompt = """
I will provide you with a set of content formatted as: "1. xxxxx\n2. xxxxx" (each item is preceded by a number, a dot, and then the content itself).
Please group these items based on the following principles:
- If they are closely related—items with a strong connection, they should be placed in the same group

Items that have no obvious link to others can be grouped alone.

You only need to output the grouping result in the specific format shown below, using the serial numbers of the items. 
For example: [[1], [2,3], [4]]

Contents:
%s
""" % summary_text
    response = llm.predict(prompt)
    print(response)
    ret = json.loads(response)

    if (len(ret) == len(summary_map.keys())):
        return None
    else:
        new_childs = []
        for r in ret:
            if (len(r) == 1):
                new_childs.append(summary_map[r[0]])
            else:
                dialogues = [summary_map[i].dialogue for i in r]
                dialogues_text = "\n".join(dialogues)
                new_sentences = split_sentences(dialogues_text)
                new_c = Tree_node(new_sentences)
                new_childs.append(new_c)
        return new_childs


class Tree_node:
    def __init__(self, sentences, root=False):
        dialogue = "\n".join([f"{s['orig']}" for s in sentences.values()])
        self.dialogue = dialogue
        self.sentences = split_sentences(self.dialogue)
        self.summary = ""
        self.root = root
        self.children = []

    def summarize_direct(self):
        self.summary = summary_dialogue(self.dialogue)
        return self.summary

    def summarize(self):
        relations = get_relation(self.sentences)
        determine_segmentation = determine_topics(self.sentences, relations)
        if (determine_segmentation):
            sub_topics = segment_dialogue(self.sentences, relations)
            independent_tag = determine_dependent(self.sentences, relations, sub_topics)
            if (sub_topics is None):
                self.summary = self.summarize_direct()
            else:
                for key in sub_topics:
                    lines = [self.sentences[idx]["orig"] for idx in sub_topics[key]]
                    dialogue = "\n".join(lines)
                    child_sentences = split_sentences(dialogue)
                    child = Tree_node(child_sentences, independent_tag)
                    self.children.append(child)
                child_summaries = []
                for child in self.children:
                    child_summary = child.summarize()
                    child_summaries.append(child_summary)
                print(child_summaries)

                checked_childs = re_check(self.children)
                if (checked_childs is None):
                    self.summary = merge_subsumaries(child_summaries, independent_tag)
                else:
                    child_summaries = []
                    for child in checked_childs:
                        if (child.summary != ""):
                            child_summaries.append(child.summary)
                        else:
                            child_summaries.append(child.summarize_direct())
                    self.summary = merge_subsumaries(child_summaries, independent_tag)
        else:
            self.summary = self.summarize_direct()
        return self.summary


def split_sentences(dialogue):
    sentences = dialogue.split("\n")
    new_sentences = {}
    for s in sentences:
        idx = len(new_sentences.keys())
        new_sentences[idx] = {"orig": s, "index": f"[{idx}] {s}"}

    return new_sentences


def process(line):
    new_sentences = split_sentences(line["dialogue"])
    root = Tree_node(new_sentences, True)
    summary = root.summarize()
    line["response"] = summary
    return line


if __name__ == "__main__":
    with open("data/datas.jsonl", "r") as f:
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

    with open("result/DIGEST.json", "w+") as f:
        json.dump(results, f, indent=4)














