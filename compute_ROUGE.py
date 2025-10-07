import json

from rouge_score import rouge_scorer



def calculate_rouge_with_library(reference, candidate):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    return scorer.score(reference, candidate)


if __name__ == "__main__":
    approaches = [
        "summary_only_LLM",
        "QA_Sum",
        "Sum_CoT",
        "DIGEST",
        "summary_only_LLM_SAMSUM",
        "QA_Sum_SAMSUM",
        "Sum_CoT_SAMSUM",
        "DIGEST_SAMSUM"
    ]

    for approach in approaches:
        with open(f"result/{approach}.json", "r") as f:
            data = json.load(f)

        # 初始化累积器
        total_scores = {
            'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
            'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
            'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        }

        candidates = []
        references = []
        sample_count = 0
        for line in data:
            if "response" not in line:
                continue

            reference = line["summary"]
            candidate = line["response"]
            candidates.append(candidate)
            references.append(reference)
            sample_count += 1

            scores = calculate_rouge_with_library(reference, candidate)
            for metric in total_scores:
                total_scores[metric]['precision'] += scores[metric].precision
                total_scores[metric]['recall'] += scores[metric].recall
                total_scores[metric]['fmeasure'] += scores[metric].fmeasure


        avg_scores = {}
        for metric in total_scores:
            avg_scores[f"{metric}_precision"] = total_scores[metric]['precision'] / sample_count
            avg_scores[f"{metric}_recall"] = total_scores[metric]['recall'] / sample_count
            avg_scores[f"{metric}_fmeasure"] = total_scores[metric]['fmeasure'] / sample_count

        print(f"================ {approach} ================")
        for metric, score_ in avg_scores.items():
            print(f"{metric}: {score_:.4f}")
            if "fmeasure" in metric:
                print("=" * 50)

        # from bert_score import score
        # P, R, F1 = score(candidates, references, lang="en")
        #
        # avg_P = P.mean()
        # avg_R = R.mean()
        # avg_F1 = F1.mean()
        #
        # print(f"============================={approach}=============================")
        # print(f"Precision: {avg_P.item():.4f}")
        # print(f"Recall: {avg_R.item():.4f}")
        # print(f"F1: {avg_F1.item():.4f}")