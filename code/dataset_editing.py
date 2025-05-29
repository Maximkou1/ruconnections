import re
from navec import Navec
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# upload from https://github.com/natasha/navec
navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
unk = navec['<unk>']


# Calculating average similarity
def average_similarity(words):
    embeddings = [navec.get(word.lower(), unk).reshape(1, -1) for word in words]
    sims = [cosine_similarity(a, b)[0][0] for a, b in combinations(embeddings, 2)]
    return sum(sims) / len(sims)


# File parsing
def parse_runs(file_content):
    runs = re.findall(r'--- Run \d+ ---\n(.*?)\n\n', file_content, re.DOTALL)
    parsed_runs = []

    for run in runs:
        lines = run.strip().split('\n')
        categories = {}
        for line in lines:
            # match = re.match(r'[A-D]\.\s(.+?):\s(.+)', line)
            # match = re.match(r'[0-3]\.\s(.+?):\s(.+)', line)
            match = re.match(r'[1-4]\.\s(.+?):\s(.+)', line)
            if match:
                name, words = match.groups()
                word_list = [w.strip() for w in words.split(',')]
                categories[name] = word_list
        parsed_runs.append(categories)

    return parsed_runs


def process_runs(runs):
    outputs = []

    for i, run in enumerate(runs, 1):
        print(f"\n--- Run {i} ---")

        scores = {}
        for cat, words in run.items():
            sim = average_similarity(words)
            scores[cat] = sim
            print(f"{cat}: average similarity = {sim:.4f}")

        # Sort by descending similarity (easy —> difficult)
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Assigning difficulty levels 1–4
        ranked = [(j + 1, name, run[name]) for j, (name, _) in enumerate(sorted_cats)]

        # Formatting
        formatted = [f"{rank}. {name.upper()}: {', '.join(words)}" for rank, name, words in ranked]
        outputs.append("\n".join(formatted))

    return outputs


def main(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    runs = parse_runs(file_content)
    processed = process_runs(runs)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(processed))


input_file = 'dataset_fg.txt'
output_file = 'dataset_fg_ranked.txt'

main(input_file, output_file)
print(f"Results saved to '{output_file}'")
