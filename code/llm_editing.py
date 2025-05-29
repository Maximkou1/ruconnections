from openai import OpenAI
import re
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from navec import Navec

MY_KEY = "API_KEY"
client = OpenAI(api_key=MY_KEY)

# upload from https://github.com/natasha/navec
navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
unk = navec['<unk>']

INSTRUCTION = """
Ты — редактор категорий для игры Connections на русском языке. Твоя цель — исправить названия и содержание категорий, если в них содержатся ошибки.
Словами в категориях должны быть СУЩЕСТВИТЕЛЬНЫЕ РУССКОГО ЯЗЫКА, состоящие из ОДНОГО СЛОВА
"""


def parse_initial(text):
    runs = []
    blocks = text.strip().split('--- Run')
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        run_dict = {}
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                line = re.sub(r'^\d+\.\s*', '', line)
                if ': ' in line:
                    cat, words_str = line.split(': ', 1)
                    words = [w.strip() for w in words_str.split(',') if w.strip()]
                    run_dict[cat] = words
        if len(run_dict) == 4:
            runs.append(run_dict)
    return runs


def parse_text_to_dict(text):
    result = {}
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '. ' in line:
            line = line.split('. ', 1)[1]
        if ': ' in line:
            cat, words_str = line.split(': ', 1)
            words = [w.strip() for w in words_str.split(',') if w.strip()]
            result[cat] = words
    return result


def save_dicts_to_file(dicts, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for d in dicts:
            for i, (cat, words) in enumerate(d.items(), 1):
                f.write(f"{i}. {cat.upper()}: {', '.join(words)}\n")
            f.write("\n")


def edit_game(categories):
    formatted = "\n".join([f"{i+1}. {cat}: {', '.join(words)}" for i, (cat, words) in enumerate(categories.items())])
    user_prompt = f"""
Вот четыре категории и относящиеся к ним слова:
{formatted}

Проверь, что все слова - СУЩЕСТВУЮЩИЕ слова русского языка, состоят из одного слова, не содержат грамматических ошибок и в полной мере относятся к категории. Если для какого-то слова это не так, исправь это слово. 
Также, если это необходимо, исправь название категории — оно должно быть кратким, при этом чётко и полно описывая все слова в категории. 
В ответе выдай все четыре категории, заменив старые названия категорий и соответствующих ей слова на новые, если это необходимо. 

Не добавляй никаких пояснений и рассуждений.
Ответ должен содержать ровно четыре строки, начинающиеся с цифр от 1 до 4.

Формат ответа:
1. КАТЕГОРИЯ1: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4
2. КАТЕГОРИЯ2: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4
3. КАТЕГОРИЯ3: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4
4. КАТЕГОРИЯ4: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4
"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def average_similarity(words):
    embeddings = [navec.get(word.lower(), unk).reshape(1, -1) for word in words]
    sims = [cosine_similarity(a, b)[0][0] for a, b in combinations(embeddings, 2)]
    return sum(sims) / len(sims)


def process_runs(runs):
    outputs = []
    for i, run in enumerate(runs, 1):
        print(f"\n--- Run {i} ---")
        scores = {}
        for cat, words in run.items():
            sim = average_similarity(words)
            scores[cat] = sim
            print(f"{cat}: Average similarity = {sim:.4f}")

        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked = [(j + 1, name, run[name]) for j, (name, _) in enumerate(sorted_cats)]
        formatted = [f"{rank}. {name.upper()}: {', '.join(words)}" for rank, name, words in ranked]
        outputs.append("\n".join(formatted))
    return outputs


INPUT_FILE = "llm_io.txt"
OUTPUT_FILE = "llm_io_edited&ranked.txt"

with open(INPUT_FILE, encoding='utf-8') as f:
    text = f.read()

games = parse_initial(text)
new_games = []

for game in games:
    output = edit_game(game)
    parsed_dict = parse_text_to_dict(output)
    new_games.append(parsed_dict)

save_dicts_to_file(new_games, INPUT_FILE.split('.')[0]+"_edited.txt")

processed = process_runs(new_games)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("\n\n".join(processed))
