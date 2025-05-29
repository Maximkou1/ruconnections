from openai import OpenAI
import random
from navec import Navec
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import csv
from collections import defaultdict
import re


MY_KEY = "API_KEY"
client = OpenAI(api_key=MY_KEY)

# upload from https://github.com/natasha/navec
navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
unk = navec['<unk>']


INSTRUCTION = """
Представь, что три эксперта создают головоломку Connections на русском языке.
Каждый эксперт записывает своё размышление и делится им с остальными. Затем все эксперты переходят к следующему шагу и так далее.
Если кто-то из экспертов осознаёт, что его ответ не удовлетворяет заданным требованиям, он от него отказывается. Таким образом эксперты приходят к единому мнению.

Задача экспертов - создать четыре группы по четыре слова с чётко выраженной категоризацией. 
Категории должны быть разнообразными, уникальными и хорошо определёнными.

1. Категории в одной игре должны принадлежать к разным типам из списка:
   a) Значение слова - конкретные, чётко определённые группы предметов или ПОНЯТИЙ ОДНОГО РОДА; они могут быть основаны на базовой семантике или на "энциклопедических" знаниях
    - ПОРОДЫ СОБАК: МОПС, ХАСКИ, ПУДЕЛЬ, ТАКСА
    - СИНОНИМЫ СЛОВА «ДОМ»: ХАТА, ЖИЛИЩЕ, ЛАЧУГА, ГНЕЗДО
    - ВЕЩИ КРАСНОГО ЦВЕТА: КЛУБНИКА, МАРС, КРОВЬ, ЧИЛИ
    - ФИЛЬМЫ ТАРКОВСКОГО: СТАЛКЕР, ЗЕРКАЛО, НОСТАЛЬГИЯ, ЖЕРТВОПРИНОШЕНИЕ
    Также слова могут быть связаны более абстрактной ассоциацией, но слова не должны быть слишком разнородными.
   b) Форма слова - слова, встречающиеся в устойчивых сочетаниях с одним общим словом, или имеющие общие структурные особенности
    - ФРАНЦУЗСКИЙ ___: ПОЦЕЛУЙ, БУЛЬДОГ, МАНИКЮР, ЖИМ
    - ЧЁРНЫЙ ___: ПЯТНИЦА, РЫНОК, СПИСОК, ИКРА
    - АНАГРАММЫ: СЕКТА, СЕТКА, ТЕСАК, АСКЕТ
    - СЛОВА С ОДИНАКОВЫМИ ГЛАСНЫМИ: ПОМОР, КАТАМАРАН, БУРУНДУК, ПЕРЕПЕЛ
   c) Сочетание значения и формы слова
    - СЛОВА, НАЧИНАЮЩИЕСЯ НА ЧИСЛИТЕЛЬНОЕ: ОДИНОЧКА, ДВАРФ, ТРИТОН, СЕМЬЯ

2. Словами в категориях должны быть СУЩЕСТВИТЕЛЬНЫЕ РУССКОГО ЯЗЫКА, состоящие из ОДНОГО СЛОВА

3. Каждая категория не должна быть слишком общей и должна иметь КОНКРЕТНОЕ название (не "ЖИВОТНЫЕ", а "ДОМАШНИЕ ЖИВОТНЫЕ" или "ХИЩНЫЕ ЖИВОТНЫЕ")

4. Категории должны быть НЕЗАВИСИМЫМИ друг от друга - игрок должен иметь возможность решить головоломку единственным верным образом.
Не должно быть ситуации, когда два слова, относящиеся к разным категориям, можно поменять между собой, и связь останется верной.

ИЗБЕГАЙ:
- Повторяющихся слов внутри категорий и между категориями (ВСЕ 16 СЛОВ ДОЛЖНЫ БЫТЬ УНИКАЛЬНЫ)
- Слишком общих категорий (например, "ЕДА", "ЦВЕТА", "ЖИВОТНЫЕ")
- Негомогенных слов, относящихся к одной категории (например, "ОБЕЗЬЯНА", "БОНОБО" и "МЛЕКОПИТАЮЩЕЕ" в категории "ЖИВОТНЫЕ")
- Категорий с нечёткими границами
- Объяснений в скобках

ПРИМЕРЫ ПЛОХИХ КАТЕГОРИЙ:
- МУЗЫКАЛЬНЫЙ ФЕСТИВАЛЬ: СЦЕНА, ПУБЛИКА, БИЛЕТ, АРТИСТ
Это плохая категория, потому что относящиеся к ней слова слишком разнородные.
- ПРОФЕССИИ: ОХРАННИК, ПОВАР, ВОДИТЕЛЬ, ВРАЧ
Это плохая категория, потому что она слишком общая.
- СЛОВА, ЗАКАНЧИВАЮЩИЕСЯ НА "Ы": ОГНИ, МОСТЫ, ВЕРШИНЫ, САДЫ
Это плохая категория, потому что не все относящиеся к ней слова ей соответствуют ("ОГНИ" не заканчивается на Ы)
- ЭЛЕМЕНТЫ ГРАФА: ВЕРШИНА, РЁБРО, ГАММИЛЬТОНИАН, КОМПОНЕНТА
Это плохая группа, потому что она содержит несуществующие в русском языке слова: РЁБРО (правильно "РЕБРО") и ГАММИЛЬТОНИАН
- ГЕРОИ РУССКИХ НАРОДНЫХ СКАЗОК: СОЛОВЕЙРАЗБОЙНИК, ИВАНЦАРЕВИЧ, БАБАЯГА, ВАСИЛИСАПРЕКРАСНАЯ
Это плохая категория, потому что она тоже содержит несуществующие слова. Слова "ИВАНЦАРЕВИЧ" не существует, имя героя пишется раздельно в два слова

ПРИ СОЗДАНИИ КАТЕГОРИИ НУЖНО ПРОВЕРИТЬ, ЧТО ОНА:
1. Имеет чёткое, однозначное название
2. Содержит 4 слова, каждое из которых точно соответствует категории
3. Название и слова записаны в верхнем регистре
4. Дополнительно: категория отличается по типу от предыдущих категорий 
"""


def append_to_txt(run_number, step, category, words, path):
    with open(path, "a", encoding="utf-8") as f:
        if step == 1:
            f.write(f"\n--- Run {run_number} ---\n")
        f.write(f"{step}. {category.upper()}: {', '.join(words).upper()}\n")
        if step == 4:
            f.write("\n-------------------------------------\n")


def average_similarity(words):
    embeddings = [navec.get(word.lower(), unk).reshape(1, -1) for word in words]
    sims = [cosine_similarity(a, b)[0][0] for a, b in combinations(embeddings, 2)]
    return sum(sims) / len(sims)


def pick_closest(words, num):
    best_group = None
    best_score = -1
    for combo in combinations(words, num):
        score = average_similarity(combo)
        if score > best_score:
            best_group = combo
            best_score = score
    return list(best_group)


def gen_initial_groups_from_ambiguous(ambiguous_list):
    word_options = []
    for word, senses in ambiguous_list:
        senses_text = "; ".join(senses)
        word_options.append(f"{word.upper()} ({senses_text})")

    words_block = ", ".join(word_options)
    user_prompt = f"""
Ты создаёшь две категории для головоломки Connections на русском языке.

Для вдохновения используй одно из следующих многозначных слов, которое кажется наиболее перспективным (возможные значения указаны в скобках):
{words_block}

Придумай две РАЗНЫЕ категории, в которую могло входить бы выбранное слово, но НЕ ВКЛЮЧАЙ в них само это слово:
1. Категорию, в которую входит это слово — используй одно из его значений.
2. Вторую категорию — вдохновлённую другим значением.

Обе категории должны соответствовать правилам игры Connections:
- По 8 слов в каждой категории.
- Категории не должны пересекаться между собой.
- Только существительные, по одному слову, в верхнем регистре.

До того, как вынести финальный вердикт рассуждения почему та или иная категория лучше.
Твой финальный ответ должен СТРОГО соответствовать формату ниже (без дополнительных строк, разделителей и выделений)!

Формат:
Многозначное слово: СЛОВО
Категория 1: НАЗВАНИЕ
Слова 1: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4, СЛОВО5, СЛОВО6, СЛОВО7, СЛОВО8
Категория 2: НАЗВАНИЕ
Слова 2: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4, СЛОВО5, СЛОВО6, СЛОВО7, СЛОВО8
"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.9,
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


def gen_overlap_group(picked_words, game):
    words_with_categories = []
    for category, words in game.items():
        for word in words:
            if word not in picked_words:  # Исключаем уже выбранные слова
                words_with_categories.append(f"{word} (категория: {category})")

    used_words = [word for words_list in game.values() for word in words_list]
    used_categories = [word for words_list in game.keys() for word in words_list]

    user_prompt2 = f"""
Я создаю головоломку Connections с намеренным пересечением слов внутри категорий.

Вот список слов, которые я уже использовал в предыдущих категориях (в скобках указана категория, в которой слово было использовано): 
{', '.join(words_with_categories)}

Для каждого из этих слов предложи одну или несколько альтернативных категорий. В приоритете использовать полисемичность и омонимию используемых слов. 
Новые категории должны относится использовать ДРУГОЕ значение или особенность формы слова, отличное от категории, в которой оно уже было использовано.
Желательно также использовать разные ТИПЫ категорий (значение слова, форма слова, сочетание формы и значения).
 
Теперь выбери ОДНУ из этих альтернативных категорий, связь между которой и относящимся к ней словом кажется наиболее перспективной.
Хорошо, если новое значение слова в категории отличается от значения, уже описанного в других категориях. В приоритете использовать многозначность слов
Создай категорию с таким названием и аналогично сгенерируй 8 слов, которые подходят под эту категорию

ВАЖНО:
1. Все слова в новой категории должны быть ОДНОГО УРОВНЯ КЛАССИФИКАЦИИ между собой и с исходным словом из предыдущей категории
2. НЕ используй слова из предыдущих категорий: {', '.join(used_words)}
3. Новая категория должна быть уникальной и не повторять уже использованные: {', '.join(used_categories)} — важно, чтобы два слова из разных категорий нельзя было беспрепятственно поменять между ними
4. Помни, что категории должны быть четко различимыми, чтобы каждая головоломка имела только одно возможное корректное решение

До того, как вынести финальный вердикт рассуждения почему та или иная группа лучше.
Твой финальный ответ должен СТРОГО соответствовать формату ниже (без дополнительных строк, разделителей и выделений)!

Формат ответа:
Выбранное слово: СЛОВО
Категория: НАЗВАНИЕ
Слова: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4, СЛОВО5, СЛОВО6, СЛОВО7, СЛОВО8
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": user_prompt2}
        ]
    )
    return response.choices[0].message.content


def parse_double_initial_response(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    ambiguous_word = next(line for line in lines if line.startswith("Многозначное слово:"))
    cat1_line = next(line for line in lines if line.startswith("Категория 1:"))
    words1_line = next(line for line in lines if line.startswith("Слова 1:"))
    cat2_line = next(line for line in lines if line.startswith("Категория 2:"))
    words2_line = next(line for line in lines if line.startswith("Слова 2:"))

    ambiguous_word = ambiguous_word.split("Многозначное слово:")[-1].strip()
    cat1 = cat1_line.split("Категория 1:")[-1].strip()
    words1 = words1_line.split("Слова 1:")[-1].strip().split(", ")
    cat2 = cat2_line.split("Категория 2:")[-1].strip()
    words2 = words2_line.split("Слова 2:")[-1].strip().split(", ")
    return ambiguous_word, cat1, words1, cat2, words2


def parse_overlap_response(text):
    matches = re.findall(
        r"Выбранное слово: (.+?)\s+Категория: (.+?)\s+Слова: (.+?)(?:\n|$)",
        text,
        flags=re.DOTALL
    )
    if not matches:
        raise ValueError("Response does not contain needed lines.")

    # Taking last fitting block
    chosen_word, category, word_line = matches[-1]

    # Parsing category words
    words = [w.strip() for w in word_line.strip().split(",")]

    return chosen_word.strip(), category.strip(), words


def intentional_overlap_pipeline_ambiguous(ambiguous_data, num_games: int, output_filename: str):
    for cycle in range(num_games):
        print(f"\nGame generation {cycle + 1}...")
        picked_words = []
        game = {}
        used_words = set()

        run_number = cycle + 1
        try:
            ambiguous_list = random.sample(list(ambiguous_data.items()), 5)
            response_text = gen_initial_groups_from_ambiguous(ambiguous_list)
            # ambiguous_word, senses = random.choice(list(ambiguous_data.items()))
            # response_text = gen_initial_groups_from_ambiguous(ambiguous_word, senses)
            ambiguous_word, category1, words1, category2, words2 = parse_double_initial_response(response_text)

            words1 = [word for word in words1 if word != ambiguous_word]
            core_group1 = pick_closest(words1, 3)
            core_group1.append(ambiguous_word)
            used_words.update(core_group1)

            words2 = [word for word in words2 if word not in used_words]
            core_group2 = pick_closest(words2, 4)

            print(f"Category 1: {category1} — {core_group1}")
            append_to_txt(run_number, 1, category1, core_group1)
            game[category1] = core_group1

            print(f"Category 2: {category2} — {core_group2}")
            append_to_txt(run_number, 2, category2, core_group2)
            game[category2] = core_group2

            for step in range(3, 5):
                try:
                    overlap_raw = gen_overlap_group(picked_words, game)
                    picked_word, new_category, new_words = parse_overlap_response(overlap_raw)

                    picked_words.append(picked_word)

                    new_words = [word for word in new_words if word not in used_words]
                    new_core_group = pick_closest(new_words, 4)
                    used_words.update(new_core_group)

                    print(f"Category {step}: {new_category} — {new_core_group}")
                    append_to_txt(run_number, step, new_category, new_core_group)
                    game[new_category] = new_core_group
                except Exception as e:
                    print(f"Error on step {step}: {e}")
                    continue

        except Exception as e:
            print(f"Error with initial categories generation: {e}")
            continue

    print(f"\nResults saved to '{output_filename}'")


result = defaultdict(list)
with open("ambiguous.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        word = row["word"].strip()
        hypernym = row["hypernym"].strip()
        result[word].append(hypernym)
ambiguous = dict(result)


if __name__ == "__main__":
    NUMBER_OF_RUNS = 5
    OUTPUT_FILE = "llm_io_ds.txt"

    intentional_overlap_pipeline_ambiguous(ambiguous, NUMBER_OF_RUNS, OUTPUT_FILE)
