from openai import OpenAI
import random
import pandas as pd
from navec import Navec
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

MY_KEY = "API_KEY"
client = OpenAI(api_key=MY_KEY)

# upload from https://github.com/natasha/navec
navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
unk = navec['<pad>']

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


# def append_to_csv(step, category, words, path):
#     df = pd.DataFrame([{
#         "id": step,
#         "category": category,
#         "words": ", ".join(words)
#     }])
#     df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


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


def pick_closest_four(words):
    best_group = None
    best_score = -1
    words = list(set(words))
    for combo in combinations(words, 4):
        score = average_similarity(combo)
        if score > best_score:
            best_group = combo
            best_score = score
    return list(best_group)


def gen_initial_group(random_words):
    user_prompt1 = f"""
Пожалуйста создай категорию для головоломки Connections. Сперва напиши короткую историю НА РУССКОМ, опираясь на перевод этих слов: {', '.join(random_words)}.
Затем, используя историю как вдохновение, придумай какую-то тематическую категорию и 8 разных слов, которые подходят под неё.

Формат ответа:
Категория: НАЗВАНИЕ
Слова: СЛОВО1, СЛОВО2, СЛОВО3, СЛОВО4, СЛОВО5, СЛОВО6, СЛОВО7, СЛОВО8
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.8,
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": user_prompt1}
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


def parse_initial_response(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    category_line = None
    words_line = None

    for line in lines:
        if "Категория:" in line:
            category_line = line
        elif "Слова:" in line:
            words_line = line

    if not category_line or not words_line:
        raise ValueError(f"Response does not contain needed lines. Received:\n{text}")

    category = category_line.split("Категория:")[-1].strip()
    words = words_line.split("Слова:")[-1].strip().split(", ")
    return category, words


def parse_overlap_response(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    picked_line = None
    category_line = None
    words_line = None

    for line in lines:
        if "Выбранное слово:" in line:
            picked_line = line
        elif "Категория:" in line:
            category_line = line
        elif "Слова:" in line:
            words_line = line

    if not picked_line or not category_line or not words_line:
        raise ValueError(f"Response does not contain needed lines. Received:\n{text}")

    picked_word = picked_line.split("Выбранное слово:")[-1].strip()
    category = category_line.split("Категория:")[-1].strip()
    words = words_line.split("Слова:")[-1].strip().split(", ")
    return picked_word, category, words


def intentional_overlap_pipeline(word_bank, num_games: int, output_filename: str):
    for cycle in range(num_games):
        print(f"\nGame generation {cycle + 1}...")
        picked_words = []
        game = {}

        run_number = cycle + 1
        try:
            random_words = random.sample(word_bank, 4)
            initial_category_raw = gen_initial_group(random_words)
            initial_category, initial_words = parse_initial_response(initial_category_raw)
            initial_core_group = pick_closest_four(initial_words)
        except Exception as e:
            print(f"Error with generating initial category: {e}")
            continue

        print(f"Category 1: {initial_category} — {initial_core_group}")
        append_to_txt(run_number, 1, initial_category, initial_core_group, output_filename)
        game[initial_category] = initial_core_group

        used_words = set()
        for step in range(2, 5):
            try:
                overlap_raw = gen_overlap_group(picked_words, game)
                picked_word, new_category, new_words = parse_overlap_response(overlap_raw)

                picked_words.append(picked_word)

                new_words = [word for word in new_words if word not in used_words]
                new_core_group = pick_closest_four(new_words)
                used_words.update(new_core_group)

                print(f"Category {step}: {new_category} — {new_core_group}")
                append_to_txt(run_number, step, new_category, new_core_group, output_filename)
                game[new_category] = new_core_group

            except Exception as e:
                print(f"Error on step {step}: {e}")
                continue

    print(f"\nResults saves to '{output_filename}'")


if __name__ == "__main__":
    NUMBER_OF_RUNS = 5
    OUTPUT_FILE = "llm_io.txt"

    df = pd.read_csv("nyt_connections.csv")
    word_list = []
    for words in df['words']:
        word_list.extend([w.strip().lower() for w in words.split(",")])
    word_list = list(set(word_list))
    intentional_overlap_pipeline(word_list, NUMBER_OF_RUNS, OUTPUT_FILE)
