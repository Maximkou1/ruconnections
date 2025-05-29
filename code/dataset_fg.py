import os
import random
import csv
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

DATA_DIR = 'datasets'
CATEGORY_SIZE = 4


def load_datasets_with_subtypes() -> Tuple[
    Dict[str, Dict[str, Dict[str, Set[str]]]],
    Dict[str, Dict[str, Dict[str, Set[str]]]]
]:
    """
    Loads datasets from the DATA_DIR, organizing them by main type, subtype,
    category, and word.
    """
    data_by_category = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    data_by_word = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return {}, {}

    for main_type in os.listdir(DATA_DIR):
        main_type_path = os.path.join(DATA_DIR, main_type)
        if not os.path.isdir(main_type_path):
            continue

        for subtype in os.listdir(main_type_path):
            subtype_path = os.path.join(main_type_path, subtype)
            if not os.path.isdir(subtype_path):
                continue

            for fname in os.listdir(subtype_path):
                if fname.endswith('.csv'):
                    file_path = os.path.join(subtype_path, fname)
                    try:
                        with open(file_path, encoding='utf-8') as f:
                            reader = csv.reader(f, delimiter=';')
                            for row in reader:
                                if len(row) == 2:
                                    cat, word = row[0].strip(), row[1].strip()
                                    if not cat or not word:
                                        continue
                                    data_by_category[main_type][subtype][cat].add(word)
                                    data_by_word[main_type][subtype][word].add(cat)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

    # Convert to regular dicts to prevent defaultdict behavior on missing keys later
    final_data_by_category = {
        mt: {st: dict(cats) for st, cats in sub_data.items()}
        for mt, sub_data in data_by_category.items()
    }
    final_data_by_word = {
        mt: {st: {w: cats for w, cats in word_data.items()} for st, word_data in sub_data.items()}
        for mt, sub_data in data_by_word.items()
    }
    return final_data_by_category, final_data_by_word


def pick_random_category(
        all_data_by_category: Dict[str, Dict[str, Dict[str, Set[str]]]],
        used_words: Set[str]
) -> Optional[Tuple[str, str, str, List[str]]]:  # main_type, subtype, category_name, sampled_words
    """
    Selects a random category with a subtype and 4 random words from it,
    ensuring words are not in used_words.
    """
    eligible_categories = []
    for main_type, subtypes in all_data_by_category.items():
        for subtype, categories in subtypes.items():
            for category_name, words_in_cat in categories.items():
                available_words = list(words_in_cat - used_words)
                if len(available_words) >= CATEGORY_SIZE:
                    # Store all words of the category to sample from later
                    eligible_categories.append((main_type, subtype, category_name, list(words_in_cat)))

    if not eligible_categories:
        return None

    chosen_main_type, chosen_subtype, chosen_category_name, all_words_of_category = random.choice(eligible_categories)

    # Sample from the chosen category's words, excluding already used_words
    available_for_sampling = list(set(all_words_of_category) - used_words)
    if len(available_for_sampling) < CATEGORY_SIZE:  # Should ideally not happen due to earlier check
        return None

    sampled_words = random.sample(available_for_sampling, CATEGORY_SIZE)
    return chosen_main_type, chosen_subtype, chosen_category_name, sampled_words


def get_related_category_containing_word(
        word_to_include: str,
        current_main_type_of_word: str,
        all_data_by_word: Dict[str, Dict[str, Dict[str, Set[str]]]],
        all_data_by_category: Dict[str, Dict[str, Dict[str, Set[str]]]],
        used_words: Set[str]
) -> Optional[Tuple[str, str, str, List[str]]]:  # main_type, subtype, category_name, words_for_category
    """
    Finds a random category in a different main data type that CONTAINS the word_to_include.
    Returns this category with 4 words (word_to_include + 3 other new words not in used_words).
    """
    possible_main_types = [mt for mt in all_data_by_word if mt != current_main_type_of_word]
    if not possible_main_types:
        return None

    target_main_type = random.choice(possible_main_types)

    if target_main_type in all_data_by_word:
        # Shuffle subtypes to introduce more randomness in selection
        subtypes_to_search = list(all_data_by_word[target_main_type].keys())
        random.shuffle(subtypes_to_search)

        for subtype in subtypes_to_search:
            if word_to_include in all_data_by_word[target_main_type][subtype]:
                candidate_categories = list(all_data_by_word[target_main_type][subtype][word_to_include])
                random.shuffle(candidate_categories)

                for category_name in candidate_categories:
                    if target_main_type in all_data_by_category and \
                            subtype in all_data_by_category[target_main_type] and \
                            category_name in all_data_by_category[target_main_type][subtype]:

                        all_words_in_found_category = all_data_by_category[target_main_type][subtype][category_name]

                        # Ensure the word_to_include is actually in the definitive list of words for this category
                        if word_to_include not in all_words_in_found_category:
                            continue

                        # Find 3 other words, not including word_to_include and not in global used_words
                        available_other_words = list(all_words_in_found_category - {word_to_include} - used_words)

                        if len(available_other_words) >= CATEGORY_SIZE - 1:
                            other_new_words = random.sample(available_other_words, CATEGORY_SIZE - 1)
                            return target_main_type, subtype, category_name, [word_to_include] + other_new_words
    return None


def generate_false_group(
        data_by_category: Dict[str, Dict[str, Dict[str, Set[str]]]],
        data_by_word: Dict[str, Dict[str, Dict[str, Set[str]]]],
        max_attempts_initial_category: int = 500
) -> Optional[Tuple[Tuple[str, List[str]], List[Tuple[str, List[str]]]]]:
    """
    Generates a "false group" puzzle.
    It starts with an initial category of 4 words. For each of these 4 words,
    it finds a new, distinct category of a *different main type* that includes
    that specific word plus 3 other new words.
    Returns: ((initial_cat_name, [initial_words]), [(related_cat_name_i, [related_words_i])])
    Returns None if a full puzzle cannot be generated.
    """
    attempts_for_new_initial = 0
    while attempts_for_new_initial < max_attempts_initial_category:
        attempts_for_new_initial += 1

        # These sets are for the current attempt to build one full puzzle
        current_puzzle_used_words = set()
        current_puzzle_used_category_names = set()

        # Step 1: Pick an initial random category and its 4 words
        initial_category_data = pick_random_category(data_by_category, current_puzzle_used_words)
        if not initial_category_data:
            continue  # try picking another initial category

        initial_main_type, _initial_subtype, initial_category_name, initial_words = initial_category_data

        initial_category_tuple = (initial_category_name, initial_words)
        current_puzzle_used_words.update(initial_words)
        current_puzzle_used_category_names.add(initial_category_name)

        related_categories_list = []
        possible_to_generate_all_related = True

        # Step 2: For each word in the initial category, find a related category
        for word_from_initial in initial_words:
            related_category_data = get_related_category_containing_word(
                word_from_initial,
                initial_main_type,  # Main type of the category the word_from_initial belongs to
                data_by_word,
                data_by_category,
                current_puzzle_used_words  # Words already used in this puzzle attempt
            )

            if related_category_data:
                _related_main_type, _related_subtype, related_category_name, related_words = related_category_data

                if related_category_name in current_puzzle_used_category_names:
                    possible_to_generate_all_related = False  # Category name collision
                    break

                related_categories_list.append((related_category_name, related_words))
                current_puzzle_used_category_names.add(related_category_name)
                current_puzzle_used_words.update(related_words)  # Add words from this new category
            else:
                possible_to_generate_all_related = False  # Couldn't find a related category for this word
                break

        if possible_to_generate_all_related and len(related_categories_list) == CATEGORY_SIZE:
            return initial_category_tuple, related_categories_list  # Successfully generated a full puzzle

    return None  # Failed to generate a puzzle after many attempts


def false_group_pipeline(num_runs: int, output_filename: str):
    if not DATA_BY_CATEGORY_GLOBAL_SUBTYPES or not DATA_BY_WORD_GLOBAL_SUBTYPES:
        print("Cannot run generations: datasets are not loaded or are empty.")
        return

    print(f"\n--- Starting {num_runs} False Group Generations ---")
    print(f"Results will be saved to '{output_filename}'")

    with open(output_filename, 'w', encoding='utf-8') as f:
        successful_runs = 0
        for i in range(num_runs):
            f.write(f"--- Run {i + 1} ---\n")
            generated_data = generate_false_group(
                DATA_BY_CATEGORY_GLOBAL_SUBTYPES,
                DATA_BY_WORD_GLOBAL_SUBTYPES
            )

            if generated_data:
                successful_runs += 1
                initial_category, related_categories = generated_data
                initial_name, initial_words = initial_category
                f.write(f"{initial_name}: {', '.join(initial_words)}\n")
                for j, (rel_name, rel_words) in enumerate(related_categories):
                    f.write(f"{j + 1}. {rel_name}: {', '.join(rel_words)}\n")
            else:
                f.write("No connections were generated for this run (or an error occurred).\n")
            f.write("\n-------------------------------------\n\n")

    print(f"\nFinished {num_runs} False Group runs. Results saved to '{output_filename}'.")
    print(f"Successfully generated full false group puzzles: {successful_runs}/{num_runs} times.")


# Global data loading
DATA_BY_CATEGORY_GLOBAL_SUBTYPES, DATA_BY_WORD_GLOBAL_SUBTYPES = {}, {}

if __name__ == "__main__":
    print("Initializing and loading datasets...")
    DATA_BY_CATEGORY_GLOBAL_SUBTYPES, DATA_BY_WORD_GLOBAL_SUBTYPES = load_datasets_with_subtypes()

    if DATA_BY_CATEGORY_GLOBAL_SUBTYPES and DATA_BY_WORD_GLOBAL_SUBTYPES:
        NUMBER_OF_RUNS = 5
        OUTPUT_FILE = "dataset_fg.txt"
        false_group_pipeline(NUMBER_OF_RUNS, OUTPUT_FILE)
    else:
        print("Datasets could not be loaded. Exiting.")