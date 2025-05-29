import os
import random
import csv
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

DATA_DIR = 'datasets'
CATEGORY_SIZE = 4

FORM_SUBTYPE_WEIGHTS = {
    'collocations': 4,
    'anagrams': 1
}
DEFAULT_SUBTYPE_WEIGHT = 1


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
        main_type_to_pick: str,
        all_data_by_category: Dict[str, Dict[str, Dict[str, Set[str]]]],
        used_words: Set[str],
        used_categories: Set[str]
) -> Optional[Tuple[str, str, str, List[str]]]:  # main_type, subtype, category_name, words
    """
    Selects a random category of the specified main_type.
    For 'form', considers subtype weights. Returns None if no suitable category is found.
    """
    if main_type_to_pick not in all_data_by_category:
        return None

    subtypes_data = all_data_by_category[main_type_to_pick]
    eligible_categories_with_weights = []  # Stores (subtype, cat_name, available_words, weight)

    for subtype_name, categories_in_subtype in subtypes_data.items():
        weight = DEFAULT_SUBTYPE_WEIGHT
        if main_type_to_pick == 'form':
            weight = FORM_SUBTYPE_WEIGHTS.get(subtype_name, DEFAULT_SUBTYPE_WEIGHT)

        if weight <= 0:
            continue

        for cat_name, words_in_cat in categories_in_subtype.items():
            if cat_name in used_categories:
                continue

            available_words = list(words_in_cat - used_words)
            if len(available_words) >= CATEGORY_SIZE:
                eligible_categories_with_weights.append(
                    (subtype_name, cat_name, available_words, weight)
                )

    if not eligible_categories_with_weights:
        return None

    subtypes_chosen_list = [item[0] for item in eligible_categories_with_weights]
    cat_names_chosen_list = [item[1] for item in eligible_categories_with_weights]
    available_words_lists = [item[2] for item in eligible_categories_with_weights]
    weights_list = [item[3] for item in eligible_categories_with_weights]

    try:
        chosen_index = random.choices(range(len(eligible_categories_with_weights)), weights=weights_list, k=1)[0]
    except ValueError:  # Fallback if all weights are 0 or list is empty (though checked)
        if not eligible_categories_with_weights: return None  # Should not happen due to check above
        chosen_index = random.choice(range(len(eligible_categories_with_weights)))  # Uniform choice

    chosen_subtype = subtypes_chosen_list[chosen_index]
    chosen_category_name = cat_names_chosen_list[chosen_index]
    words_for_sampling = available_words_lists[chosen_index]

    sampled_words = random.sample(words_for_sampling, CATEGORY_SIZE)
    return main_type_to_pick, chosen_subtype, chosen_category_name, sampled_words


def get_new_category_by_word(
        word_to_connect: str,
        all_data_by_word: Dict[str, Dict[str, Dict[str, Set[str]]]],
        all_data_by_category: Dict[str, Dict[str, Dict[str, Set[str]]]],
        target_main_type: str,
        used_categories: Set[str],
        used_words: Set[str],
) -> Optional[Tuple[str, str, str, List[str]]]:
    """
    Finds a new category of target_main_type containing word_to_connect.
    For 'form', subtype selection is weighted. Returns 4 words not in used_words.
    """
    if target_main_type not in all_data_by_word:
        return None

    subtypes_word_data = all_data_by_word[target_main_type]
    eligible_subtypes_with_weights = []

    for subtype_name, word_to_cats in subtypes_word_data.items():
        if word_to_connect in word_to_cats:
            weight = DEFAULT_SUBTYPE_WEIGHT
            if target_main_type == 'form':
                weight = FORM_SUBTYPE_WEIGHTS.get(subtype_name, DEFAULT_SUBTYPE_WEIGHT)
            if weight > 0:
                eligible_subtypes_with_weights.append((subtype_name, weight))

    if not eligible_subtypes_with_weights:
        return None

    subtypes_list = [item[0] for item in eligible_subtypes_with_weights]
    weights_list = [item[1] for item in eligible_subtypes_with_weights]

    try:
        chosen_subtype = random.choices(subtypes_list, weights=weights_list, k=1)[0]
    except ValueError:  # Fallback if list is empty or weights are problematic
        if not subtypes_list: return None
        chosen_subtype = random.choice(subtypes_list)

    if chosen_subtype in subtypes_word_data and word_to_connect in subtypes_word_data[chosen_subtype]:
        candidate_categories = list(subtypes_word_data[chosen_subtype][word_to_connect])
        random.shuffle(candidate_categories)

        for cat_name in candidate_categories:
            if cat_name in used_categories:
                continue

            if target_main_type in all_data_by_category and \
                    chosen_subtype in all_data_by_category[target_main_type] and \
                    cat_name in all_data_by_category[target_main_type][chosen_subtype]:

                category_all_words = all_data_by_category[target_main_type][chosen_subtype][cat_name]
                available_new_words = list(category_all_words - used_words)

                # new category should contain word_to_connect
                potential_words_for_category = set(available_new_words)
                if word_to_connect in category_all_words and word_to_connect not in used_words:
                    potential_words_for_category.add(word_to_connect)

                if len(available_new_words) >= CATEGORY_SIZE:
                    sampled_words = random.sample(available_new_words, CATEGORY_SIZE)
                    return target_main_type, chosen_subtype, cat_name, sampled_words

    return None


def generate_intentional_overlap(
        data_by_category: Dict[str, Dict[str, Dict[str, Set[str]]]],
        data_by_word: Dict[str, Dict[str, Dict[str, Set[str]]]]
) -> List[Tuple[str, List[str]]]:
    used_words = set()
    used_categories = set()
    result_categories_details = []

    # First category is always 'meaning'
    current_main_type = 'meaning'
    first_cat_data = pick_random_category(current_main_type, data_by_category, used_words, used_categories)
    if not first_cat_data:
        return []

    result_categories_details.append(first_cat_data)
    used_words.update(first_cat_data[3])
    used_categories.add(first_cat_data[2])

    # Determine the type for the next category (alternating)
    next_target_main_type = 'form' if current_main_type == 'meaning' else 'meaning'

    while len(result_categories_details) < 4:
        found_category_for_this_step = False
        category_data_for_this_step = None
        actual_main_type_chosen_this_step = None

        # Define search order: 1. Overlap with target_type, 2. Overlap with other_type,
        # 3. Random with target_type, 4. Random with other_type.
        primary_search_type = next_target_main_type
        secondary_search_type = 'meaning' if primary_search_type == 'form' else 'form'

        shuffled_used_words = list(used_words)
        random.shuffle(shuffled_used_words)

        # Attempt 1: Find overlap with the primary_search_type
        for word_conn in shuffled_used_words:
            category_data_for_this_step = get_new_category_by_word(
                word_conn, data_by_word, data_by_category,
                primary_search_type, used_categories, used_words
            )
            if category_data_for_this_step:
                actual_main_type_chosen_this_step = primary_search_type
                found_category_for_this_step = True
                break

        # Attempt 2: If no overlap with primary, try overlap with secondary_search_type
        if not found_category_for_this_step:
            for word_conn in shuffled_used_words:
                category_data_for_this_step = get_new_category_by_word(
                    word_conn, data_by_word, data_by_category,
                    secondary_search_type, used_categories, used_words
                )
                if category_data_for_this_step:
                    actual_main_type_chosen_this_step = secondary_search_type
                    found_category_for_this_step = True
                    break

        # Attempt 3: If still no overlap, pick a random category of primary_search_type
        if not found_category_for_this_step:
            category_data_for_this_step = pick_random_category(
                primary_search_type, data_by_category, used_words, used_categories
            )
            if category_data_for_this_step:
                actual_main_type_chosen_this_step = primary_search_type
                found_category_for_this_step = True

        # Attempt 4: If even that fails, pick a random category of secondary_search_type
        if not found_category_for_this_step:
            category_data_for_this_step = pick_random_category(
                secondary_search_type, data_by_category, used_words, used_categories
            )
            if category_data_for_this_step:
                actual_main_type_chosen_this_step = secondary_search_type
                found_category_for_this_step = True

        if found_category_for_this_step and category_data_for_this_step:
            result_categories_details.append(category_data_for_this_step)
            used_words.update(category_data_for_this_step[3])
            used_categories.add(category_data_for_this_step[2])
            # Next target type alternates based on the type actually chosen for this step
            next_target_main_type = 'meaning' if actual_main_type_chosen_this_step == 'form' else 'form'
        else:
            # Failed to find any category for this step, generation might be incomplete
            break

    return [(details[2], details[3]) for details in result_categories_details]


print("Initializing and loading datasets...")
DATA_BY_CATEGORY_GLOBAL_SUBTYPES, DATA_BY_WORD_GLOBAL_SUBTYPES = load_datasets_with_subtypes()


def intentional_overlap_pipeline(num_runs: int, output_filename: str):
    if not DATA_BY_CATEGORY_GLOBAL_SUBTYPES or not DATA_BY_WORD_GLOBAL_SUBTYPES:
        print("Cannot run generations: datasets are not loaded or are empty.")
        return

    print(f"\n--- Starting {num_runs} Intentional Overlap Generations ---")
    print(f"Results will be saved to '{output_filename}'")

    with open(output_filename, 'w', encoding='utf-8') as f:
        successful_runs = 0
        for i in range(num_runs):
            f.write(f"--- Run {i + 1} ---\n")
            generated_data = generate_intentional_overlap(
                DATA_BY_CATEGORY_GLOBAL_SUBTYPES,
                DATA_BY_WORD_GLOBAL_SUBTYPES
            )

            if generated_data:
                if len(generated_data) == 4:  # Assuming 4 categories per puzzle
                    successful_runs += 1
                for step, (category, words) in enumerate(generated_data):
                    f.write(f"{step + 1}. {category}: {', '.join(words)}\n")
            else:
                f.write("No connections were generated for this run (or an error occurred).\n")
            f.write("\n-------------------------------------\n\n")

    print(f"\nFinished {num_runs} Intentional Overlap runs. Results saved to '{output_filename}'.")
    print(f"Successfully generated full 4-category puzzles: {successful_runs}/{num_runs} times.")


if __name__ == "__main__":
    NUMBER_OF_RUNS = 5
    OUTPUT_FILE = "dataset_io.txt"

    intentional_overlap_pipeline(NUMBER_OF_RUNS, OUTPUT_FILE)