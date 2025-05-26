# ruconnections
Pipelines for generation NYT Connections categories in Russian language.

Implemented pipelines:
* Dataset (intentional overlap & false group)
* LLM (intentional overlap & false group)
* Dataset+LLM (intentional overlap)
* Translation

Evaluation results:

![User Rating](https://github.com/Maximkou1/ruconnections/raw/master/images/ruconnections_rating.png)


![User Average Mistake](https://github.com/Maximkou1/ruconnections/raw/master/images/ruconnections_average_mistake.png)


![User Mistake Distribution](https://github.com/Maximkou1/ruconnections/raw/master/images/ruconnections_mistake_distribution.png)

Used datasets:
* RuWordNet — https://ruwordnet.ru/ru
* KartaSlov — https://github.com/dkulagin/kartaslov
* Corpus-based dictionary of government (E. Klyshinsky, A. Bogdanova, M. Kopotev. Towards a Corpus-Based Dictionary of Verbal Government for the Russian)
* Russian Wikitionary — https://dumps.wikimedia.org/ruwiktionary/latest
  
Used LLM: GPT-4.1
