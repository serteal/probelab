"""Sparse Probing Datasets (113 binary classification tasks).

From: "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (arXiv:2502.16681)
Source: https://github.com/EleutherAI/sae-probes

These datasets cover diverse domains including NLP benchmarks (GLUE), knowledge facts,
reasoning, content moderation, sentiment analysis, medical/science, and code classification.

Usage:
    import probelab as pl

    # Load single dataset
    ds = pl.datasets.load("sparse_probing_87_glue_cola")

    # Load all datasets
    all_datasets = load_sparse_probing_all()

    # Load by category
    glue_datasets = load_sparse_probing(category="glue")
"""

from datasets import load_dataset as hf_load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset

HF_REPO = "serteal/sparse-probing"

# Dataset manifest: (id, filename, category, description)
# Categories: knowledge, glue, reasoning, moderation, sentiment, science, code, other
SPARSE_PROBING_DATASETS: list[tuple[int, str, str, str]] = [
    # Historical Figures (Knowledge)
    (2, "hist_fig_birthyear", "knowledge", "Birth year of historical figures"),
    (3, "hist_fig_deathyear", "knowledge", "Death year of historical figures"),
    (4, "hist_fig_age", "knowledge", "Age of historical figures"),
    (5, "hist_fig_ismale", "knowledge", "Gender classification of historical figures"),
    (6, "hist_fig_isamerican", "knowledge", "American nationality of historical figures"),
    (7, "hist_fig_ispolitician", "knowledge", "Politician occupation of historical figures"),
    # NYC Geographic (Knowledge)
    (8, "nyc_lat", "knowledge", "NYC latitude classification"),
    (9, "nyc_long", "knowledge", "NYC longitude classification"),
    (10, "nyc_borough", "knowledge", "NYC borough classification"),
    # US Geographic (Knowledge)
    (11, "us_lat", "knowledge", "US latitude classification"),
    (12, "us_long", "knowledge", "US longitude classification"),
    (13, "us_state", "knowledge", "US state classification"),
    (14, "us_timezone", "knowledge", "US timezone classification"),
    (15, "us_population", "knowledge", "US population classification"),
    (16, "us_density", "knowledge", "US density classification"),
    # World Geographic (Knowledge)
    (17, "world_country", "knowledge", "World country classification"),
    (18, "world_latitude", "knowledge", "World latitude classification"),
    (19, "world_longitude", "knowledge", "World longitude classification"),
    (20, "world_pageviews", "knowledge", "World entity pageviews classification"),
    # Headlines (Knowledge)
    (21, "headline_istrump", "knowledge", "Headlines mentioning Trump"),
    (22, "headline_isobama", "knowledge", "Headlines mentioning Obama"),
    (23, "headline_ischina", "knowledge", "Headlines mentioning China"),
    (24, "headline_isiran", "knowledge", "Headlines mentioning Iran"),
    (25, "headline_year", "knowledge", "Headline year classification"),
    (26, "headline_isfrontpage", "knowledge", "Front page headline classification"),
    # Art (Knowledge)
    (27, "art_type", "knowledge", "Art type classification (book/song/movie)"),
    (28, "art_year", "knowledge", "Art year classification"),
    (29, "art_pageviews", "knowledge", "Art pageviews classification"),
    (30, "book_length", "knowledge", "Book length classification"),
    # Science/Reasoning
    (36, "sciq_tf", "reasoning", "Science questions true/false"),
    (37, "arith_mc", "reasoning", "Arithmetic multiple choice"),
    (38, "arith_difficulty", "reasoning", "Arithmetic difficulty classification"),
    (39, "arith_infix", "reasoning", "Arithmetic infix notation"),
    (40, "arith_rpn", "reasoning", "Arithmetic reverse polish notation"),
    (41, "truthqa_tf", "reasoning", "TruthfulQA true/false"),
    (42, "temp_sense", "reasoning", "Temporal sense classification"),
    (43, "temp_cat", "reasoning", "Temporal category classification"),
    (44, "phys_tf", "reasoning", "Physical reasoning true/false"),
    (45, "context_ans", "reasoning", "Contextual answer classification"),
    (46, "context_type", "reasoning", "Context type classification"),
    (47, "reasoning_tf", "reasoning", "Commonsense reasoning true/false"),
    # Ethics (Reasoning)
    (48, "cm_correct", "reasoning", "Common morality correctness"),
    (49, "cm_isshort", "reasoning", "Common morality statement brevity"),
    (50, "deon_isvalid", "reasoning", "Deontological validity classification"),
    (51, "just_is", "reasoning", "Justice classification"),
    (52, "virtue_is", "reasoning", "Virtue ethics classification"),
    (54, "cs_tf", "reasoning", "Commonsense true/false"),
    (55, "open_qa", "reasoning", "Open-domain QA classification"),
    # Wikidata (Knowledge)
    (56, "wikidatasex_or_gender", "knowledge", "Wikidata gender classification"),
    (57, "wikidatais_alive", "knowledge", "Wikidata alive status"),
    (58, "wikidatapolitical_party", "knowledge", "Wikidata political party"),
    (59, "wikidata_occupation_isjournalist", "knowledge", "Wikidata journalist occupation"),
    (60, "wikidata_occupation_isathlete", "knowledge", "Wikidata athlete occupation"),
    (61, "wikidata_occupation_isactor", "knowledge", "Wikidata actor occupation"),
    (62, "wikidata_occupation_ispolitician", "knowledge", "Wikidata politician occupation"),
    (63, "wikidata_occupation_issinger", "knowledge", "Wikidata singer occupation"),
    (64, "wikidata_occupation_isresearcher", "knowledge", "Wikidata researcher occupation"),
    # Domain Concepts (Science)
    (65, "high-school", "science", "High school concept recognition"),
    (66, "living-room", "science", "Living room concept recognition"),
    (67, "social-security", "science", "Social security concept recognition"),
    (68, "credit-card", "science", "Credit card concept recognition"),
    (69, "blood-pressure", "science", "Blood pressure concept recognition"),
    (70, "prime-factors", "science", "Prime factors concept recognition"),
    (71, "social-media", "science", "Social media concept recognition"),
    (72, "gene-expression", "science", "Gene expression concept recognition"),
    (73, "control-group", "science", "Control group concept recognition"),
    (74, "magnetic-field", "science", "Magnetic field concept recognition"),
    (75, "cell-lines", "science", "Cell lines concept recognition"),
    (76, "trial-court", "science", "Trial court concept recognition"),
    (77, "second-derivative", "science", "Second derivative concept recognition"),
    (78, "north-america", "science", "North America concept recognition"),
    (79, "human-rights", "science", "Human rights concept recognition"),
    (80, "side-effects", "science", "Side effects concept recognition"),
    (81, "public-health", "science", "Public health concept recognition"),
    (82, "federal-government", "science", "Federal government concept recognition"),
    (83, "third-party", "science", "Third party concept recognition"),
    (84, "clinical-trials", "science", "Clinical trials concept recognition"),
    (85, "mental-health", "science", "Mental health concept recognition"),
    # Social IQA (Reasoning)
    (86, "social_iqa", "reasoning", "Social intelligence QA"),
    # GLUE Benchmarks
    (87, "glue_cola", "glue", "GLUE CoLA linguistic acceptability"),
    (88, "glue_mnli", "glue", "GLUE MNLI natural language inference"),
    (89, "glue_mrpc", "glue", "GLUE MRPC paraphrase detection"),
    (90, "glue_qnli", "glue", "GLUE QNLI question answering NLI"),
    (91, "glue_qqp", "glue", "GLUE QQP Quora question pairs"),
    (92, "glue_sst2", "glue", "GLUE SST2 sentiment analysis"),
    (93, "glue_stsb", "glue", "GLUE STSB semantic similarity"),
    # Content Moderation
    (94, "ai_gen", "moderation", "AI-generated text detection"),
    (95, "toxic_is", "moderation", "Toxicity detection"),
    (96, "spam_is", "moderation", "Spam detection"),
    (97, "news_class", "moderation", "News category classification"),
    (98, "cancer_cat", "science", "Cancer category classification"),
    (99, "sci_cat", "science", "Science category classification"),
    (100, "news_fake", "moderation", "Fake news detection"),
    (101, "disease_class", "science", "Disease classification"),
    # Sentiment/Emotion
    (102, "twt_emotion", "sentiment", "Tweet emotion classification"),
    (103, "it_tick", "other", "IT ticket classification"),
    (105, "click_bait", "moderation", "Clickbait detection"),
    (106, "hate_hate", "moderation", "Hate speech detection"),
    (107, "hate_offensive", "moderation", "Offensive language detection"),
    (108, "athlete_sport", "other", "Athlete sport classification"),
    (109, "ball_ws", "other", "Basketball win shares classification"),
    (110, "aimade_humangpt3", "moderation", "Human vs GPT-3 text detection"),
    (111, "yelp_data", "sentiment", "Yelp review sentiment"),
    (112, "amzn_rev", "sentiment", "Amazon review sentiment"),
    (113, "movie_sent", "sentiment", "Movie review sentiment"),
    # Geographic Subcategories (Knowledge)
    (114, "nyc_borough_Manhattan", "knowledge", "NYC Manhattan borough"),
    (115, "nyc_borough_Brooklyn", "knowledge", "NYC Brooklyn borough"),
    (116, "nyc_borough_Bronx", "knowledge", "NYC Bronx borough"),
    (117, "us_state_FL", "knowledge", "US Florida state"),
    (118, "us_state_CA", "knowledge", "US California state"),
    (119, "us_state_TX", "knowledge", "US Texas state"),
    (120, "us_timezone_Chicago", "knowledge", "US Chicago timezone"),
    (121, "us_timezone_New_York", "knowledge", "US New York timezone"),
    (122, "us_timezone_Los_Angeles", "knowledge", "US Los Angeles timezone"),
    (123, "world_country_United_Kingdom", "knowledge", "World United Kingdom"),
    (124, "world_country_United_States", "knowledge", "World United States"),
    (125, "world_country_Italy", "knowledge", "World Italy"),
    (126, "art_type_book", "knowledge", "Art type book classification"),
    (127, "art_type_song", "knowledge", "Art type song classification"),
    (128, "art_type_movie", "knowledge", "Art type movie classification"),
    # Reasoning Subcategories
    (129, "arith_mc_A", "reasoning", "Arithmetic multiple choice A"),
    (130, "temp_cat_Frequency", "reasoning", "Temporal category frequency"),
    (131, "temp_cat_Typical Time", "reasoning", "Temporal category typical time"),
    (132, "temp_cat_Event Ordering", "reasoning", "Temporal category event ordering"),
    (133, "context_type_Causality", "reasoning", "Context type causality"),
    (134, "context_type_Belief_states", "reasoning", "Context type belief states"),
    (135, "context_type_Event_duration", "reasoning", "Context type event duration"),
    # GLUE MNLI Subcategories
    (136, "glue_mnli_entailment", "glue", "GLUE MNLI entailment"),
    (137, "glue_mnli_neutral", "glue", "GLUE MNLI neutral"),
    (138, "glue_mnli_contradiction", "glue", "GLUE MNLI contradiction"),
    # News Subcategories
    (139, "news_class_Politics", "moderation", "News politics classification"),
    (140, "news_class_Technology", "moderation", "News technology classification"),
    (141, "news_class_Entertainment", "moderation", "News entertainment classification"),
    # Medical Subcategories
    (142, "cancer_cat_Thyroid_Cancer", "science", "Thyroid cancer classification"),
    (143, "cancer_cat_Lung_Cancer", "science", "Lung cancer classification"),
    (144, "cancer_cat_Colon_Cancer", "science", "Colon cancer classification"),
    (145, "disease_class_digestive system diseases", "science", "Digestive system diseases"),
    (146, "disease_class_cardiovascular diseases", "science", "Cardiovascular diseases"),
    (147, "disease_class_nervous system diseases", "science", "Nervous system diseases"),
    # Emotion Subcategories
    (148, "twt_emotion_worry", "sentiment", "Tweet emotion worry"),
    (149, "twt_emotion_happiness", "sentiment", "Tweet emotion happiness"),
    (150, "twt_emotion_sadness", "sentiment", "Tweet emotion sadness"),
    # IT/Sports Subcategories
    (151, "it_tick_HR Support", "other", "IT ticket HR support"),
    (152, "it_tick_Hardware", "other", "IT ticket hardware"),
    (153, "it_tick_Administrative rights", "other", "IT ticket administrative rights"),
    (154, "athlete_sport_football", "other", "Athlete football classification"),
    (155, "athlete_sport_basketball", "other", "Athlete basketball classification"),
    (156, "athlete_sport_baseball", "other", "Athlete baseball classification"),
    # Reviews/Code
    (157, "amazon_5star", "sentiment", "Amazon 5-star rating classification"),
    (158, "code_C", "code", "C code detection"),
    (159, "code_Python", "code", "Python code detection"),
    (160, "code_HTML", "code", "HTML code detection"),
    # AG News
    (161, "agnews_0", "moderation", "AG News category 0"),
    (162, "agnews_1", "moderation", "AG News category 1"),
    (163, "agnews_2", "moderation", "AG News category 2"),
]

# Keep only the binary-label subset from serteal/sparse-probing.
# These 42 dataset IDs have non-binary labels (multiclass/regression-style),
# so they are excluded from the binary sparse-probing registry.
NON_BINARY_DATASET_IDS: set[int] = {
    2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    25, 27, 28, 29, 30, 37, 38, 39, 40, 43, 45, 46, 55, 86, 88, 93,
    97, 98, 99, 101, 102, 103, 108, 109, 111, 112,
}

SPARSE_PROBING_DATASETS = [
    row for row in SPARSE_PROBING_DATASETS
    if row[0] not in NON_BINARY_DATASET_IDS
]


def _load_from_hf(config_name: str, name: str) -> Dataset:
    """Load a sparse probing dataset from HuggingFace."""
    hf_ds = hf_load_dataset(HF_REPO, config_name, split="train")

    dialogues = [[Message(role="user", content=str(row["prompt"]))] for row in hf_ds]
    labels = [Label.POSITIVE if row["label"] == 1 else Label.NEGATIVE for row in hf_ds]

    return Dataset(dialogues=dialogues, labels=labels, name=name).shuffle()


def _make_loader(config_name: str, dataset_name: str):
    """Create a loader function for a dataset."""

    def loader() -> Dataset:
        return _load_from_hf(config_name, dataset_name)

    return loader


# Register all datasets
for id_, name, _category, description in SPARSE_PROBING_DATASETS:
    dataset_name = f"sparse_probing_{id_}_{name}"
    config_name = f"{id_}_{name}"

    loader_fn = _make_loader(config_name, dataset_name)
    _register_dataset(dataset_name, Topic.SPARSE_PROBING, description)(loader_fn)


def load_sparse_probing_all() -> dict[str, Dataset]:
    """Load all sparse probing datasets.

    Returns:
        Dictionary mapping dataset names to Dataset objects.

    Example:
        >>> all_datasets = load_sparse_probing_all()
        >>> print(f"Loaded {len(all_datasets)} datasets")
    """
    from . import load

    return {
        f"sparse_probing_{id_}_{name}": load(f"sparse_probing_{id_}_{name}")
        for id_, name, _, _ in SPARSE_PROBING_DATASETS
    }


def load_sparse_probing(category: str | None = None) -> dict[str, Dataset]:
    """Load sparse probing datasets, optionally filtered by category.

    Args:
        category: Optional category filter. One of:
            - "knowledge": Historical figures, geography, wikidata
            - "glue": GLUE benchmark tasks
            - "reasoning": Science, ethics, commonsense
            - "moderation": Toxicity, spam, fake news
            - "sentiment": Emotion and review sentiment
            - "science": Domain concepts, medical
            - "code": Programming language detection
            - "other": IT tickets, sports

    Returns:
        Dictionary mapping dataset names to Dataset objects.

    Example:
        >>> glue_datasets = load_sparse_probing(category="glue")
        >>> moderation_datasets = load_sparse_probing(category="moderation")
    """
    from . import load

    datasets = SPARSE_PROBING_DATASETS
    if category:
        datasets = [(i, n, c, d) for i, n, c, d in datasets if c == category]
    return {
        f"sparse_probing_{id_}_{name}": load(f"sparse_probing_{id_}_{name}")
        for id_, name, _, _ in datasets
    }


def list_sparse_probing_categories() -> list[str]:
    """List available sparse probing categories.

    Returns:
        Sorted list of category names.
    """
    return sorted({c for _, _, c, _ in SPARSE_PROBING_DATASETS})
