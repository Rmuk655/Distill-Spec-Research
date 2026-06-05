"""
fetch_datasets.py — download eval + train datasets to raw/ as JSONL files.

Usage:
    python fetch_datasets.py                       # fetch all eval sets
    python fetch_datasets.py --train               # also fetch train splits
    python fetch_datasets.py --datasets gsm8k,math500
    python fetch_datasets.py --n 30                # eval prompts per dataset

Eval sets (test splits):
    diverse50, gsm8k, humaneval, math500, mtbench, alpaca

Train splits (for DistillSpec fine-tuning):
    gsm8k_train  — 7,473 prompts from GSM8K train split
    math_train   — 7,500 prompts from MATH competition train split
    alpaca_train — 52,002 prompts (the whole Alpaca dataset is training data)

HumanEval and MTBench have no train splits and are eval-only.

Each output file: one JSON object per line with at least {"prompt": "..."}.
Falls back to hardcoded representative samples if HuggingFace datasets API fails.
"""

import os, sys, json, random, argparse

random.seed(42)
# Save to raw/ — matches experiment.py _data(), all YAML configs, and .gitignore.
DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(DATA_DIR, exist_ok=True)


def save_jsonl(path: str, items: list):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            if isinstance(item, str):
                item = {"prompt": item}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  -> {len(items)} prompts saved to {os.path.basename(path)}")


def _hf_load(dataset_id, split, field, config=None, n=None, trust=False,
             extra_fields=None):
    """Try datasets library first; fall back to huggingface_hub + pandas parquet.

    extra_fields: list of additional field names to include alongside `field`.
    Returns list of strings (if extra_fields is None) or list of dicts.
    """
    # Method 1: datasets library
    try:
        from datasets import load_dataset
        kwargs = {"trust_remote_code": trust}
        ds = load_dataset(dataset_id, config, split=split, **kwargs) if config \
             else load_dataset(dataset_id, split=split, **kwargs)
        results = []
        for row in ds:
            val = row.get(field, "")
            if isinstance(val, list):
                val = val[0] if val else ""
            if not (val and isinstance(val, str)):
                continue
            if extra_fields:
                item = {"prompt": val.strip()}
                for ef in extra_fields:
                    item[ef] = row.get(ef, "")
                results.append(item)
            else:
                results.append(val.strip())
        if n:
            random.shuffle(results)
            results = results[:n]
        return results
    except Exception as e:
        print(f"  [datasets failed: {e}]")

    # Method 2: huggingface_hub download + pandas parquet (bypasses numpy ABI issue)
    try:
        return _hf_parquet_load(dataset_id, split, field, config, n, extra_fields)
    except Exception as e2:
        print(f"  [huggingface_hub fallback failed: {e2}]")
        return None


def _hf_parquet_load(dataset_id, split, field, config=None, n=None, extra_fields=None):
    """Download parquet shard via huggingface_hub and read with pandas."""
    import logging
    import pandas as pd
    from huggingface_hub import hf_hub_download
    # Suppress the noisy "Couldn't access the Hub to check for update but local file
    # already exists" WARNING that fires when HF_HUB_OFFLINE=1 is set.  The fallback
    # to the cached file is exactly what we want — no action needed from the user.
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)

    repo_id = dataset_id
    subfolder = config if config else ""
    candidates = [
        f"data/{split}-00000-of-00001.parquet",
        f"{subfolder}/{split}-00000-of-00001.parquet",
        f"{split}-00000-of-00001.parquet",
        f"data/{split}.parquet",
    ]

    df = None
    for candidate in candidates:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=candidate,
                                   repo_type="dataset", local_dir=DATA_DIR + "/.cache")
            df = pd.read_parquet(path)
            break
        except Exception:
            continue

    if df is None:
        raise RuntimeError(f"Could not find parquet for {dataset_id}")

    if field not in df.columns:
        raise RuntimeError(f"Field '{field}' not in columns: {list(df.columns)}")

    results = []
    for _, row in df.iterrows():
        val = str(row[field]).strip() if row[field] is not None else ""
        if not val:
            continue
        if extra_fields:
            item = {"prompt": val}
            for ef in extra_fields:
                item[ef] = str(row.get(ef, "")) if ef in df.columns else ""
            results.append(item)
        else:
            results.append(val)

    if n:
        random.shuffle(results)
        results = results[:n]
    return results


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------
_GSM8K_FALLBACK = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She has 20 chickens. How many cups of feed does she need to buy to feed her chickens for 1 week?",
    "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
    "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
    "A painter needed to paint 12 rooms in a building. Each room takes 7 hours to paint. If he already painted 5 rooms, how much more time will he need to paint the rest?",
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
    "Shawn has five toys. Christmas his mom and dad gave him two toys each. How many toys does he have now?",
    "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
    "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?",
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "In a school, there are 4 classes. One class has 20 students, another has 22, a third has 25, and the fourth has 28. How many students are there in total?",
    "A store has 50 apples. They sell 18 on Monday, 12 on Tuesday, and restock with 25 on Wednesday. How many apples are there at the end of Wednesday?",
    "Tom reads 15 pages per day. He reads for 7 days. How many pages does he read in total?",
    "A factory produces 120 widgets per hour. How many widgets does it produce in an 8-hour shift?",
    "Sarah earns $15 per hour. She works 6 hours on Saturday and 4 hours on Sunday. How much does she earn over the weekend?",
    "A recipe requires 2.5 cups of flour per batch of cookies. How much flour is needed for 4 batches?",
    "There are 24 students in a class. The teacher splits them into groups of 4. How many groups are there?",
    "A car travels at 60 mph. How far does it travel in 2.5 hours?",
    "John has 3 bags with 8 apples each. He gives away 5 apples. How many apples does he have left?",
    "A movie is 2 hours 15 minutes long. If it started at 3:00 PM, when does it end?",
    "A store sells notebooks for $3 each and pens for $1 each. Maria buys 4 notebooks and 6 pens. How much does she spend?",
    "A swimming pool holds 5000 gallons. It leaks 15 gallons per hour. How many gallons remain after 10 hours?",
    "There are 365 days in a year. How many weeks and days is that?",
    "A box contains 100 chocolates. 40% are dark chocolate, 35% are milk chocolate, and the rest are white. How many white chocolates are there?",
]

def fetch_gsm8k(n=30, force=False):
    """GSM8K test split — VAL POOL (items[0:n], isolated seed=42).

    Used for validation during training: gsm8k_30.jsonl (fast val / early stopping)
    and gsm8k_100.jsonl (slow val / checkpoint selection).  These prompts are
    explicitly EXCLUDED from the eval pool (see fetch_gsm8k_eval).

    Uses an isolated random.Random(42) so results are deterministic regardless of
    the order in which downloader functions are called or how many other random
    operations have occurred since module import.
    """
    path = os.path.join(DATA_DIR, f"gsm8k_{n}.jsonl")
    if os.path.exists(path) and not force:
        print(f"  GSM8K val({n}) already exists, skipping. Use --force to re-download.")
        return path
    print("Fetching GSM8K (val pool)...")
    # Load all 1319 test items, shuffle with isolated RNG, take first n.
    items = _hf_load("gsm8k", "test", "question", config="main", n=None,
                     extra_fields=["answer"])
    if not items:
        print("  Using fallback GSM8K problems")
        items = [{"prompt": p} for p in _GSM8K_FALLBACK[:n]]
        save_jsonl(path, items)
        return path
    _rng = random.Random(42)   # isolated — does not affect global random state
    _rng.shuffle(items)
    save_jsonl(path, items[:n])
    return path


# _GSM8K_VAL_POOL_SIZE: number of items reserved for validation files
# (gsm8k_30.jsonl + gsm8k_100.jsonl).  fetch_gsm8k_eval skips this many items
# from the front of the shuffled test set so val and eval never overlap.
# Must be ≥ max(n) passed to fetch_gsm8k.  Currently max is 100 (slow val).
_GSM8K_VAL_POOL_SIZE = 200   # 100 buffer beyond the largest val file


def fetch_gsm8k_eval(n=100, force=False):
    """GSM8K test split — EVAL POOL (items[VAL_POOL:VAL_POOL+n], isolated seed=42).

    These prompts are GUARANTEED not to overlap with gsm8k_30.jsonl or
    gsm8k_100.jsonl (the val files used during training for checkpoint selection).
    This is the file used for post-training evaluation of all models.

    For finalization (n=100): items[200:300] of the shuffled 1319-item test set.
    For paper (n=1319): use fetch_gsm8k_eval(n=1319) → items[200:1319] = 1119 items.
    All models (baseline, kl, traversal_tree, …) are evaluated on the SAME file
    so loss-to-loss comparisons are valid.

    Val-pool separation diagram (1319 test items, seed=42 shuffle):
      [0 ──── 99]  gsm8k_100.jsonl    (slow val / checkpoint selection)
      [0 ──── 29]  gsm8k_30.jsonl     (fast val / early stopping)
      [100 ─ 199]  buffer             (unused — extra separation)
      [200 ─ 299]  gsm8k_eval_100.jsonl  ← this file (finalization eval)
      [200 ─ 1318] gsm8k_eval_1119.jsonl ← paper eval (all non-val prompts)
    """
    _n_capped = min(n, 1319 - _GSM8K_VAL_POOL_SIZE)   # max 1119 non-val items
    path = os.path.join(DATA_DIR, f"gsm8k_eval_{_n_capped}.jsonl")
    if os.path.exists(path) and not force:
        print(f"  GSM8K eval({_n_capped}) already exists, skipping. Use --force to re-download.")
        return path
    print("Fetching GSM8K (eval pool)...")
    items = _hf_load("gsm8k", "test", "question", config="main", n=None,
                     extra_fields=["answer"])
    if not items:
        print("  Using fallback GSM8K problems (eval pool)")
        fallback = [{"prompt": p} for p in _GSM8K_FALLBACK]
        save_jsonl(path, fallback[:_n_capped])
        return path
    _rng = random.Random(42)   # same seed as fetch_gsm8k → same shuffle order
    _rng.shuffle(items)
    eval_items = items[_GSM8K_VAL_POOL_SIZE : _GSM8K_VAL_POOL_SIZE + _n_capped]
    save_jsonl(path, eval_items)
    return path


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------
_HUMANEVAL_FALLBACK = [
    # HumanEval-0..49: representative subset of well-known problems
    'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """Check if any two numbers are closer than threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    """\n',
    'def separate_paren_groups(paren_string: str) -> List[str]:\n    """Separate groups of nested parentheses into separate strings.\n    >>> separate_paren_groups("( ) (( )) (( )( ))")\n    ["()", "(())", "(()())"]\n    """\n',
    'def truncate_number(number: float) -> float:\n    """Return the decimal part of a positive float.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
    'def below_zero(operations: List[int]) -> bool:\n    """Return True if balance ever falls below zero.\n    >>> below_zero([1, 2, -4, 5])\n    True\n    """\n',
    'def mean_absolute_deviation(numbers: List[float]) -> float:\n    """Calculate Mean Absolute Deviation around the mean.\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """\n',
    'def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    """Insert delimeter between every two consecutive elements.\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    """\n',
    'def parse_nested_parens(paren_string: str) -> List[int]:\n    """Output deepest nesting level for each group.\n    >>> parse_nested_parens("(()()) ((())) () ((())()())")\n    [2, 3, 1, 3]\n    """\n',
    'def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    """Filter list to strings containing given substring.\n    >>> filter_by_substring(["abc", "bacd", "cde", "array"], "a")\n    ["abc", "bacd", "array"]\n    """\n',
    'def sum_product(numbers: List[int]) -> Tuple[int, int]:\n    """Return (sum, product) of all integers in list.\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    """\n',
    'def rolling_max(numbers: List[int]) -> List[int]:\n    """Generate rolling maximum found until each position.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    """\n',
    'def make_palindrome(string: str) -> str:\n    """Find shortest palindrome beginning with given string.\n    >>> make_palindrome("")\n    ""\n    >>> make_palindrome("cat")\n    "catac"\n    """\n',
    'def string_xor(a: str, b: str) -> str:\n    """XOR two binary strings digit by digit.\n    >>> string_xor("010", "110")\n    "100"\n    """\n',
    'def longest(strings: List[str]) -> Optional[str]:\n    """Return the longest string from the list.\n    >>> longest(["a", "bb", "ccc"])\n    "ccc"\n    """\n',
    'def greatest_common_divisor(a: int, b: int) -> int:\n    """Return GCD of two integers.\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 15)\n    5\n    """\n',
    'def all_prefixes(string: str) -> List[str]:\n    """Return list of all prefixes from shortest to longest.\n    >>> all_prefixes("abc")\n    ["a", "ab", "abc"]\n    """\n',
    'def string_sequence(n: int) -> str:\n    """Return a string of space-separated numbers from 0 to n.\n    >>> string_sequence(5)\n    "0 1 2 3 4 5"\n    """\n',
    'def count_distinct_characters(string: str) -> int:\n    """Count distinct characters in string ignoring case.\n    >>> count_distinct_characters("xyzXYZ")\n    3\n    """\n',
    'def parse_music(music_string: str) -> List[int]:\n    """Parse ASCII music notation to note durations.\n    o = whole (4), o| = half (2), .| = quarter (1).\n    >>> parse_music("o o| .| o| o| .| .| .| .| o o")\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    """\n',
    'def how_many_times(string: str, substring: str) -> int:\n    """Count occurrences of substring in string, including overlapping.\n    >>> how_many_times("aaa", "a")\n    3\n    """\n',
    'def sort_numbers(numbers: str) -> str:\n    """Sort space-delimited list of number words.\n    >>> sort_numbers("three one five")\n    "one three five"\n    """\n',
    'def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:\n    """Find the two closest elements in a sorted list.\n    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])\n    (2.0, 2.2)\n    """\n',
    'def rescale_to_unit(numbers: List[float]) -> List[float]:\n    """Rescale list to unit interval [0, 1].\n    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n    [0.0, 0.25, 0.5, 0.75, 1.0]\n    """\n',
    'def filter_integers(values: List[Any]) -> List[int]:\n    """Filter only integers from a list of values.\n    >>> filter_integers(["a", 3.14, 5])\n    [5]\n    """\n',
    'def strlen(string: str) -> int:\n    """Return length of the given string.\n    >>> strlen("")\n    0\n    >>> strlen("abc")\n    3\n    """\n',
    'def largest_divisor(n: int) -> int:\n    """For n > 1, return the largest divisor smaller than n.\n    >>> largest_divisor(15)\n    5\n    """\n',
    'def factorize(n: int) -> List[int]:\n    """Return list of prime factors of n in ascending order.\n    >>> factorize(8)\n    [2, 2, 2]\n    >>> factorize(25)\n    [5, 5]\n    """\n',
    'def remove_duplicates(numbers: List[int]) -> List[int]:\n    """Remove all duplicates from a list, keeping only unique elements.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n    """\n',
    'def flip_case(string: str) -> str:\n    """Flip lowercase to uppercase and vice versa.\n    >>> flip_case("Hello")\n    "hELLO"\n    """\n',
    'def concatenate(strings: List[str]) -> str:\n    """Concatenate list of strings into a single string.\n    >>> concatenate(["a", "b", "c"])\n    "abc"\n    """\n',
    'def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:\n    """Filter strings with given prefix.\n    >>> filter_by_prefix(["abc", "bcd", "cde", "array"], "a")\n    ["abc", "array"]\n    """\n',
    'def get_positive(l: list) -> list:\n    """Return only positive numbers from list.\n    >>> get_positive([-1, 2, -4, 3, 5])\n    [2, 3, 5]\n    """\n',
    'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    """\n',
    'def find_zero(xs: list) -> float:\n    """xs is coefficients of a polynomial. Find a zero.\n    >>> round(find_zero([1, 2]), 2)\n    -0.5\n    """\n',
    'def sort_third(l: list) -> list:\n    """Return list where values at indices divisible by 3 are sorted.\n    >>> sort_third([1, 2, 3])\n    [1, 2, 3]\n    >>> sort_third([5, 6, 3, 4, 8, 9, 2])\n    [2, 6, 3, 4, 8, 9, 5]\n    """\n',
    'def unique(l: list) -> list:\n    """Return sorted unique elements of list.\n    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])\n    [0, 2, 3, 5, 9, 123]\n    """\n',
    'def max_element(l: list) -> int:\n    """Return maximum element in list.\n    >>> max_element([1, 2, 3])\n    3\n    """\n',
    'def fizz_buzz(n: int) -> int:\n    """Count occurrences of 7 in numbers divisible by 11 or 13 below n.\n    >>> fizz_buzz(50)\n    0\n    >>> fizz_buzz(78)\n    2\n    """\n',
    'def sort_even(l: list) -> list:\n    """Return list with even-indexed values sorted, odd-indexed unchanged.\n    >>> sort_even([1, 2, 3])\n    [1, 2, 3]\n    """\n',
    'def encode_cyclic(s: str) -> str:\n    """Encode string by cycling groups of 3 characters.\n    """\n',
    'def prime_fib(n: int) -> int:\n    """Return nth Fibonacci number that is also prime.\n    >>> prime_fib(1)\n    2\n    >>> prime_fib(4)\n    89\n    """\n',
    'def triples_sum_to_zero(l: List[int]) -> bool:\n    """Return True if any three distinct elements sum to zero.\n    >>> triples_sum_to_zero([1, 3, 5, 0])\n    False\n    >>> triples_sum_to_zero([-3, 0, 1, 3, -2])\n    True\n    """\n',
    'def car_race_collision(n: int) -> int:\n    """n cars going left collide with n going right. Return collision count.\n    >>> car_race_collision(2)\n    4\n    """\n',
    'def incr_list(l: list) -> list:\n    """Return list with each element incremented by 1.\n    >>> incr_list([1, 2, 3])\n    [2, 3, 4]\n    """\n',
    'def pairs_sum_to_zero(l: list) -> bool:\n    """Return True if any two distinct elements sum to zero.\n    >>> pairs_sum_to_zero([1, 3, -2, 1])\n    False\n    >>> pairs_sum_to_zero([-1, 0, 1])\n    True\n    """\n',
    'def change_base(x: int, base: int) -> str:\n    """Change numerical base of integer x to given base.\n    >>> change_base(8, 3)\n    "22"\n    """\n',
    'def triangle_area(a: float, h: float) -> float:\n    """Compute triangle area given base and height.\n    >>> triangle_area(5, 3)\n    7.5\n    """\n',
    'def fib4(n: int) -> int:\n    """Compute n-th element of fib4 sequence (0,0,2,0,2,2,4,...).\n    """\n',
    'def median(l: list) -> float:\n    """Return median of list.\n    >>> median([3, 1, 2, 4, 5])\n    3\n    """\n',
    'def is_palindrome(text: str) -> bool:\n    """Return True if string is a palindrome.\n    >>> is_palindrome("")\n    True\n    >>> is_palindrome("aba")\n    True\n    """\n',
    'def modp(n: int, p: int) -> int:\n    """Return 2^n modulo p.\n    >>> modp(3, 5)\n    3\n    >>> modp(1101, 101)\n    2\n    """\n',
]

def fetch_humaneval(n=None, force=False):
    fname = f"humaneval_{n}.jsonl" if n else "humaneval.jsonl"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and not force:
        print(f"  HumanEval already exists, skipping.")
        return path
    print("Fetching HumanEval...")
    # Save test cases + entry_point alongside prompt for pass@1 evaluation
    items = _hf_load("openai_humaneval", "test", "prompt", n=n,
                     extra_fields=["test", "entry_point", "canonical_solution"])
    if not items:
        print("  Using fallback HumanEval prompts")
        items = [{"prompt": p} for p in _HUMANEVAL_FALLBACK]
    save_jsonl(path, items)
    return path


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------
_MATH_FALLBACK = [
    "Find the sum of all positive integers n such that n^2 - 19n + 99 is a perfect square.",
    "Let a, b, c be positive real numbers such that a + b + c = 1. Prove that a^2/(b+c) + b^2/(a+c) + c^2/(a+b) >= 1/2.",
    "How many positive integers less than 1000 are divisible by 2, 3, or 5?",
    "Find the number of ways to write 10 as an ordered sum of positive integers.",
    "What is the probability that two randomly chosen integers from 1 to 100 have no common factor greater than 1?",
    "Solve: |2x - 3| + |x + 1| = 6",
    "Find all complex numbers z such that z^4 = -4.",
    "A regular hexagon has side length 2. What is its area?",
    "If log_2(x) + log_4(x) + log_8(x) = 11, find x.",
    "Find the area of the region bounded by y = x^2 and y = x + 2.",
    "How many ways can 8 people be seated around a circular table if two particular people must not sit adjacent?",
    "Compute the sum of the infinite series 1 + 2/3 + 3/9 + 4/27 + ...",
    "If the roots of x^3 - 6x^2 + 11x - 6 = 0 are a, b, c, find a^2 + b^2 + c^2.",
    "A fair die is rolled 4 times. What is the probability that the product of the results is even?",
    "Find the largest prime factor of 2^15 - 1.",
    "A triangle has sides 13, 14, 15. Find its area.",
    "Evaluate: integral from 0 to pi of x*sin(x) dx",
    "Find the number of integer solutions to |x| + |y| + |z| = 10.",
    "What is the smallest n such that n! is divisible by 10^6?",
    "Find all real solutions of sqrt(x + sqrt(x + 11)) + sqrt(x - sqrt(x + 11)) = 4.",
    "If z = 3 + 4i, compute |z^3|.",
    "How many 4-digit palindromes are divisible by 9?",
    "Find the distance from the point (1, 2, 3) to the plane 2x - y + 2z = 5.",
    "Compute the determinant of the 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]].",
    "If f(x) = x^2 + 1 and g(x) = sqrt(x), find all x where f(g(x)) = g(f(x)).",
    "A circle with radius 5 is centered at (0,0). How many lattice points are inside or on the circle?",
    "Find the remainder when 7^100 is divided by 50.",
    "Solve the differential equation dy/dx = y*cos(x), y(0) = 1.",
    "Find all values of k for which the system x + 2y = 3, 2x + ky = 6 has no solution.",
    "What is the coefficient of x^5 in the expansion of (1 + x + x^2)^10?",
]

def fetch_math500(n=30, force=False):
    path = os.path.join(DATA_DIR, f"math500_{n}.jsonl")
    if os.path.exists(path) and not force:
        print(f"  MATH500({n}) already exists, skipping.")
        return path
    print("Fetching MATH-500...")
    prompts = _hf_load("lighteval/MATH-Hard", "test", "problem", n=n)
    if not prompts:
        prompts = _hf_load("hendrycks/competition_math", "test", "problem", n=n)
    if not prompts:
        print("  Using fallback MATH problems")
        prompts = _MATH_FALLBACK[:n]
    save_jsonl(path, prompts)
    return path


# ---------------------------------------------------------------------------
# MT-Bench (80 questions across 8 categories)
# ---------------------------------------------------------------------------
_MTBENCH_QUESTIONS = [
    # Writing (10)
    "Compose a haiku about a robot learning to love.",
    "Write a short story (100 words) about an AI that discovers it has emotions.",
    "Write a persuasive essay arguing that remote work increases productivity.",
    "Compose a cover letter for a software engineer applying to a startup.",
    "Write a poem about the ocean at night using personification.",
    "Draft a professional email declining a job offer politely.",
    "Write a 3-paragraph introduction for a blog post about machine learning.",
    "Compose a dialogue between two scientists debating whether AI can be conscious.",
    "Write a product description for a smartwatch targeting fitness enthusiasts.",
    "Create a short motivational speech for a team facing a difficult deadline.",
    # Roleplay (10)
    "You are a medieval knight. Describe your daily routine.",
    "Act as a financial advisor. I have $10,000 to invest. What do you recommend?",
    "You are Albert Einstein in 1905. Explain your theory of special relativity in simple terms.",
    "Act as a travel agent. Plan a 7-day trip to Japan for a family of 4.",
    "You are a chef. Explain how to make a classic French omelette step by step.",
    "Act as a career counselor helping a student choose between CS and medicine.",
    "You are a Socratic philosopher. Challenge the idea that money equals happiness.",
    "Act as a fitness coach creating a beginner workout plan for someone with no equipment.",
    "You are a museum curator. Describe the most significant artifact in your collection.",
    "Act as a debate coach and give me tips for winning a debate on climate change.",
    # Reasoning (10)
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If all bloops are razzles and all razzles are lazzles, are all bloops definitely lazzles?",
    "You have 12 balls, one of which is heavier. Using a balance scale only 3 times, how do you find the heavy ball?",
    "Three friends split a restaurant bill. The bill is $30 so each pays $10. The waiter returns $5 but keeps $2 as tip. Each friend gets $1 back, so paid $9 each = $27 + $2 = $29. Where's the missing dollar?",
    "A snail climbs 3 meters up a 10-meter wall each day but slides 2 meters each night. How many days to reach the top?",
    "You're in a room with two doors. One leads to freedom, one to death. Two guards, one always lies, one always tells truth. What question do you ask?",
    "Five houses in a row, each with a different color. Use the clues to determine who owns the fish (Einstein's puzzle).",
    "A train leaves Chicago at 60 mph. Another leaves NYC at 90 mph. The cities are 790 miles apart. When do they meet?",
    "If I have a 3-gallon jug and a 5-gallon jug, how do I measure exactly 4 gallons?",
    "There are 100 prisoners numbered 1-100. Each must find their own number in a room with 100 boxes. Strategy?",
    # Math (10)
    "Solve: 2x^2 - 5x + 3 = 0",
    "What is the integral of x*e^x dx?",
    "Prove that the square root of 2 is irrational.",
    "Find the sum of the first 100 positive integers.",
    "What is 15% of 240?",
    "If f(x) = 3x^2 - 2x + 1, find f'(x) and f''(x).",
    "Solve the system: 2x + 3y = 12, x - y = 1.",
    "What is the probability of rolling at least one 6 in 4 dice rolls?",
    "Compute: lim(x->0) sin(x)/x",
    "How many permutations of the letters MISSISSIPPI are there?",
    # Coding (10)
    "Write a Python function that finds the longest common subsequence of two strings.",
    "Implement a binary tree in Python with insert, search, and delete methods.",
    "Write a Python decorator that caches function results (memoization).",
    "Implement merge sort in Python and explain its time complexity.",
    "Write a Python class implementing a min-heap.",
    "Write a SQL query to find the second-highest salary from an employees table.",
    "Implement a simple LRU cache in Python using collections.OrderedDict.",
    "Write a regex pattern to validate email addresses.",
    "Write a Python function to detect if a linked list has a cycle.",
    "Implement Dijkstra's algorithm for shortest path in Python.",
    # Extraction (10)
    "Extract all email addresses from this text: 'Contact john@example.com or mary.smith@corp.org for details.'",
    "Parse the following JSON and extract all values for the key 'name'.",
    "List the main arguments made in the following paragraph (summarize as bullet points).",
    "From the following table, identify rows where sales exceeded $1000.",
    "Extract all dates mentioned in this paragraph about historical events.",
    "Identify the sentiment (positive/negative/neutral) of each sentence in the following paragraph.",
    "From the following resume text, extract: name, email, skills, years of experience.",
    "Parse the following CSV row: '42, \"Smith, John\", 85000, \"New York\"'",
    "Extract all named entities (people, places, organizations) from this news excerpt.",
    "Given the following product reviews, rank products by average rating.",
    # STEM (10)
    "Explain how CRISPR-Cas9 gene editing works.",
    "What is the difference between nuclear fission and nuclear fusion?",
    "Explain the mechanism of action of mRNA vaccines.",
    "What is the Schrödinger equation and what does it describe?",
    "Explain how a transistor works and why it's important for computing.",
    "What causes the northern lights (aurora borealis)?",
    "Explain the second law of thermodynamics with a concrete example.",
    "What is the difference between a virus and a bacterium?",
    "Explain how GPS works to determine location.",
    "What is dark matter and how do scientists detect it?",
    # Humanities (10)
    "Compare the philosophies of Plato and Aristotle on the nature of reality.",
    "What were the main causes of World War I? Rank them by importance.",
    "Explain the key ideas in Kant's Critique of Pure Reason.",
    "How did the Renaissance transform European art and thought?",
    "Compare capitalism and socialism as economic systems.",
    "What is postmodernism? Give an example from literature or art.",
    "Explain the concept of cognitive dissonance and give an example.",
    "How did colonialism shape the modern world?",
    "What is the trolley problem and what does it reveal about moral philosophy?",
    "Explain the significance of the Magna Carta in the history of democracy.",
]

def fetch_mtbench(force=False):
    path = os.path.join(DATA_DIR, "mtbench_80.jsonl")
    if os.path.exists(path) and not force:
        print(f"  MTBench already exists, skipping.")
        return path
    print("Fetching MTBench...")
    prompts = _hf_load("HuggingFaceH4/mt_bench_prompts", "train", "turns", n=None)
    if not prompts:
        print("  Using hardcoded MTBench questions")
        prompts = _MTBENCH_QUESTIONS
    save_jsonl(path, prompts)
    return path


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------
_ALPACA_FALLBACK = [
    "Explain the concept of machine learning to a 10-year-old.",
    "Give me 5 tips for improving my public speaking skills.",
    "Write a recipe for banana bread.",
    "What are the pros and cons of electric vehicles?",
    "Explain the difference between machine learning and deep learning.",
    "Suggest 5 books everyone should read before they turn 30.",
    "How do I start learning a new programming language effectively?",
    "What is the best way to prepare for a job interview?",
    "Write a morning routine for someone who wants to be more productive.",
    "Explain blockchain in simple terms.",
    "What are some strategies for managing stress?",
    "How do I negotiate a higher salary?",
    "Describe the scientific method.",
    "What are the key principles of good software design?",
    "Give me a meal plan for a week for someone trying to lose weight.",
    "How do I improve my memory and retain information better?",
    "Explain the difference between HTTP and HTTPS.",
    "Write a cover letter template for a marketing position.",
    "What are the best practices for cybersecurity at home?",
    "How does compound interest work? Give an example.",
    "What is the difference between introversion and shyness?",
    "Explain how vaccines work.",
    "Give 5 tips for writing better code.",
    "What causes inflation and how can it be controlled?",
    "Describe the process of photosynthesis.",
    "How do I build a habit and make it stick?",
    "Explain the concept of supply and demand.",
    "What is the difference between machine learning and statistics?",
    "How can I improve my time management?",
    "Give me 10 interesting facts about space.",
]

def fetch_alpaca(n=30, force=False):
    path = os.path.join(DATA_DIR, f"alpaca_{n}.jsonl")
    if os.path.exists(path) and not force:
        print(f"  Alpaca({n}) already exists, skipping.")
        return path
    print("Fetching Alpaca...")
    prompts = _hf_load("tatsu-lab/alpaca", "train", "instruction", n=n)
    if not prompts:
        print("  Using fallback Alpaca instructions")
        prompts = _ALPACA_FALLBACK[:n]
    save_jsonl(path, prompts)
    return path


# ---------------------------------------------------------------------------
# Diverse-50 (our existing test set)
# ---------------------------------------------------------------------------
_DIVERSE50 = [
    # Factual
    "What is the capital of France?", "What is the capital of Japan?",
    "What is Newton's second law of motion?", "What is the speed of light?",
    "What is the Pythagorean theorem?", "What is Ohm's law?",
    "What is the difference between RAM and ROM?", "What is Moore's law?",
    "Summarize the French Revolution in two sentences.",
    "What is quantum entanglement in simple terms?",
    # Coding
    "Write a Python function that returns the factorial of n.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a function to reverse a string.",
    "Write a function to check if a number is prime.",
    "Write a function to find the maximum subarray sum.",
    "Write a function to do binary search on a sorted array.",
    "Write a function to flatten a nested list.",
    "Write a function to find the first non-repeating character.",
    "Write a Python decorator that measures execution time.",
    "Write a function to compute Fibonacci numbers efficiently.",
    # CS concepts
    "Explain what a linked list is in one sentence.",
    "How does a hash table work?",
    "What is the difference between a stack and a queue?",
    "What is the time complexity of merge sort?",
    "What is the difference between BFS and DFS?",
    "What is dynamic programming?",
    "What is the difference between a process and a thread?",
    "How does virtual memory work?",
    "What is a deadlock?",
    "What is the difference between TCP and UDP?",
    # ML/AI
    "How does gradient descent work?",
    "What is overfitting in machine learning and how do you prevent it?",
    "What is backpropagation?",
    "What is the softmax function?",
    "Explain what attention mechanism does in transformers.",
    "What is the difference between supervised and unsupervised learning?",
    "What is cross-entropy loss?",
    "What is batch normalization?",
    "What is transfer learning?",
    "What is the bias-variance tradeoff?",
    # Math
    "What is 2 + 2?",
    "What is the derivative of sin(x)?",
    "Solve the equation x^2 - 5x + 6 = 0.",
    "What is Bayes' theorem?",
    "What is Big-O notation? Give an example.",
    "What is a normal distribution?",
    "What is the central limit theorem?",
    "What is a p-value in statistics?",
    "What is standard deviation?",
    "What is linear regression?",
]

_DIVERSE50_CATEGORIES = (
    ["factual"] * 10 + ["coding"] * 10 + ["cs_concepts"] * 10
    + ["ml_ai"] * 10 + ["math"] * 10
)

def write_diverse50(force=False):
    path = os.path.join(DATA_DIR, "diverse50.jsonl")
    if os.path.exists(path) and not force:
        print(f"  diverse50 already exists, skipping.")
        return path
    items = [{"prompt": p, "category": c}
             for p, c in zip(_DIVERSE50, _DIVERSE50_CATEGORIES)]
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"  -> 50 prompts saved to {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# Training split datasets (for DistillSpec fine-tuning)
# ---------------------------------------------------------------------------

def fetch_gsm8k_train(n=None, force=False):
    """Download GSM8K train split (7,473 math word problems)."""
    fname = f"gsm8k_train_{n}.jsonl" if n else "gsm8k_train.jsonl"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and not force:
        print(f"  GSM8K train already exists, skipping.")
        return path
    print("Fetching GSM8K train split...")
    prompts = _hf_load("gsm8k", "train", "question", config="main", n=n)
    if not prompts:
        print("  [warning] GSM8K train fallback — using eval fallback prompts as placeholder")
        prompts = _GSM8K_FALLBACK
    save_jsonl(path, prompts)
    return path


def fetch_math_train(n=None, force=False):
    """Download MATH competition dataset train split (~7,500 problems)."""
    fname = f"math_train_{n}.jsonl" if n else "math_train.jsonl"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and not force:
        print(f"  MATH train already exists, skipping.")
        return path
    print("Fetching MATH train split...")
    prompts = _hf_load("hendrycks/competition_math", "train", "problem", n=n)
    if not prompts:
        prompts = _hf_load("lighteval/MATH-Hard", "train", "problem", n=n)
    if not prompts:
        print("  [warning] MATH train fallback — using eval fallback problems as placeholder")
        prompts = _MATH_FALLBACK
    save_jsonl(path, prompts)
    return path


def fetch_alpaca_train(n=None, force=False):
    """Download full Alpaca dataset (52K instructions — all training data, no eval split)."""
    fname = f"alpaca_train_{n}.jsonl" if n else "alpaca_train.jsonl"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and not force:
        print(f"  Alpaca train already exists, skipping.")
        return path
    print("Fetching Alpaca train (full dataset)...")
    prompts = _hf_load("tatsu-lab/alpaca", "train", "instruction", n=n)
    if not prompts:
        print("  [warning] Alpaca train fallback — using small fallback set")
        prompts = _ALPACA_FALLBACK
    save_jsonl(path, prompts)
    return path


def fetch_wikitext_train(force=False):
    """Download WikiText-2 for GPT-2 family distillation.

    WikiText-2 is in-distribution for distilgpt2 and gpt2-medium (both trained on
    WebText / curated web text).  GSM8K math is OOD for GPT-2 → noisy distillation.
    This dataset produces clean convergence for all losses on the GPT-2 family.
    """
    train_path = os.path.join(DATA_DIR, "wikitext_train.jsonl")
    eval10_path = os.path.join(DATA_DIR, "wikitext_10.jsonl")
    eval5_path  = os.path.join(DATA_DIR, "wikitext_5.jsonl")
    if os.path.exists(train_path) and not force:
        n = sum(1 for _ in open(train_path))
        print(f"  wikitext_train.jsonl already exists ({n} prompts), skipping.")
        return train_path
    print("Fetching WikiText-2 (wikitext-2-raw-v1, train split)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        prompts = []
        for item in ds:
            text = item["text"].strip()
            if not text or text.startswith("="):
                continue
            words = text.split()
            if 20 <= len(words) <= 300:
                prompts.append({"prompt": text})
        print(f"  {len(prompts)} paragraphs (filtered from {len(ds)} raw entries)")
        save_jsonl(train_path, prompts)
        val = random.sample(prompts, min(10, len(prompts)))
        save_jsonl(eval10_path, val)
        save_jsonl(eval5_path, val[:5])
        return train_path
    except Exception as e:
        print(f"  [warning] WikiText download failed: {e}")
        print("  Run: python core/datasets/download_wikitext.py")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_DATASETS = ["diverse50", "gsm8k", "gsm8k_eval", "humaneval", "math500", "mtbench", "alpaca"]
TRAIN_DATASETS = ["gsm8k_train", "math_train", "alpaca_train", "wikitext_train"]

def fetch_all(n=30, force=False, datasets=None, train=False):
    if datasets is None:
        datasets = ALL_DATASETS
    print(f"\nFetching datasets: {datasets}\n")
    paths = {}
    if "diverse50" in datasets:
        paths["diverse50"] = write_diverse50(force=force)
    if "gsm8k" in datasets:
        paths["gsm8k"] = fetch_gsm8k(n=n, force=force)
    if "gsm8k_eval" in datasets:
        paths["gsm8k_eval"] = fetch_gsm8k_eval(n=n, force=force)
    if "humaneval" in datasets:
        paths["humaneval"] = fetch_humaneval(force=force)
    if "math500" in datasets:
        paths["math500"] = fetch_math500(n=n, force=force)
    if "mtbench" in datasets:
        paths["mtbench"] = fetch_mtbench(force=force)
    if "alpaca" in datasets:
        paths["alpaca"] = fetch_alpaca(n=n, force=force)
    if train or "gsm8k_train" in datasets:
        paths["gsm8k_train"] = fetch_gsm8k_train(force=force)
    if train or "math_train" in datasets:
        paths["math_train"] = fetch_math_train(force=force)
    if train or "alpaca_train" in datasets:
        paths["alpaca_train"] = fetch_alpaca_train(force=force)
    if train or "wikitext_train" in datasets:
        # Always download WikiText for GPT-2 family convergence verification
        paths["wikitext_train"] = fetch_wikitext_train(force=force)
    print("\nDone.")
    return paths


def get_dataset_path(name: str, n: int = 30) -> str:
    """Return JSONL path for a dataset (creating it if needed)."""
    # gsm8k      → val pool  (items[0:n], seed=42) — used during training val/checkpoint selection
    # gsm8k_eval → eval pool (items[200:200+n], seed=42) — used for post-training evaluation
    # These two pools are non-overlapping by construction (see fetch_gsm8k_eval docstring).
    _eval_n = min(n, 1319 - _GSM8K_VAL_POOL_SIZE)
    mapping = {
        "diverse50":   os.path.join(DATA_DIR, "diverse50.jsonl"),
        "gsm8k":       os.path.join(DATA_DIR, f"gsm8k_{n}.jsonl"),
        "gsm8k_eval":  os.path.join(DATA_DIR, f"gsm8k_eval_{_eval_n}.jsonl"),
        "humaneval":   os.path.join(DATA_DIR, "humaneval.jsonl"),
        "math500":     os.path.join(DATA_DIR, f"math500_{n}.jsonl"),
        "mtbench":     os.path.join(DATA_DIR, "mtbench_80.jsonl"),
        "alpaca":      os.path.join(DATA_DIR, f"alpaca_{n}.jsonl"),
    }
    path = mapping.get(name)
    if path and not os.path.exists(path):
        fetch_all(n=n, datasets=[name])
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(ALL_DATASETS),
                   help="Comma-separated eval datasets to fetch.")
    p.add_argument("--n", type=int, default=30,
                   help="Max eval prompts per dataset (where applicable).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if files exist.")
    p.add_argument("--train", action="store_true",
                   help="Also fetch train splits (GSM8K, MATH, Alpaca) for DistillSpec training.")
    args = p.parse_args()
    fetch_all(n=args.n, force=args.force, datasets=args.datasets.split(","), train=args.train)
