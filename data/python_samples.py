
"""
Small Python dataset for training a tiny code language model.

These examples are for experimentation, not production use.
"""

PYTHON_SAMPLES = [
    """
def add(a, b):
    return a + b
""",
    """
def subtract(a, b):
    return a - b
""",
    """
def multiply(a, b):
    return a * b
""",
    """
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""",
    """
def square(n):
    return n * n
""",
    """
def cube(n):
    return n * n * n
""",
    """
def is_even(n):
    return n % 2 == 0
""",
    """
def is_odd(n):
    return n % 2 != 0
""",
    """
def factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
""",
    """
def fibonacci(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""",
    """
def find_max(numbers):
    if not numbers:
        return None
    return max(numbers)
""",
    """
def find_min(numbers):
    if not numbers:
        return None
    return min(numbers)
""",
    """
def calculate_average(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
""",
    """
def count_items(items):
    return len(items)
""",
    """
def reverse_list(items):
    return items[::-1]
""",
    """
def remove_duplicates(items):
    return list(dict.fromkeys(items))
""",
    """
def filter_positive(numbers):
    return [n for n in numbers if n > 0]
""",
    """
def filter_even(numbers):
    return [n for n in numbers if n % 2 == 0]
""",
    """
def sum_squares(numbers):
    return sum(n * n for n in numbers)
""",
    """
def second_largest(numbers):
    unique = sorted(set(numbers))
    if len(unique) < 2:
        return None
    return unique[-2]
""",
    """
def reverse_text(text):
    return text[::-1]
""",
    """
def count_vowels(text):
    vowels = "aeiou"
    return sum(1 for char in text.lower() if char in vowels)
""",
    """
def is_palindrome(text):
    cleaned = "".join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]
""",
    """
def greet(name):
    return f"Hello, {name}!"
""",
    """
def capitalize_words(text):
    return text.title()
""",
    """
def count_words(text):
    return len(text.split())
""",
    """
def get_initials(name):
    return "".join(part[0].upper() for part in name.split() if part)
""",
    """
def merge_dicts(first, second):
    return {**first, **second}
""",
    """
def invert_dict(mapping):
    return {value: key for key, value in mapping.items()}
""",
    """
def get_value(mapping, key, default=None):
    return mapping.get(key, default)
""",
    """
def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
""",
    """
def clamp(value, minimum, maximum):
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    return max(minimum, min(value, maximum))
""",
    """
def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32
""",
    """
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9
""",
    """
def is_prime(n):
    if n < 2:
        return False
    for divisor in range(2, int(n ** 0.5) + 1):
        if n % divisor == 0:
            return False
    return True
""",
    """
def flatten_once(nested):
    return [item for group in nested for item in group]
""",
    """
def chunk_list(items, size):
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]
""",
    """
def find_indices(items, target):
    return [i for i, item in enumerate(items) if item == target]
""",
    """
def word_frequency(text):
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts
""",
    """
def safe_get(items, index, default=None):
    try:
        return items[index]
    except (IndexError, TypeError):
        return default
""",
    """
def percentage(part, total):
    if total == 0:
        raise ValueError("total cannot be zero")
    return part / total * 100
""",
]
