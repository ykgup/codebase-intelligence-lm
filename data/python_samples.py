"""Small, hand-written Python examples for initial experiments."""

PYTHON_SAMPLES = [
    """def add(a, b):
    return a + b
""",
    """def subtract(a, b):
    return a - b
""",
    """def multiply(a, b):
    return a * b
""",
    """def is_even(n):
    return n % 2 == 0
""",
    """def square(n):
    return n * n
""",
    """def find_max(numbers):
    if not numbers:
        return None
    return max(numbers)
""",
    """def count_items(items):
    return len(items)
""",
    """def reverse_text(text):
    return text[::-1]
""",
    """def greet(name):
    return f"Hello, {name}!"
""",
    """def factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
""",
    """def average(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
""",
    """def remove_duplicates(items):
    return list(dict.fromkeys(items))
""",
]