import pytest
from src.rag.notebook_chunker import extract_exercise_number

@pytest.mark.parametrize("content, expected", [
    # Pattern 1: # GRADED CELL: exercise <number>
    ("# GRADED CELL: exercise 1", "1"),
    ("# GRADED CELL: exercise 42", "42"),
    ("# graded cell: exercise 2", "2"),  # Case insensitive
    ("#GRADED CELL: exercise 3", "3"),   # No space after #
    ("#   GRADED   CELL:   exercise   4", "4"), # Multiple spaces

    # Pattern 2: ### Exercise <number>
    ("### Exercise 5:", "5"),
    ("### exercise 6:", "6"), # Case insensitive
    ("###Exercise 7:", "7"), # No space after ###
    ("###   Exercise   8:", "8"), # Multiple spaces

    # Pattern 3: Exercise <number>
    ("Exercise 9 - ", "9"),
    ("exercise 10", "10"), # Case insensitive
    ("Exercise 123", "123"), # Multi-digit

    # Real-world mixed examples
    ("Some text before. # GRADED CELL: exercise 11. Some text after.", "11"),
    ("### Exercise 12: Implement a function", "12"),

    # Negative cases (no match)
    ("Just a regular text string", None),
    ("Exercise abc", None), # No digits
    ("# GRADED CELL: something else", None),
    ("", None),
])
def test_extract_exercise_number(content, expected):
    """Test extraction of exercise numbers from notebook cell contents."""
    assert extract_exercise_number(content) == expected
