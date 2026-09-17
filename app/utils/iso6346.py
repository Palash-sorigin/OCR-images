import re


CONTAINER_PATTERN = re.compile(
    r"^[A-Z]{3}[UJZ]\d{7}$"
)


def normalize_container_number(value: str) -> str:
    """
    Normalize a possible container number.

    Removes spaces, hyphens, underscores and any other
    non-alphanumeric characters.
    """

    return re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )


def _letter_value(char: str) -> int:
    """
    ISO 6346 letter-to-number mapping.

    ISO 6346 assigns values while skipping multiples of 11.
    """

    values = {
        "A": 10,
        "B": 12,
        "C": 13,
        "D": 14,
        "E": 15,
        "F": 16,
        "G": 17,
        "H": 18,
        "I": 19,
        "J": 20,
        "K": 21,
        "L": 23,
        "M": 24,
        "N": 25,
        "O": 26,
        "P": 27,
        "Q": 28,
        "R": 29,
        "S": 30,
        "T": 31,
        "U": 32,
        "V": 34,
        "W": 35,
        "X": 36,
        "Y": 37,
        "Z": 38,
    }

    return values[char]


def calculate_check_digit(container_number_10: str) -> int:
    """
    Calculate the ISO 6346 check digit.

    Input:
        First 10 characters of a container number.

    Example:
        CMAU980286
        -> 7
    """

    value = normalize_container_number(container_number_10)

    if len(value) != 10:
        raise ValueError(
            "ISO 6346 check digit calculation requires exactly 10 characters."
        )

    if not re.fullmatch(
        r"[A-Z]{3}[UJZ]\d{6}",
        value,
    ):
        raise ValueError(
            "Invalid ISO 6346 first 10 characters."
        )

    total = 0

    for position, char in enumerate(value):
        if char.isalpha():
            numeric_value = _letter_value(char)
        else:
            numeric_value = int(char)

        total += numeric_value * (2 ** position)

    remainder = total % 11

    # ISO 6346 specifies that remainder 10 becomes check digit 0.
    if remainder == 10:
        return 0

    return remainder


def complete_iso6346_number(prefix_and_serial: str) -> str | None:
    """
    Complete a 10-character ISO 6346 container number
    by calculating its final check digit.

    Example:

        CMAU980286
            ↓
        CMAU9802867
    """

    value = normalize_container_number(prefix_and_serial)

    if len(value) != 10:
        return None

    if not re.fullmatch(
        r"[A-Z]{3}[UJZ]\d{6}",
        value,
    ):
        return None

    check_digit = calculate_check_digit(value)

    return value + str(check_digit)


def is_valid_iso6346(value: str) -> bool:
    """
    Validate a complete 11-character ISO 6346 container number.

    Format:

        AAAU1234567

    Where:

        AAA = owner code
        U/J/Z = equipment category
        123456 = serial number
        7 = check digit
    """

    value = normalize_container_number(value)

    if not CONTAINER_PATTERN.fullmatch(value):
        return False

    first_10 = value[:10]
    supplied_check_digit = int(value[10])

    calculated_check_digit = calculate_check_digit(first_10)

    return supplied_check_digit == calculated_check_digit