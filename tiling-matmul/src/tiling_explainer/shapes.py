def check_size(kernel: str, multiple: int, n: int) -> None:
    if n <= 0 or n % multiple:
        raise ValueError(f"{kernel} needs n to be a positive multiple of {multiple}, got {n}")
