from pathlib import Path


def save_file(content: str, file_path: str) -> None:
    """Save string content to a file with automatic directory creation.

    This function writes the provided string content to the specified file path.
    If the parent directories do not exist, they will be created automatically.

    Args:
        content: The string content to write to the file.
        file_path: The path where the file should be saved (relative or absolute).

    Returns:
        None

    Raises:
        OSError: When file write fails due to permissions or invalid path.
        IOError: When directory creation fails.

    Example:
        >>> save_file('Example', 'outputs/greeting.txt')
        >>> save_file('{"key": "value"}', 'outputs/results.json')
    """
    path = Path(file_path)
    parent_dir = path.parent

    # Create parent directories if they don't exist
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Write content to a file with UTF-8 encoding
    with path.open('w', encoding='utf-8') as f:
        f.write(content)
