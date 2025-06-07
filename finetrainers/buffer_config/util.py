import re

def parse_partition_string(partition_str):
    """
    Parse partition string like 'c1b3t9' into a dictionary.

    Args:
        partition_str (str): Partition string, e.g. 'c1b3t9'

    Returns:
        dict: {'condition': int, 'buffer': int, 'target': int}
    """

    match = re.fullmatch(r'c(\d+)b(\d+)t(\d+)', partition_str)
    if match is None:
        raise ValueError(f"Invalid partition string format: {partition_str}")

    return {
        'condition': int(match.group(1)),
        'buffer': int(match.group(2)),
        'target': int(match.group(3))
    }
