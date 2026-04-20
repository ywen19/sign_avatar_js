from typing import List, Tuple

from language_utils.identity_lookup import IdentityLookup


CONNECTOR_WORDS = {"to", "from"}


def _collect_forward_time_span(
    chunks: List[str],
    reorder_tags: List[int],
    start_idx: int,
) -> Tuple[List[str], int]:
    """
    Collect a forward-only time span.

    Valid starts:
    - a TIME-tagged chunk
    - "from" immediately followed by a TIME-tagged chunk
    """
    left = start_idx

    if (
        chunks[start_idx] == "from"
        and start_idx + 1 < len(chunks)
        and reorder_tags[start_idx + 1] == IdentityLookup.REORDER_TIME
    ):
        right = start_idx + 2
    else:
        right = start_idx + 1

    while right < len(chunks):
        current_chunk = chunks[right]
        current_tag = reorder_tags[right]

        if current_tag == IdentityLookup.REORDER_TIME:
            right += 1
            continue

        if current_tag == IdentityLookup.REORDER_NORMAL:
            right += 1
            continue

        if current_chunk in CONNECTOR_WORDS:
            right += 1
            continue

        break

    while right > left and chunks[right - 1] in CONNECTOR_WORDS:
        right -= 1

    return chunks[left:right], right


def reorder_by_tags(chunks: List[str], reorder_tags: List[int]) -> List[str]:
    if len(chunks) != len(reorder_tags):
        raise ValueError("chunks and reorder_tags must have the same length")

    units = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]
        tag = reorder_tags[i]

        # case 1: direct time anchor
        if tag == IdentityLookup.REORDER_TIME:
            span, next_i = _collect_forward_time_span(chunks, reorder_tags, i)
            units.append(("time", span))
            i = next_i
            continue

        # case 2: "from" introducing a time span
        if (
            chunk == "from"
            and i + 1 < len(chunks)
            and reorder_tags[i + 1] == IdentityLookup.REORDER_TIME
        ):
            span, next_i = _collect_forward_time_span(chunks, reorder_tags, i)
            units.append(("time", span))
            i = next_i
            continue

        if tag == IdentityLookup.REORDER_NEGATION:
            units.append(("negation", [chunk]))
            i += 1
            continue

        if tag == IdentityLookup.REORDER_WH:
            units.append(("wh", [chunk]))
            i += 1
            continue

        units.append(("middle", [chunk]))
        i += 1

    reordered = []

    for unit_type, unit_chunks in units:
        if unit_type == "time":
            reordered.extend(unit_chunks)

    for unit_type, unit_chunks in units:
        if unit_type == "middle":
            reordered.extend(unit_chunks)

    for unit_type, unit_chunks in units:
        if unit_type == "negation":
            reordered.extend(unit_chunks)

    for unit_type, unit_chunks in units:
        if unit_type == "wh":
            reordered.extend(unit_chunks)

    return reordered