"""
Defines the chunk reordering logic used before sign language motion generation.

Time expressions are moved to the front, normal content stays in the middle, 
negation chunks are moved later, and WH-question chunks are moved to the end.

This helps convert spoken/written English-style chunk order into a structure that
better matches BSL-style signing order.
"""


from typing import List, Tuple, Set

from language_utils.identity_lookup import IdentityLookup


CONNECTOR_WORDS: Set[str] = {"to", "from"}


def _collect_forward_time_span(
    chunks: List[str], reorder_tags: List[int], start_idx: int,
) -> Tuple[List[str], int]:
    """
    Collect a complete time phrase by scanning forward from the given position.

    This is used to group related time chunks into one span, so the whole time
    phrase can be moved to the front of the sentence later.

    The function can start from:
    1. a chunk tagged as TIME
    2. the word "from" when it is immediately followed by a TIME chunk

    While scanning forward, it keeps collecting:
    1. TIME chunks
    2. NORMAL chunks
    3. connector words such as "to" or "from"

    If the collected span ends with a connector word, the connector is removed 
    so the time phrase does not end with words like "to" or "from".

    The span stops when it reaches a chunk that is not TIME, NORMAL, or a 
    connector word.

    chunks        (List[str])  :  tokenized text chunks.
    reorder_tags  (List[int])  :  reorder tags aligned with chunks.
    start_idx     (int)        :  index where the time span starts.

    Return:
    (Tuple[List[str], int])    :  collected time span and the next index to
    continue scanning from.
    """
    left = start_idx  # left boundary of the time span

    # handle time ranges that start with "from", such as "from monday"
    if (
        chunks[start_idx] == "from"
        and start_idx + 1 < len(chunks)
        and reorder_tags[start_idx + 1] == IdentityLookup.REORDER_TIME
    ):
        # include both "from" and the following TIME chunk
        right = start_idx + 2 
    else:
        # otherwise, start with the current TIME chunk only
        right = start_idx + 1

    while right < len(chunks):
        # get the next chunk and its reorder tag
        current_chunk = chunks[right]
        current_tag = reorder_tags[right]

        # keep collecting TIME chunks
        if current_tag == IdentityLookup.REORDER_TIME:
            right += 1
            continue
        # keep collecting NORMAL chunks because they may be part 
        # of a time phrase
        if current_tag == IdentityLookup.REORDER_NORMAL:
            right += 1
            continue
        # keep collecting connector words like "to" or "from"
        if current_chunk in CONNECTOR_WORDS:
            right += 1
            continue
        # stop when the next chunk no longer looks like part of 
        # the time phrase
        break

    # defensive cleanup: remove an unfinished trailing connector, e.g. avoid
    # returning "from monday to" if the scan stops before another time chunk
    # example: ["i", "am", "free", "from", "monday", "to", "not", "sure"]
    # without defensive cleanup, we will have span as ["from", "monday", "to"]
    # with defensive cleanup we have ["from", "monday"]
    while right > left and chunks[right - 1] in CONNECTOR_WORDS:
        right -= 1
    # return the collected time span and the next index for the outer scanner
    return chunks[left:right], right


def reorder_by_tags(chunks: List[str], reorder_tags: List[int]) -> List[str]:
    """
    Reorder chunks based on their identity reorder tags.

    This function groups the input chunks into units such as time spans,
    normal middle content, negation chunks, and WH-question chunks. It then
    rebuilds the sentence in the target order used before sign language motion
    generation: time -> middle -> negation -> wh

    Time expressions are grouped as spans so phrases like "from monday to friday"
    can be moved together instead of being split apart.

    chunks        (List[str])  :  tokenized text chunks
    reorder_tags  (List[int])  :  reorder tags aligned with each chunk

    Return:
    reordered     (List[str])  :  reodered chunks
    """
    if len(chunks) != len(reorder_tags):
        raise ValueError("chunks and reorder_tags must have the same length")

    units = []  # store grouped units before rebuilding the final order
    i = 0  # current index while scanning through chunks

    # scan all chunks from left to right
    while i < len(chunks):
        chunk = chunks[i]
        tag = reorder_tags[i]

        # case 1: current chunk is directly tagged as TIME
        if tag == IdentityLookup.REORDER_TIME:
            # collect the whole time phrase starting at this chunk
            span, next_i = _collect_forward_time_span(chunks, reorder_tags, i)
            # store the full time phrase as one time unit
            units.append(("time", span))
            # move the scan index to the first chunk after this time span
            i = next_i
            continue

        # case 2: current chunk is "from" and introduces a time phrase
        if (
            chunk == "from"
            and i + 1 < len(chunks)
            and reorder_tags[i + 1] == IdentityLookup.REORDER_TIME
        ):
            # collect the whole time phrase starting from "from"
            span, next_i = _collect_forward_time_span(chunks, reorder_tags, i)
            units.append(("time", span))
            i = next_i
            continue

        # if the chunk is a negation word, store it as a negation unit
        if tag == IdentityLookup.REORDER_NEGATION:
            units.append(("negation", [chunk]))
            i += 1
            continue

        # if the chunk is a WH-question word, store it as a WH unit
        if tag == IdentityLookup.REORDER_WH:
            units.append(("wh", [chunk]))
            i += 1
            continue

        # default case: keep the chunk as normal middle content
        units.append(("middle", [chunk]))
        i += 1

    reordered = []  # final reordered chunks

    # first add all time units
    for unit_type, unit_chunks in units:
        if unit_type == "time":
            reordered.extend(unit_chunks)
    # then add all normal middle content
    for unit_type, unit_chunks in units:
        if unit_type == "middle":
            reordered.extend(unit_chunks)
    # next add all negation units
    for unit_type, unit_chunks in units:
        if unit_type == "negation":
            reordered.extend(unit_chunks)
    # finally add all WH-question units
    for unit_type, unit_chunks in units:
        if unit_type == "wh":
            reordered.extend(unit_chunks)

    return reordered