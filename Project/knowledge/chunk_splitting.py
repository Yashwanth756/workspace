"""Cuts the documents into numbered sections, and the sections into passages,
never cutting a sentence in two."""
import re

from config import config


def _single_spaced(text):
    return re.sub(config.WHITESPACE_RUN_PATTERN, " ", text).strip()


def split_documents_into_sections(document_entries):
    """Divide each document at its section headings and drop its header line.

    Each document carries exactly one header line matched by
    config.DOCUMENT_HEADER_PATTERN, which is removed, and sections numbered
    1, 2, 3 and on without a gap; a document that does not is refused. The
    text before the second heading belongs to the first section.
    """
    section_entries = []
    for document in document_entries:
        document_name = document["document_name"]
        text = document["document_text"]
        headers = re.findall(config.DOCUMENT_HEADER_PATTERN, text)
        if len(headers) != 1:
            raise ValueError(f"{document_name} carries {len(headers)} header lines where it "
                             f"carries exactly one")
        text = re.sub(config.DOCUMENT_HEADER_PATTERN, " ", text, count=1)

        headings = list(re.finditer(config.SECTION_HEADING_PATTERN, text))
        numbers = [int(heading.group(1)) for heading in headings]
        if not numbers or numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"{document_name} has its sections numbered {numbers} where they "
                             f"run 1, 2, 3 and on without a gap")
        boundaries = [0] + [heading.start() for heading in headings[1:]] + [len(text)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            section_entries.append({"document_name": document_name,
                                    "section_text": _single_spaced(text[start:end])})
    return section_entries


def _split_sentences(section_text):
    return [sentence for sentence in re.split(config.SENTENCE_BOUNDARY_PATTERN, section_text)
            if sentence.strip()]


def _joined_length(sentences):
    return len(" ".join(sentences))


def _overlap_for(passage, next_sentence):
    """The passage's last whole sentences that fit within the overlap and leave
    room for the next sentence."""
    carried = []
    for sentence in reversed(passage):
        candidate = [sentence] + carried
        if (_joined_length(candidate) > config.CHUNK_OVERLAP_CHARACTERS
                or _joined_length(candidate + [next_sentence]) > config.CHUNK_SIZE_CHARACTERS):
            break
        carried = candidate
    return carried


def build_chunk_records(section_entries):
    """Cut each section into overlapping passages of about config.CHUNK_SIZE_CHARACTERS.

    Each passage holds chunk_id (the document name, a hyphen and the passage's
    number within that document, from 0), chunk_text and document_name.
    """
    chunk_records = []
    passage_numbers = {}

    def store(document_name, sentences):
        number = passage_numbers.get(document_name, 0)
        passage_numbers[document_name] = number + 1
        chunk_records.append({"chunk_id": f"{document_name}-{number}",
                              "chunk_text": " ".join(sentences),
                              "document_name": document_name})

    for section in section_entries:
        document_name = section["document_name"]
        passage = []
        for sentence in _split_sentences(section["section_text"]):
            if not passage or _joined_length(passage + [sentence]) <= config.CHUNK_SIZE_CHARACTERS:
                passage.append(sentence)
                continue
            store(document_name, passage)
            passage = _overlap_for(passage, sentence) + [sentence]
        if passage:
            store(document_name, passage)
    return chunk_records