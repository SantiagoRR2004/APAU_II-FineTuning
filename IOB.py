import sys
from typing import Generator, List, Tuple


class IOB:
    def __init__(self, sep: str = " ") -> None:
        """
        Initialize the IOB parser.

        Args:
            - sep (str): The separator used in the IOB format. Default is a space.

        Returns:
            - None
        """
        self._sep = sep

    def parse_file(self, ifile: str) -> List[List[Tuple[str, str]]]:
        """
        Parse the IOB file and return a list of sentences.
        Each sentence is a list of tuples, where each tuple contains the token
        and its label (if available).

        Args:
            - ifile (str): The input file path.

        Returns:
            - list: A list of sentences, where each sentence is a list of tuples.
        """
        return [
            self._parse_sentence(raw) for raw in self._read_sentences_from_file(ifile)
        ]

    def _parse_sentence(self, raw_sentence: str) -> List[Tuple[str, str]]:
        """
        Parse a raw sentence string into a list of tuples.
        Each tuple contains the token and its label (if available).

        Args:
            - raw_sentence (str): The raw sentence string.

        Returns:
            - list: A list of tuples, where each tuple contains the token and its label.
        """
        return [
            tuple(
                token.split(self._sep),
            )
            for token in raw_sentence.strip().split("\n")
        ]

    def _read_sentences_from_file(self, ifile: str) -> Generator[str, None, None]:
        """
        Read sentences from an IOB file.

        Args:
            ifile (str): The input file path.

        Yields:
            str: Raw sentence strings separated by blank lines.
        """
        raw_sentence = ""
        try:
            with open(ifile, encoding="utf-8") as fhi:
                for line in fhi:
                    if line == "\n":
                        if raw_sentence == "":
                            continue
                        yield raw_sentence
                        raw_sentence = ""
                        continue

                    if line:
                        raw_sentence += line

            if raw_sentence:
                yield raw_sentence

        except IOError:
            print("Unable to read file: " + ifile)
            sys.exit()
