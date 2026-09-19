"""Supporting facts shared by recursive language-model calls."""


class Context:
    def __init__(self) -> None:
        self.supporting_facts: list[str] = []

    def get_supporting_facts(self) -> list[str]:
        return list(self.supporting_facts)

    def set_supporting_facts(self, texts: list[str]) -> None:
        self.supporting_facts = list(texts)
