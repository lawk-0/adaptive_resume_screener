import spacy
import json
from typing import List

# load spaCy model
nlp = spacy.load("en_core_web_sm")

# load skills dictionary
try:
    with open("data/skills_dict.json", "r", encoding="utf-8") as f:
        SKILLS = set(json.load(f))
except Exception:
    SKILLS = set()


class SkillExtractor:
    @staticmethod
    def extract_skills(text: str) -> List[str]:
        doc = nlp(text.lower())
        found = set()

        # simple token-based matching
        for token in doc:
            if token.text in SKILLS:
                found.add(token.text)

        # you can improve later with phrase matching
        return sorted(found)

    @staticmethod
    def extract_experience_years(text: str) -> float:
        # TODO: implement pattern like "X years"
        # for now, returning dummy value
        return 0.0
