import re
import spacy
import json
from typing import List
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Load single-token skills
# ---------------------------
try:
    with open("data/skills_dict.json", "r", encoding="utf-8") as f:
        SKILLS = set(json.load(f))
except Exception:
    SKILLS = set()

# ---------------------------
# Load phrase-level skills
# ---------------------------
try:
    with open("data/skills_phrases.json", "r", encoding="utf-8") as f:
        SKILL_PHRASES = json.load(f)
except Exception:
    SKILL_PHRASES = []

# Build a global PhraseMatcher for phrases
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for phrase in SKILL_PHRASES:
    phrase_matcher.add("SKILL_PHRASE", [nlp.make_doc(phrase)])


class SkillExtractor:
    @staticmethod
    def extract_skills(text: str) -> List[str]:
        """
        Extract skills from resume text using:
        - token-level dictionary (SKILLS)
        - phrase-level dictionary (SKILL_PHRASES with PhraseMatcher)
        """
        text_l = text.lower()
        doc = nlp(text_l)
        found = set()

        # Token-level matching
        for token in doc:
            if token.text in SKILLS:
                found.add(token.text)

        # Phrase-level matching
        matches = phrase_matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            phrase_text = span.text.strip().lower()
            found.add(phrase_text)

        return sorted(found)

    @staticmethod
    def extract_experience_years(text: str) -> float:
        """
        Extracts numeric experience from phrases like:
        - "2 years"
        - "3+ years"
        - "5 yrs"
        - "experience of 4 years"

        Returns the highest detected experience value.
        """
        text_lower = text.lower()

        patterns = [
            r"(\d+)\s*\+?\s*years",
            r"(\d+)\s*\+?\s*yrs",
            r"experience of\s+(\d+)\s*years",
        ]

        years = 0
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                try:
                    val = int(m)
                    if val > years:
                        years = val
                except ValueError:
                    continue

        return float(years)

    @staticmethod
    def extract_education(text: str) -> str:
        """
        Detects the highest (or first matched) degree from the resume text.

        Returns a short degree code like:
        - "PHD"
        - "MTECH"
        - "MBA"
        - "BTECH"
        - "BE"
        - "BSC"
        - "MSC"
        - "MCA"
        - "DIPLOMA"
        or "UNKNOWN" if nothing is found.
        """
        text_l = text.lower()

        patterns = {
            "PHD":    ["phd", "ph.d", "doctorate"],
            "MTECH":  ["m.tech", "mtech", "master of technology"],
            "MSC":    ["m.sc", "msc", "master of science"],
            "MBA":    ["mba", "master of business administration"],
            "MCA":    ["mca", "master of computer applications"],
            "BE":     ["b.e", "be", "bachelor of engineering"],
            "BTECH":  ["b.tech", "btech", "bachelor of technology"],
            "BSC":    ["b.sc", "bsc", "bachelor of science"],
            "DIPLOMA":["diploma"]
        }

        for degree_code, keywords in patterns.items():
            for kw in keywords:
                if kw in text_l:
                    return degree_code

        return "UNKNOWN"
