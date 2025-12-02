from typing import List, Dict, Set

# Simple keyword-based domain map
DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "Data Science / ML": {
        "machine learning", "deep learning", "data science", "data analysis",
        "pandas", "numpy", "scikit-learn", "sklearn", "keras", "pytorch",
        "regression", "classification", "clustering", "nlp",
        "natural language processing", "computer vision", "time series"
    },
    "Web Development": {
        "html", "css", "javascript", "js", "react", "angular", "vue",
        "django", "flask", "php", "laravel", "node", "nodejs", "express",
        "rest api", "api development", "web development", "full stack",
        "frontend", "backend"
    },
    "Mobile Development": {
        "android", "kotlin", "java", "flutter", "react native", "ios", "swift"
    },
    "DevOps / Cloud": {
        "docker", "kubernetes", "k8s", "aws", "azure", "gcp",
        "ci/cd", "cicd", "jenkins", "linux", "shell scripting",
        "terraform", "ansible", "cloud"
    },
    "Software / Backend": {
        "java", "c++", "c#", "spring", "spring boot", ".net",
        "oop", "object oriented", "microservices", "sql", "databases"
    }
}


def infer_domain(skills: List[str], text: str) -> str:
    """
    Infer a coarse domain for the candidate based on skills + raw text.
    Falls back to 'GENERAL / OTHER' if nothing is strong.
    """
    text_l = text.lower()
    skill_set = {s.lower() for s in skills}

    scores: Dict[str, int] = {d: 0 for d in DOMAIN_KEYWORDS.keys()}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l or kw in skill_set:
                scores[domain] += 1

    # Find best domain by max score
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score == 0:
        return "GENERAL / OTHER"
    return best_domain


def compute_skill_gaps(jd_skills: Set[str], cand_skills: Set[str]) -> Dict[str, List[str]]:
    """
    Compute:
    - matched skills
    - missing skills (required in JD, not in candidate)
    - extra skills (candidate has, JD didn't mention)
    """
    jd_skills_l = {s.lower() for s in jd_skills}
    cand_skills_l = {s.lower() for s in cand_skills}

    matched = sorted(jd_skills_l & cand_skills_l)
    missing = sorted(jd_skills_l - cand_skills_l)
    extra = sorted(cand_skills_l - jd_skills_l)

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }


def infer_seniority(exp_years: float, text: str) -> str:
    """
    Infer a rough seniority level:
    - INTERN
    - JUNIOR
    - MID-LEVEL
    - SENIOR
    Based on years of experience and keywords in the resume text.
    """
    t = text.lower()

    # Hard override: clearly an intern
    if "intern" in t or "internship" in t:
        return "INTERN"

    # Keyword-based senior clues
    senior_words = ["senior", "lead", "leader", "architect", "principal", "staff engineer"]
    if any(w in t for w in senior_words) or exp_years >= 5:
        return "SENIOR"

    if exp_years >= 2:
        return "MID-LEVEL"

    # Default for freshers / <2 years
    return "JUNIOR"
