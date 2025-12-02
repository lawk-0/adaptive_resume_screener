from flask import Flask, render_template, request, abort
from core.resume_parser import ResumeParser
from core.skill_extractor import SkillExtractor
from core.embeddings import get_embedding
from core.scorer import cosine_similarity, compute_score
from core.utils import infer_domain, compute_skill_gaps, infer_seniority

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx"}

# Store last screening results in memory (for candidate detail view)
LAST_RESULTS = []
LAST_JD_TEXT = ""
LAST_JD_SKILLS = []


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload():
    global LAST_RESULTS, LAST_JD_TEXT, LAST_JD_SKILLS

    if request.method == "POST":
        jd_text = request.form.get("jd_text", "").strip()
        files = request.files.getlist("resumes")

        # Basic validation
        if not jd_text:
            return render_template("upload.html", error="Please paste a job description.")
        if not files or all(f.filename == "" for f in files):
            return render_template("upload.html", error="Please upload at least one résumé file.")

        LAST_JD_TEXT = jd_text

        # Embedding and skills for JD
        jd_emb = get_embedding(jd_text)
        jd_skills = SkillExtractor.extract_skills(jd_text)
        LAST_JD_SKILLS = jd_skills

        candidates = []

        for f in files:
            if f.filename == "":
                continue
            if not allowed_file(f.filename):
                # skip unsupported type
                continue

            # Parse resume
            parsed_text = ResumeParser.parse(f, f.filename)
            if not parsed_text or not parsed_text.strip():
                continue

            # Extract NLP features
            skills = SkillExtractor.extract_skills(parsed_text)
            exp_years = SkillExtractor.extract_experience_years(parsed_text)
            education = SkillExtractor.extract_education(parsed_text)

            # Embedding + similarity
            res_emb = get_embedding(parsed_text)
            sim = cosine_similarity(jd_emb, res_emb)

            # Feature dict for scoring
            features = {
                "similarity": sim,
                "skill_count": len(skills),
                "experience_years": exp_years,
            }
            score = compute_score(features)

            # Domain, seniority & skill gap analysis
            domain = infer_domain(skills, parsed_text)
            seniority = infer_seniority(exp_years, parsed_text)
            gaps = compute_skill_gaps(set(jd_skills), set(skills))

            candidates.append({
                "filename": f.filename,
                "skills": skills,
                "similarity": sim,
                "score": score,
                "experience_years": exp_years,
                "education": education,
                "domain": domain,
                "seniority": seniority,
                "matched_skills": gaps["matched"],
                "missing_skills": gaps["missing"],
                "extra_skills": gaps["extra"],
                # store raw text if you want to show in detail view:
                "raw_text": parsed_text,
            })

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Save for detail view
        LAST_RESULTS = candidates

        return render_template(
            "results.html",
            candidates=candidates,
            jd_len=len(jd_text),
            jd_skills=jd_skills,
        )

    # GET request
    return render_template("upload.html")


@app.route("/candidate/<int:idx>")
def candidate_detail(idx: int):
    """
    Show a detailed view of a single candidate from LAST_RESULTS.
    idx is 0-based index (we'll pass loop.index0 from template).
    """
    global LAST_RESULTS, LAST_JD_TEXT, LAST_JD_SKILLS

    if not LAST_RESULTS:
        # No previous screening done
        abort(404)

    if idx < 0 or idx >= len(LAST_RESULTS):
        abort(404)

    cand = LAST_RESULTS[idx]

    return render_template(
        "candidate_detail.html",
        candidate=cand,
        idx=idx,
        jd_text=LAST_JD_TEXT,
        jd_skills=LAST_JD_SKILLS,
    )


if __name__ == "__main__":
    app.run(debug=True)
