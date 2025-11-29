from flask import Flask, render_template, request
from core.resume_parser import ResumeParser
from core.skill_extractor import SkillExtractor
from core.embeddings import get_embedding
from core.scorer import cosine_similarity, compute_score

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        jd_text = request.form.get("jd_text", "").strip()
        files = request.files.getlist("resumes")

        # Basic validation
        if not jd_text:
            return render_template("upload.html", error="Please paste a job description.")
        if not files or all(f.filename == "" for f in files):
            return render_template("upload.html", error="Please upload at least one résumé file.")

        # Compute embedding for JD once
        jd_emb = get_embedding(jd_text)
        candidates = []

        for f in files:
            if f.filename == "":
                continue
            if not allowed_file(f.filename):
                # Skip unsupported file types silently; you can also collect errors
                continue

            # Parse resume text
            parsed_text = ResumeParser.parse(f, f.filename)
            if not parsed_text or not parsed_text.strip():
                continue

            # Extract features
            skills = SkillExtractor.extract_skills(parsed_text)
            exp_years = SkillExtractor.extract_experience_years(parsed_text)

            # Embedding for resume
            res_emb = get_embedding(parsed_text)
            sim = cosine_similarity(jd_emb, res_emb)

            # Build feature dict for scoring
            features = {
                "similarity": sim,
                "skill_count": len(skills),
                "experience_years": exp_years,
            }
            score = compute_score(features)

            candidates.append({
                "filename": f.filename,
                "skills": skills,
                "similarity": sim,
                "score": score,
                "experience_years": exp_years,
            })

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return render_template(
            "results.html",
            candidates=candidates,
            jd_len=len(jd_text),
        )

    # GET request
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
