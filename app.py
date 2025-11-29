from flask import Flask, render_template, request
from core.resume_parser import ResumeParser
from core.skill_extractor import SkillExtractor
from core.embeddings import get_embedding
from core.scorer import cosine_similarity, compute_score

app = Flask(__name__)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf", "docx"}


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        jd_text = request.form.get("jd_text", "").strip()
        files = request.files.getlist("resumes")

        if not jd_text or not files:
            return render_template("upload.html", error="Please provide JD and at least one resume.")

        jd_emb = get_embedding(jd_text)
        candidates = []

        for f in files:
            if f.filename == "" or not allowed_file(f.filename):
                continue

            # parse resume text
            parsed_text = ResumeParser.parse(f, f.filename)
            if not parsed_text:
                continue

            # extract skills
            skills = SkillExtractor.extract_skills(parsed_text)

            # embeddings and similarity
            res_emb = get_embedding(parsed_text)
            sim = cosine_similarity(jd_emb, res_emb)

            # features for scoring
            features = {
                "similarity": sim,
                "skill_count": len(skills),
                # later: add experience years, etc.
            }
            score = compute_score(features)

            candidates.append({
                "filename": f.filename,
                "skills": skills,
                "similarity": sim,
                "score": score,
            })

        # sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return render_template(
            "results.html",
            candidates=candidates,
            jd_len=len(jd_text)
        )

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
