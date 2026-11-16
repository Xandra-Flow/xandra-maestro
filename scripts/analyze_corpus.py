import os
import json
from datetime import datetime, timezone
from huggingface_hub import HfApi, upload_file
from datasets import load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN_OWNER") or os.environ.get("HF_TOKEN")
STATE_REPO = os.environ.get("HF_REPO", "smart-force/xandra-training-state")

REPOS_TO_SCAN = [
    {"id": "smart-force/xandra-arabic-corpus", "label": "المستودع العربي السيادي (Arabic Corpus)"},
    {"id": "smart-force/xandra-tech-corpus", "label": "المستودع التقني والبرمجي (Tech Corpus)"},
    {"id": "smart-force/xandra-sft-corpus", "label": "مستودع الحوارات والضبط الدقيق (SFT Corpus)"}
]

DOMAIN_KEYWORDS = {
    "البرمجة وهندسة البرمجيات (Programming & Software)": ["def ", "class ", "import ", "function", "github", "api", "برمجة", "خوارزمية", "كود", "python", "javascript", "sql"],
    "الاقتصاد والإدارة والمالية (Economics & Business)": ["اقتصاد", "مالية", "استثمار", "سوق", "بنك", "تجارة", "economy", "finance", "market", "business", "investment"],
    "العلوم والهندسة والرياضيات (STEM & Engineering)": ["فيزياء", "كيمياء", "رياضيات", "هندسة", "معادلة", "physics", "math", "engineering", "algorithm", "quantum", "theorem"],
    "الطب والصحة والأحياء (Medicine & Health)": ["طب", "علاج", "مرض", "دواء", "جراحة", "خلايا", "medicine", "clinical", "patient", "disease", "biology", "health"],
    "الشريعة والتراث واللغة العربية (Islamic Studies & Arabic)": ["حديث", "فقه", "تفسير", "نحو", "بلاغة", "أدب", "شعر", "القرآن", "حدثنا", "قال ابن"],
    "التاريخ والجغرافيا والموسوعات (History & General Knowledge)": ["تاريخ", "مدينة", "قرن", "دولة", "حرب", "جغرافيا", "history", "century", "empire", "geography"]
}

def classify_text(text):
    t_lower = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in t_lower:
                scores[domain] += 1
    best_domain = max(scores, key=scores.get)
    return best_domain if scores[best_domain] > 0 else "معارف عامة ومتنوعة (General Knowledge)"

def format_size(size_bytes):
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    return f"{size_bytes / (1024**2):.2f} MB"

def main():
    api = HfApi(token=HF_TOKEN)
    report = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_size_bytes": 0,
        "total_files": 0,
        "total_estimated_docs": 0,
        "repositories": [],
        "domains_breakdown": {}
    }

    domain_counts = {}
    sample_total = 0

    for repo_info in REPOS_TO_SCAN:
        repo_id = repo_info["id"]
        print(f"🔍 فحص المستودع: {repo_id}...")
        try:
            tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True, token=HF_TOKEN)
            repo_bytes = 0
            data_files = 0

            for item in tree:
                if hasattr(item, "size") and item.size:
                    path = item.path
                    if path.startswith(".git"):
                        continue
                    repo_bytes += item.size
                    ext = os.path.splitext(path).lower() or "other"
                    if ext in (".parquet", ".jsonl", ".json", ".txt", ".csv", ".md"):
                        data_files += 1

            est_docs = int(repo_bytes / 2500)

            try:
                ds = load_dataset(repo_id, split="train", streaming=True, token=HF_TOKEN)
                for idx, row in enumerate(ds):
                    if idx >= 400:
                        break
                    txt = str(row.get("text") or row.get("content") or row.get("instruction") or "")
                    if len(txt) > 30:
                        dom = classify_text(txt[:1500])
                        domain_counts[dom] = domain_counts.get(dom, 0) + 1
                        sample_total += 1
            except Exception as e:
                print(f"   ⚠️ تعذر أخذ عينة نصية من {repo_id}: {e}")

            report["total_size_bytes"] += repo_bytes
            report["total_files"] += data_files
            report["total_estimated_docs"] += est_docs

            report["repositories"].append({
                "id": repo_id,
                "label": repo_info["label"],
                "size_bytes": repo_bytes,
                "size_formatted": format_size(repo_bytes),
                "data_files_count": data_files,
                "estimated_docs": est_docs
            })
        except Exception as e:
            print(f"❌ فشل فحص {repo_id}: {e}")

    report["total_size_formatted"] = format_size(report["total_size_bytes"])

    for dom, cnt in sorted(domain_counts.items(), key=lambda x: -x):
        pct = round((cnt / sample_total) * 100, 1) if sample_total > 0 else 0
        est_dom_bytes = int(report["total_size_bytes"] * (pct / 100.0))
        est_dom_docs = int(report["total_estimated_docs"] * (pct / 100.0))
        report["domains_breakdown"][dom] = {
            "percentage": pct,
            "estimated_size": format_size(est_dom_bytes),
            "estimated_docs": est_dom_docs
        }

    with open("corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    upload_file(
        path_or_fileobj="corpus_stats.json",
        path_in_repo="corpus_stats.json",
        repo_id=STATE_REPO,
        repo_type="model",
        token=HF_TOKEN
    )
    print("✅ تم رفع تقرير المعرفة corpus_stats.json بنجاح!")

if __name__ == "__main__":
    main()
