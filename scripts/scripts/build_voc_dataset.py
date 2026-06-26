import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
SOURCE_INPUTS = DATASET_ROOT / "source_inputs"
PUBLIC_INPUT = SOURCE_INPUTS / "E-commerce product review dataset-20260625T012504Z-3-001" / "E-commerce product review dataset"
UTC = timezone.utc
RNG = random.Random(73217)
BATCH_FULL = "synthetic_clouddesk_full_v1_2026_06_25"
BATCH_PILOT = "synthetic_clouddesk_pilot_v1_2026_06_25"
SCHEMA_VERSION = "voc_schema_v1"
PROMPT_VERSION = "deterministic_template_v1"
TARGET_FULL_COUNTS = {
    "accounts": 500,
    "customers": 1250,
    "opportunities": 2000,
    "support": 3000,
    "sales": 1000,
    "reviews": 1000,
    "churn": 300,
    "features": 500,
    "interviews": 200,
    "github_issues": 500,
    "usage": 1000,
}
TARGET_PILOT_COUNTS = {
    "accounts": 25,
    "customers": 50,
    "opportunities": 25,
    "support": 50,
    "sales": 20,
    "reviews": 25,
    "churn": 15,
    "features": 20,
    "interviews": 10,
    "github_issues": 20,
    "usage": 50,
}


PRODUCT_AREAS = [
    "onboarding",
    "permissions",
    "notifications",
    "reporting",
    "integrations",
    "mobile_application",
    "billing",
    "workspace_management",
    "search",
    "customer_support",
]

TOPICS = {
    "onboarding": ["sso_configuration", "bulk_user_import", "workspace_setup", "data_migration", "onboarding_documentation", "administrator_training"],
    "permissions": ["role_templates", "permission_inheritance", "guest_access", "audit_logs", "client_workspace_permissions"],
    "notifications": ["email_notifications", "push_notifications", "notification_preferences", "time_zone_handling", "delayed_notifications", "duplicate_notifications"],
    "reporting": ["csv_export", "pdf_export", "scheduled_reports", "dashboard_customization", "large_dataset_performance", "report_permissions"],
    "integrations": ["slack", "microsoft_teams", "google_drive", "jira", "salesforce"],
    "mobile_application": ["offline_task_viewing", "offline_editing", "delayed_synchronization", "mobile_search", "push_delivery"],
    "billing": ["plan_limits", "feature_gating", "invoice_clarity", "required_upgrades", "seat_pricing"],
    "workspace_management": ["workspace_switching", "client_context", "archive_restore", "workspace_templates", "cross_workspace_search"],
    "search": ["exact_match_dependency", "filtering", "archived_tasks", "attachment_search", "relevance_ranking"],
    "customer_support": ["response_time", "account_manager_help", "workarounds", "escalation_quality", "troubleshooting_clarity"],
}

SEGMENTS = ["individual", "small_business", "mid_market", "enterprise", "agency"]
INDUSTRIES = ["software", "marketing", "consulting", "construction", "healthcare", "education", "financial_services", "retail", "manufacturing"]
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
COUNTRIES = ["US", "CA", "GB", "DE", "AU", "BR", "IN", "FR"]
ROLES = ["administrator", "project_manager", "operations_lead", "designer", "developer", "finance_manager", "customer_success_manager", "executive"]
PLANS = ["starter", "team", "business", "enterprise"]


def now_iso():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso(dt):
    return dt.replace(tzinfo=UTC, microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs():
    dirs = [
        "data/manifests", "data/synthetic/reference", "data/synthetic/raw", "data/synthetic/structured",
        "data/public/raw", "data/public/normalized", "data/public/quarantined", "data/public/reference",
        "data/processed", "data/evaluation", "reports", "schemas", "docs",
    ]
    for d in dirs:
        (DATASET_ROOT / d).mkdir(parents=True, exist_ok=True)


def stable_hash(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deterministic_id(prefix, *parts):
    return f"{prefix}_{stable_hash('|'.join(map(str, parts)))[:16]}"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def extract_docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    parts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml)
    return "".join(html.unescape(p) for p in parts)


def clean_text(text):
    text = text or ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def language(text):
    text = text or ""
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return "en" if ascii_ratio > 0.85 else "unknown"


def product_rules(filename, text, title=""):
    blob = f"{title} {text}".lower()
    if "echo" in filename:
        target = "Amazon Echo Dot"
        wrong = any(x in blob for x in ["echo show", "echo studio", "fire tv", "kindle"])
        direct = any(x in blob for x in ["echo dot", "dot"])
        family = "Echo"
        model = "Echo Dot" if direct else ("Alexa/Echo unspecified" if "alexa" in blob or "echo" in blob else None)
        if wrong:
            return target, "Amazon", family, model, None, None, False, 0.2, "Mentions a different Echo/Amazon device family.", "quarantined_wrong_product"
        if direct:
            return target, "Amazon", family, "Echo Dot", None, None, True, 0.9, "Direct Echo Dot mention.", "accepted"
        if "alexa" in blob or "echo" in blob:
            return target, "Amazon", family, model, None, None, False, 0.45, "General Alexa/Echo mention without Echo Dot identity.", "quarantined_ambiguous_product"
        return target, None, None, None, None, None, False, 0.05, "No target product evidence.", "quarantined_off_topic"
    if "sony" in filename:
        target = "Sony WH-1000XM5"
        if any(x in blob for x in ["xm6", "1000xm6", "wh-1000xm6"]):
            return target, "Sony", "WH headphones", "WH-1000XM6", "XM6", None, False, 0.1, "Mentions XM6 rather than XM5.", "quarantined_wrong_product"
        if any(x in blob for x in ["wf-1000xm5", "wf1000xm5"]):
            return target, "Sony", "WF earbuds", "WF-1000XM5", "XM5", None, False, 0.15, "Mentions WF earbuds, not WH headphones.", "quarantined_wrong_product"
        if any(x in blob for x in ["wh-1000xm5", "wh1000xm5", "xm5"]):
            return target, "Sony", "WH headphones", "WH-1000XM5", "XM5", None, True, 0.88, "XM5/WH-1000XM5 mention.", "accepted"
        if "sony" in blob or "headphone" in blob:
            return target, "Sony", "headphones", None, None, None, False, 0.4, "Sony headphone discussion without clear XM5 identity.", "quarantined_ambiguous_product"
        return target, None, None, None, None, None, False, 0.05, "No target product evidence.", "quarantined_off_topic"
    target = "Wyze Cam v3"
    if any(x in blob for x in ["v4", "cam pan", "doorbell"]):
        return target, "Wyze", "camera", "other Wyze camera", None, None, False, 0.2, "Mentions another Wyze camera product.", "quarantined_wrong_product"
    if any(x in blob for x in ["wyze cam v3", "cam v3", "v3"]):
        return target, "Wyze", "Wyze Cam", "Wyze Cam v3", "v3", None, True, 0.9, "Direct Wyze Cam v3 mention.", "accepted"
    if "wyze" in blob or "camera" in blob:
        return target, "Wyze", "camera", None, None, None, False, 0.42, "Wyze/camera mention without clear v3 identity.", "quarantined_ambiguous_product"
    return target, None, None, None, None, None, False, 0.05, "No target product evidence.", "quarantined_off_topic"


def classify_claim(text):
    blob = (text or "").lower()
    if len(blob) < 35 or re.search(r"\b(lol|haha|meme|joke)\b", blob):
        return "humor"
    if "?" in blob and not any(x in blob for x in ["i have", "my ", "we "]):
        return "question"
    if any(x in blob for x in ["my ", "i have", "i bought", "we use", "we had", "our "]):
        return "firsthand"
    if any(x in blob for x in ["friend", "people say", "someone"]):
        return "secondhand"
    return "unknown"


def direct_permalink(url):
    return bool(re.search(r"reddit\.com/r/[^/]+/comments/[^/]+/.+", url or ""))


def generic_url(url):
    return bool(re.match(r"https?://(www\.)?reddit\.com/r/[^/]+/?$", url or ""))


def audit_public_csvs():
    audits = []
    normalized = []
    quarantined = []
    md = ["# Existing Data Audit", "", "Generated at: " + now_iso(), ""]
    for path in sorted(PUBLIC_INPUT.glob("*.csv")):
        rows = read_csv(path)
        fields = list(rows[0].keys()) if rows else []
        texts = [clean_text((r.get("title") or "") + " " + (r.get("body") or "")) for r in rows]
        lengths = [len(t) for t in texts]
        hashes = [stable_hash(t) for t in texts]
        exact_dups = sum(c - 1 for c in Counter(hashes).values() if c > 1)
        near_keys = [re.sub(r"[^a-z0-9 ]", "", t.lower())[:140] for t in texts]
        near_dups = sum(c - 1 for c in Counter(near_keys).values() if c > 1 and c != len(rows))
        dates = []
        for r in rows:
            try:
                dates.append(datetime.fromisoformat((r.get("createdAt") or "").replace("Z", "+00:00")))
            except Exception:
                pass
        product_counts = Counter()
        claim_counts = Counter()
        missing_parent = 0
        languages = Counter()
        for r, t in zip(rows, texts):
            target, brand, family, model, gen, variant, is_target, conf, reason, status = product_rules(path.name, r.get("body", ""), r.get("title", ""))
            product_counts[status] += 1
            claim = classify_claim(t)
            claim_counts[claim] += 1
            languages[language(t)] += 1
            if r.get("dataType") == "comment" and not direct_permalink(r.get("url")):
                missing_parent += 1
            fid = deterministic_id("pubfb", path.name, r.get("id", ""), stable_hash(t)[:10])
            item = {
                "feedback_id": fid,
                "source_id": deterministic_id("source", path.name),
                "source_name": "Reddit CSV sample",
                "source_type": "reddit_discussion",
                "source_external_id": r.get("id"),
                "thread_id": ((r.get("url") or "").split("/comments/")[1].split("/")[0] if "/comments/" in (r.get("url") or "") else None),
                "root_post_id": None,
                "parent_id": None,
                "author_hash": stable_hash(r.get("username", ""))[:24] if r.get("username") else None,
                "title": r.get("title"),
                "raw_text": r.get("body"),
                "clean_text": t,
                "cleaning_actions": ["unicode_nfkc", "html_removed", "whitespace_normalized"],
                "rating": None,
                "helpful_votes": None,
                "engagement_score": None,
                "verified_purchase": None,
                "target_product": target,
                "detected_brand": brand,
                "detected_product_family": family,
                "detected_product_model": model,
                "detected_generation": gen,
                "detected_variant": variant,
                "is_target_product": is_target,
                "product_match_confidence": conf,
                "product_match_reason": reason,
                "product_version": None,
                "firmware_version": None,
                "application_version": None,
                "device_platform": None,
                "country": None,
                "language": language(t),
                "created_at": r.get("createdAt"),
                "edited_at": None,
                "fetched_at": None,
                "source_url": r.get("url"),
                "scraper_version": "provided_csv_v1",
                "raw_content_hash": stable_hash(r),
                "normalized_content_hash": stable_hash(t),
                "content_hash": stable_hash(t),
                "validation_status": status,
                "data_origin": "scraped",
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
            }
            (normalized if status == "accepted" else quarantined).append(item)
        missing = {f: round(100 * sum(1 for r in rows if not (r.get(f) or "").strip()) / max(1, len(rows)), 2) for f in fields}
        audit = {
            "file_name": path.name,
            "record_count": len(rows),
            "column_names": fields,
            "missing_value_percentages": missing,
            "date_range": {
                "min": iso(min(dates)) if dates else None,
                "max": iso(max(dates)) if dates else None,
            },
            "duplicate_count": exact_dups + near_dups,
            "exact_duplicate_count": exact_dups,
            "near_duplicate_count": near_dups,
            "unique_urls": len({r.get("url") for r in rows if r.get("url")}),
            "generic_urls": sum(1 for r in rows if generic_url(r.get("url"))),
            "direct_post_or_comment_permalinks": sum(1 for r in rows if direct_permalink(r.get("url")) or "/comments/" in (r.get("url") or "")),
            "records_matching_target_product": product_counts["accepted"],
            "records_mentioning_another_product": product_counts["quarantined_wrong_product"],
            "likely_jokes_offtopic_questions_non_customer": claim_counts["humor"] + claim_counts["question"] + product_counts["quarantined_off_topic"],
            "firsthand_experiences": claim_counts["firsthand"],
            "secondhand_claims": claim_counts["secondhand"],
            "records_missing_parent_context": missing_parent,
            "language_distribution": dict(languages),
            "average_text_length": round(statistics.mean(lengths), 2) if lengths else 0,
            "median_text_length": statistics.median(lengths) if lengths else 0,
            "validation_status_counts": dict(product_counts),
            "claim_type_counts": dict(claim_counts),
        }
        audits.append(audit)
        md += [
            f"## {path.name}",
            f"- Records: {audit['record_count']}",
            f"- Columns: {', '.join(fields)}",
            f"- Date range: {audit['date_range']['min']} to {audit['date_range']['max']}",
            f"- Duplicates: {audit['duplicate_count']} ({audit['exact_duplicate_count']} exact, {audit['near_duplicate_count']} near)",
            f"- URLs: {audit['unique_urls']} unique, {audit['generic_urls']} generic, {audit['direct_post_or_comment_permalinks']} direct post/comment permalinks",
            f"- Product identity: {audit['records_matching_target_product']} accepted, {audit['records_mentioning_another_product']} wrong product, {audit['validation_status_counts']}",
            f"- Claim quality: {audit['firsthand_experiences']} firsthand, {audit['secondhand_claims']} secondhand, {audit['likely_jokes_offtopic_questions_non_customer']} jokes/off-topic/questions/non-customer",
            f"- Missing parent context: {audit['records_missing_parent_context']}",
            f"- Language distribution: {audit['language_distribution']}",
            f"- Text length: avg {audit['average_text_length']}, median {audit['median_text_length']}",
            f"- Missing values: `{json.dumps(missing, sort_keys=True)}`",
            "",
        ]
    write_json(DATASET_ROOT / "reports/existing_data_audit.json", {"generated_at": now_iso(), "audits": audits})
    (DATASET_ROOT / "reports/existing_data_audit.md").write_text("\n".join(md), encoding="utf-8")
    write_jsonl(DATASET_ROOT / "data/public/normalized/feedback_items.jsonl", normalized)
    write_jsonl(DATASET_ROOT / "data/public/quarantined/feedback_items.jsonl", quarantined)
    return audits, normalized, quarantined


def reference_artifacts():
    taxonomy = []
    for area, subs in TOPICS.items():
        root_id = f"topic_{area}"
        taxonomy.append({
            "topic_id": root_id, "topic_name": area, "parent_topic_id": None, "description": f"Top-level CloudDesk area for {area.replace('_', ' ')} feedback.",
            "product_area": area, "example_phrases": [area.replace("_", " ")], "excluded_meanings": ["General sentiment without an actionable customer signal."],
        })
        for sub in subs:
            taxonomy.append({
                "topic_id": f"topic_{sub}",
                "topic_name": sub,
                "parent_topic_id": root_id,
                "description": f"Customer feedback about {sub.replace('_', ' ')}.",
                "product_area": area,
                "example_phrases": [sub.replace("_", " "), sub.split("_")[0]],
                "excluded_meanings": ["Unrelated use of the same words outside CloudDesk workflows."],
            })
    profile = {
        "company_name": "CloudDesk",
        "company_description": "Fictional B2B SaaS company for reproducible Voice-of-Customer evaluation.",
        "product_description": "A project-management and collaboration platform for planning work, coordinating teams, tracking client projects, and reporting delivery progress.",
        "main_customer_jobs": ["plan projects", "assign tasks", "coordinate clients", "report delivery status", "manage permissions", "connect work tools"],
        "features": ["task boards", "project timelines", "client workspaces", "SSO", "bulk import", "notifications", "CSV/PDF reporting", "mobile app", "Slack/Teams/Jira/Salesforce/Drive integrations"],
        "customer_segments": SEGMENTS,
        "industries_served": INDUSTRIES,
        "product_areas": PRODUCT_AREAS,
        "product_limitations": ["large exports may be slow", "offline mobile editing is limited", "complex enterprise onboarding can require services", "search favors exact matches"],
        "integrations": ["Slack", "Microsoft Teams", "Google Drive", "Jira", "Salesforce"],
        "support_model": {"starter": "email", "team": "email and chat", "business": "priority chat", "enterprise": "dedicated CSM and onboarding services"},
        "known_hidden_customer_problems": ["enterprise_onboarding", "agency_workspace_switching", "mobile_offline_access", "release_3_2_notification_regression", "smb_pricing_concern", "large_export_failure", "search_complaints", "integration_reliability"],
        "expected_positive_feedback": ["fast support", "clear workarounds", "useful collaboration", "helpful account managers"],
        "expected_counterevidence": ["paid onboarding reduces enterprise issues", "some SMBs are unaffected by SSO", "constant-connectivity mobile users rarely request offline mode"],
        "business_terminology": ["ACV", "MRR", "renewal", "opportunity", "churn", "customer health", "implementation", "workspace"],
    }
    pricing = [
        {"plan_id": "starter", "name": "Starter", "monthly_price_per_user": 8, "target_segments": ["individual", "small_business"], "limits": ["5 projects", "basic reports"], "support_plan": "email"},
        {"plan_id": "team", "name": "Team", "monthly_price_per_user": 14, "target_segments": ["small_business", "agency"], "limits": ["25 projects", "standard integrations"], "support_plan": "email_chat"},
        {"plan_id": "business", "name": "Business", "monthly_price_per_user": 24, "target_segments": ["mid_market", "agency"], "limits": ["advanced reporting", "SSO add-on"], "support_plan": "priority"},
        {"plan_id": "enterprise", "name": "Enterprise", "monthly_price_per_user": None, "target_segments": ["enterprise"], "limits": ["contracted seats", "dedicated onboarding"], "support_plan": "dedicated_csm"},
    ]
    releases = []
    start = datetime(2024, 1, 15, tzinfo=UTC)
    for i in range(26):
        version = f"3.{i//4}.{i%4}"
        date = start + timedelta(days=i * 31)
        known = ["Duplicate notifications can occur across time zones."] if version == "3.2.0" else []
        areas = ["notifications"] if version == "3.2.0" else RNG.sample(PRODUCT_AREAS, 2)
        releases.append({
            "release_id": f"rel_{version.replace('.', '_')}",
            "version": version,
            "release_date": iso(date),
            "title": f"CloudDesk {version}",
            "release_notes": f"CloudDesk {version} improved {', '.join(areas)} workflows.",
            "features_added": [f"{areas[0].replace('_', ' ')} enhancements"],
            "bugs_fixed": [f"Minor {areas[-1].replace('_', ' ')} fixes"],
            "known_issues": known,
            "affected_product_areas": areas,
        })
    hidden = {
        "retrieval_index_policy": "Ground-truth pattern definitions must be excluded from normal RAG indexing.",
        "patterns": [
            {"pattern_id": "enterprise_onboarding", "description": "Enterprise administrators struggle with SSO, bulk user import, permission setup, and outdated docs.", "segments": ["enterprise"], "channels": ["support", "sales", "churn", "feature_request", "interview"], "counterevidence": ["paid onboarding lowers complaints", "some enterprises complete setup without support"]},
            {"pattern_id": "agency_workspace_switching", "description": "Agencies struggle switching client workspaces and sometimes post in the wrong context.", "segments": ["agency"], "counterevidence": ["workspace templates reduce confusion"]},
            {"pattern_id": "mobile_offline_access", "description": "Mobile users request offline task viewing/editing and delayed sync.", "segments": SEGMENTS, "counterevidence": ["always-connected users rarely mention it", "some only need read-only offline mode"]},
            {"pattern_id": "release_3_2_notification_regression", "description": "Notification failures, timezone scheduling problems, and duplicates increase after 3.2.0.", "release_id": "rel_3_2_0", "counterevidence": ["not all regions or integrations are affected"]},
            {"pattern_id": "smb_pricing_concern", "description": "Small businesses report plan limits, feature gating, confusing pricing, and required upgrades.", "segments": ["small_business"]},
            {"pattern_id": "large_export_failure", "description": "Large CSV exports time out, freeze, or become incomplete among mid-market and enterprise accounts.", "segments": ["mid_market", "enterprise"]},
            {"pattern_id": "positive_support_experience", "description": "Some customers praise fast support, helpful account managers, and effective workarounds."},
            {"pattern_id": "search_complaints", "description": "Search depends on exact matches and struggles with filters and archived tasks."},
            {"pattern_id": "integration_reliability", "description": "Slack is mostly reliable, Teams has intermittent notification mapping, Google Drive has file permission confusion, Jira sync is mixed, Salesforce is more brittle."},
            {"pattern_id": "insufficient_evidence", "description": "No meaningful evidence is created for cryptocurrency payments, VR workspaces, drone integrations, or blockchain identity.", "reserved_for_abstention": True},
        ],
    }
    write_json(DATASET_ROOT / "data/synthetic/reference/company_profile.json", profile)
    write_json(DATASET_ROOT / "data/synthetic/reference/product_taxonomy.json", {"taxonomy_version": "clouddesk_taxonomy_v1", "topics": taxonomy})
    write_json(DATASET_ROOT / "data/synthetic/reference/pricing_plans.json", {"pricing_version": "clouddesk_pricing_v1", "plans": pricing})
    write_json(DATASET_ROOT / "data/synthetic/reference/release_history.json", {"release_history_version": "clouddesk_releases_v1", "releases": releases})
    write_json(DATASET_ROOT / "data/synthetic/reference/releases.json", {"release_history_version": "clouddesk_releases_v1", "releases": releases})
    write_json(DATASET_ROOT / "data/synthetic/reference/hidden_patterns.json", hidden)
    docs = []
    for i in range(125):
        area = PRODUCT_AREAS[i % len(PRODUCT_AREAS)]
        version = "3.0.0" if i % 13 == 0 else "3.2.0"
        docs.append({
            "document_section_id": f"docsec_{i+1:03d}",
            "document_title": f"CloudDesk {area.replace('_', ' ').title()} Guide",
            "section_title": f"{TOPICS[area][i % len(TOPICS[area])].replace('_', ' ').title()}",
            "content": f"Use this section to configure {area.replace('_', ' ')}. Some screenshots may refer to older navigation in versions before {version}.",
            "product_area": area,
            "applicable_plans": RNG.sample(PLANS, RNG.randint(2, 4)),
            "applicable_versions": [version, "3.3.0"],
            "published_at": iso(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 5)),
            "updated_at": iso(datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i * 3)),
            "source_type": "synthetic_documentation",
            "data_origin": "synthetic",
            "generation_batch_id": BATCH_FULL,
        })
    write_jsonl(DATASET_ROOT / "data/synthetic/reference/product_documentation.jsonl", docs)
    return releases


def account_name(i):
    a = ["Northstar", "Bluefield", "Keystone", "Juniper", "Riverbend", "Metroline", "Clearpath", "Summit", "Harbor", "Brightlane"]
    b = ["Studios", "Systems", "Partners", "Works", "Group", "Labs", "Services", "Collective", "Solutions", "Digital"]
    return f"{a[i % len(a)]} {b[(i * 7) % len(b)]} {i:03d}"


def plan_for_segment(segment):
    return {
        "individual": "starter",
        "small_business": RNG.choice(["starter", "team"]),
        "agency": RNG.choice(["team", "business"]),
        "mid_market": RNG.choice(["business", "enterprise"]),
        "enterprise": "enterprise",
    }[segment]


def generate_accounts(n):
    rows = []
    weights = [0.18, 0.26, 0.22, 0.18, 0.16]
    start = datetime(2023, 1, 1, tzinfo=UTC)
    for i in range(n):
        seg = RNG.choices(SEGMENTS, weights)[0]
        plan = plan_for_segment(seg)
        employees = {
            "individual": RNG.randint(1, 3),
            "small_business": RNG.randint(5, 80),
            "agency": RNG.randint(15, 250),
            "mid_market": RNG.randint(150, 1200),
            "enterprise": RNG.randint(1200, 18000),
        }[seg]
        mrr = {"starter": 40, "team": 350, "business": 1900, "enterprise": 12000}[plan] * RNG.uniform(0.65, 1.6)
        created = start + timedelta(days=RNG.randint(0, 850))
        contract_start = created + timedelta(days=RNG.randint(0, 60))
        status = RNG.choices(["active", "active", "active", "churned", "trial"], [78, 0, 0, 14, 8])[0]
        churn_date = contract_start + timedelta(days=RNG.randint(120, 700)) if status == "churned" else None
        rows.append({
            "account_id": f"acct_{i+1:04d}",
            "company_name": account_name(i),
            "segment": seg,
            "industry": RNG.choice(INDUSTRIES),
            "employee_count": employees,
            "subscription_plan": plan,
            "annual_contract_value": round(mrr * 12, 2),
            "monthly_recurring_revenue": round(mrr, 2),
            "country": RNG.choice(COUNTRIES),
            "region": RNG.choice(REGIONS),
            "created_at": iso(created),
            "contract_start_date": iso(contract_start),
            "contract_end_date": iso(contract_start + timedelta(days=365)),
            "account_status": status,
            "churn_status": "churned" if status == "churned" else "not_churned",
            "churn_date": iso(churn_date) if churn_date else "",
            "health_score": RNG.randint(28, 96) if status != "churned" else RNG.randint(12, 55),
            "customer_success_owner": f"csm_{RNG.randint(1, 18):02d}" if plan in ["business", "enterprise"] else "",
            "support_plan": {"starter": "email", "team": "email_chat", "business": "priority", "enterprise": "dedicated"}[plan],
        })
    return rows


def generate_customers(accounts, n):
    rows = []
    assigned = list(accounts[: min(len(accounts), n)])
    while len(assigned) < n:
        assigned.append(RNG.choice(accounts))
    for i, acct in enumerate(assigned):
        role = RNG.choice(ROLES)
        rows.append({
            "customer_id": f"cust_{i+1:05d}",
            "account_id": acct["account_id"],
            "customer_role": role,
            "department": RNG.choice(["operations", "product", "sales", "customer_success", "finance", "it", "creative"]),
            "seniority": RNG.choice(["individual_contributor", "manager", "director", "vp", "executive"]),
            "country": acct["country"],
            "language": "en",
            "is_admin": role in ["administrator", "operations_lead", "executive"],
            "created_at": acct["created_at"],
        })
    return rows


def choose_pattern(acct, date):
    seg = acct["segment"]
    options = ["positive_support_experience", "search_complaints", "integration_reliability", "mobile_offline_access"]
    if seg == "enterprise":
        options += ["enterprise_onboarding"] * 4
    if seg == "agency":
        options += ["agency_workspace_switching"] * 4
    if seg == "small_business":
        options += ["smb_pricing_concern"] * 3
    if seg in ["mid_market", "enterprise"]:
        options += ["large_export_failure"] * 3
    if date >= datetime(2024, 9, 1, tzinfo=UTC):
        options += ["release_3_2_notification_regression"] * 4
    return RNG.choice(options)


def pattern_area_topic(pattern):
    return {
        "enterprise_onboarding": ("onboarding", RNG.choice(["sso_configuration", "bulk_user_import", "onboarding_documentation", "role_templates"])),
        "agency_workspace_switching": ("workspace_management", RNG.choice(["workspace_switching", "client_context", "client_workspace_permissions"])),
        "mobile_offline_access": ("mobile_application", RNG.choice(["offline_task_viewing", "offline_editing", "delayed_synchronization"])),
        "release_3_2_notification_regression": ("notifications", RNG.choice(["delayed_notifications", "duplicate_notifications", "time_zone_handling"])),
        "smb_pricing_concern": ("billing", RNG.choice(["plan_limits", "feature_gating", "required_upgrades", "invoice_clarity"])),
        "large_export_failure": ("reporting", RNG.choice(["csv_export", "large_dataset_performance", "scheduled_reports"])),
        "positive_support_experience": ("customer_support", RNG.choice(["response_time", "account_manager_help", "workarounds"])),
        "search_complaints": ("search", RNG.choice(["exact_match_dependency", "filtering", "archived_tasks"])),
        "integration_reliability": ("integrations", RNG.choice(["slack", "microsoft_teams", "google_drive", "jira", "salesforce"])),
    }[pattern]


def make_statement(pattern, channel, positive=False):
    templates = {
        "enterprise_onboarding": [
            "Our admin team got stuck connecting SSO and then had to invite people in small batches.",
            "The permissions page looked simple until we tried to map it to our enterprise roles.",
            "The onboarding article still shows the old SSO screen, so setup took another support call.",
        ],
        "agency_workspace_switching": [
            "Switching client workspaces is too easy to miss, and one update landed in the wrong client project.",
            "Our account managers need a clearer client context before posting comments.",
        ],
        "mobile_offline_access": [
            "The mobile app is fine online, but field staff cannot view tasks when the train connection drops.",
            "Offline editing would save us because notes are getting pasted in later from another app.",
        ],
        "release_3_2_notification_regression": [
            "Since the recent update, due-date reminders arrive late for users outside our main time zone.",
            "Several people received the same task notification twice while others missed it completely.",
        ],
        "smb_pricing_concern": [
            "We only need one reporting feature, but it appears to require an upgrade to the next plan.",
            "The plan limits are hard to explain to our owner because the upgrade trigger is not obvious.",
        ],
        "large_export_failure": [
            "CSV exports for the full quarter freeze before the file finishes downloading.",
            "The export completes but several rows are missing when we include archived projects.",
        ],
        "positive_support_experience": [
            "Support answered quickly and gave us a workaround that kept the project moving.",
            "Our account manager explained the issue clearly and followed up after the fix.",
        ],
        "search_complaints": [
            "Search only finds the task if I remember the exact wording.",
            "Archived tasks are hard to find even when I filter by the right client.",
        ],
        "integration_reliability": [
            "Slack updates are reliable, but the Teams mapping sometimes drops the project name.",
            "The Salesforce sync needs retries after custom fields change.",
        ],
    }
    text = RNG.choice(templates[pattern])
    if channel == "support" and RNG.random() < 0.25:
        text += " We tried clearing cache and reauthorizing, but it did not change the result."
    if positive and pattern != "positive_support_experience":
        text += " To be fair, a smaller pilot workspace did not show the same problem."
    return text


def date_for_index(i, post_release_bias=False):
    start = datetime(2024, 2, 1, tzinfo=UTC)
    if post_release_bias and RNG.random() < 0.7:
        start = datetime(2024, 9, 1, tzinfo=UTC)
        return start + timedelta(days=RNG.randint(0, 460))
    return start + timedelta(days=RNG.randint(0, 720))


def release_for_date(releases, dt):
    candidates = [r for r in releases if datetime.fromisoformat(r["release_date"].replace("Z", "+00:00")) <= dt]
    return candidates[-1] if candidates else releases[0]


def linked_customer(customers_by_account, account_id):
    return RNG.choice(customers_by_account[account_id])


def add_hash(row):
    copy = {k: v for k, v in row.items() if k != "content_hash"}
    row["content_hash"] = stable_hash(copy)
    return row


def generate_synthetic(n_accounts=500, n_customers=1250, scale="full", releases=None):
    releases = releases or reference_artifacts()
    batch = BATCH_FULL if scale == "full" else BATCH_PILOT
    counts = TARGET_FULL_COUNTS if scale == "full" else TARGET_PILOT_COUNTS
    accounts = generate_accounts(n_accounts)
    customers = generate_customers(accounts, n_customers)
    by_acct = defaultdict(list)
    for c in customers:
        by_acct[c["account_id"]].append(c)
    opportunities = []
    for i in range(counts["opportunities"]):
        acct = RNG.choice(accounts)
        opportunities.append({
            "opportunity_id": f"opp_{i+1:04d}",
            "account_id": acct["account_id"],
            "deal_stage": RNG.choice(["discovery", "security_review", "proposal", "procurement", "closed_won", "closed_lost"]),
            "estimated_deal_value": round(acct["annual_contract_value"] * RNG.uniform(0.8, 1.5), 2),
            "created_at": iso(date_for_index(i)),
        })
    support = []
    for i in range(counts["support"]):
        acct = RNG.choice(accounts)
        cust = linked_customer(by_acct, acct["account_id"])
        dt = date_for_index(i, True)
        rel = release_for_date(releases, dt)
        pattern = choose_pattern(acct, dt)
        area, topic = pattern_area_topic(pattern)
        resolved = dt + timedelta(minutes=RNG.randint(40, 9000))
        desc = make_statement(pattern, "support")
        if RNG.random() < 0.12:
            desc += " Support was helpful, but we still need a product fix."
        row = {
            "ticket_id": f"ticket_{scale}_{i+1:05d}", "account_id": acct["account_id"], "customer_id": cust["customer_id"],
            "subject": f"{topic.replace('_', ' ').title()} issue", "description": desc,
            "conversation_messages": [{"role": "customer", "message": desc}, {"role": "agent", "message": "Thanks, we are checking the account settings and recent release notes."}],
            "ticket_category": area, "priority": RNG.choice(["low", "medium", "high", "urgent"]), "status": RNG.choice(["resolved", "resolved", "open", "pending"]),
            "channel": RNG.choice(["email", "chat", "portal"]), "created_at": iso(dt), "first_response_at": iso(dt + timedelta(minutes=RNG.randint(5, 240))),
            "resolved_at": iso(resolved), "resolution_minutes": int((resolved - dt).total_seconds() // 60),
            "resolution_summary": RNG.choice(["configuration corrected", "workaround provided", "bug escalated", "documentation link sent"]),
            "support_agent_id": f"agent_{RNG.randint(1, 30):02d}", "product_version": rel["version"], "related_release_id": rel["release_id"],
            "customer_satisfaction_score": RNG.choice([2, 3, 4, 5, None]), "escalated": RNG.random() < 0.18, "reopened": RNG.random() < 0.08,
            "data_origin": "synthetic", "generation_batch_id": batch, "pattern_id": pattern, "schema_version": SCHEMA_VERSION,
        }
        support.append(add_hash(row))
    sales = []
    for i in range(counts["sales"]):
        acct = RNG.choice(accounts)
        dt = date_for_index(i, True)
        pattern = choose_pattern(acct, dt)
        area, _ = pattern_area_topic(pattern)
        delayed = RNG.random() < (0.5 if pattern in ["enterprise_onboarding", "large_export_failure", "integration_reliability"] else 0.2)
        row = {
            "call_excerpt_id": f"call_{scale}_{i+1:05d}", "account_id": acct["account_id"], "opportunity_id": RNG.choice(opportunities)["opportunity_id"],
            "customer_role": RNG.choice(ROLES), "deal_stage": RNG.choice(["discovery", "technical_validation", "proposal", "procurement"]),
            "deal_status": RNG.choice(["open", "closed_won", "closed_lost"]), "estimated_deal_value": round(acct["annual_contract_value"] * RNG.uniform(0.5, 2.0), 2),
            "competitors_considered": RNG.sample(["Asana", "Monday", "ClickUp", "Smartsheet", "Jira"], RNG.randint(0, 2)),
            "call_date": iso(dt), "excerpt": make_statement(pattern, "sales"),
            "objection_type": RNG.choice(["security", "migration", "pricing", "integration", "reporting", "none"]),
            "product_area": area, "deal_blocked": delayed and RNG.random() < 0.3, "deal_delayed": delayed, "loss_reason": "missing capability" if delayed and RNG.random() < 0.25 else None,
            "data_origin": "synthetic", "generation_batch_id": batch, "pattern_id": pattern, "schema_version": SCHEMA_VERSION,
        }
        sales.append(add_hash(row))
    reviews = []
    for i in range(counts["reviews"]):
        acct = RNG.choice(accounts); cust = linked_customer(by_acct, acct["account_id"]); dt = date_for_index(i, True)
        pattern = choose_pattern(acct, dt); rel = release_for_date(releases, dt)
        rating = 5 if pattern == "positive_support_experience" else RNG.choice([2, 3, 3, 4])
        row = {
            "review_id": f"review_{scale}_{i+1:05d}", "account_id": acct["account_id"], "customer_id": cust["customer_id"],
            "review_source": RNG.choice(["G2", "Capterra", "in_app_nps"]), "rating": rating,
            "title": RNG.choice(["Useful but has rough edges", "Good team workspace", "Needs attention", "Solid support"]),
            "review_text": make_statement(pattern, "review", positive=rating >= 4),
            "helpful_votes": RNG.randint(0, 31), "verified_customer": True, "created_at": iso(dt), "product_version": rel["version"],
            "data_origin": "synthetic", "generation_batch_id": batch, "pattern_id": pattern, "schema_version": SCHEMA_VERSION,
        }
        reviews.append(add_hash(row))
    churn = []
    churn_accounts = [a for a in accounts if a["churn_status"] == "churned"] or accounts[:counts["churn"]]
    for i in range(counts["churn"]):
        acct = RNG.choice(churn_accounts); cust = linked_customer(by_acct, acct["account_id"]); dt = date_for_index(i, True)
        pattern = choose_pattern(acct, dt); area, _ = pattern_area_topic(pattern)
        row = {
            "churn_feedback_id": f"churn_{scale}_{i+1:05d}", "account_id": acct["account_id"], "customer_id": cust["customer_id"],
            "churn_date": acct["churn_date"] or iso(dt + timedelta(days=60)), "primary_reason": RNG.choice(["budget", "missing_capability", "implementation_risk", "internal_reorg"]),
            "secondary_reasons": RNG.sample(["pricing", "support", "reporting", "onboarding", "usage"], 2),
            "comment": make_statement(pattern, "churn") + " This was one factor, not the only reason.",
            "competitor_selected": RNG.choice(["Asana", "Monday", "ClickUp", "none", None]), "avoidable": RNG.random() < 0.55,
            "annual_revenue_lost": round(float(acct["annual_contract_value"]) * RNG.uniform(0.7, 1.0), 2),
            "product_area": area, "issue_relationship": RNG.choice(["mentioned_issue", "contributing_factor", "primary_churn_reason", "unknown_relationship"]),
            "data_origin": "synthetic", "generation_batch_id": batch, "pattern_id": pattern, "schema_version": SCHEMA_VERSION,
        }
        churn.append(add_hash(row))
    features = []
    for i in range(counts["features"]):
        acct = RNG.choice(accounts); cust = linked_customer(by_acct, acct["account_id"]); dt = date_for_index(i, True)
        pattern = choose_pattern(acct, dt); area, topic = pattern_area_topic(pattern)
        row = {
            "request_id": f"feat_{scale}_{i+1:05d}", "account_id": acct["account_id"], "customer_id": cust["customer_id"],
            "request_title": f"Improve {topic.replace('_', ' ')}",
            "requested_outcome": make_statement(pattern, "feature").replace("We ", "Users "),
            "problem_description": make_statement(pattern, "feature"),
            "current_workaround": RNG.choice(["manual spreadsheet", "admin support ticket", "copy notes later", "third-party automation", None]),
            "product_area": area, "request_status": RNG.choice(["new", "under_review", "planned", "closed_no_action"]),
            "votes": RNG.randint(1, 210), "created_at": iso(dt), "data_origin": "synthetic", "generation_batch_id": batch, "pattern_id": pattern, "schema_version": SCHEMA_VERSION,
        }
        features.append(add_hash(row))
    interviews = []
    for i in range(counts["interviews"]):
        acct = RNG.choice(accounts); cust = linked_customer(by_acct, acct["account_id"]); dt = date_for_index(i, True)
        pattern = choose_pattern(acct, dt); area, _ = pattern_area_topic(pattern)
        row = {
            "interview_excerpt_id": f"interview_{scale}_{i+1:05d}", "account_id": acct["account_id"], "customer_id": cust["customer_id"],
            "customer_role": cust["customer_role"], "interview_date": iso(dt), "research_topic": area,
            "question": "Walk me through the last time this workflow was difficult.",
            "response": make_statement(pattern, "interview") + " The workaround is manageable for one project but breaks down when several teams are involved.",
            "researcher_note": RNG.choice(["Probe for frequency.", "Compare with usage logs.", "Potential counterexample from same segment.", "Needs follow-up with admin."]),
            "product_area": area, "data_origin": "synthetic", "generation_batch_id": batch, "pattern_id": pattern, "schema_version": SCHEMA_VERSION,
        }
        interviews.append(add_hash(row))
    github_issues = []
    for i in range(counts["github_issues"]):
        acct = RNG.choice(accounts)
        dt = date_for_index(i, True)
        pattern = choose_pattern(acct, dt)
        area, topic = pattern_area_topic(pattern)
        body = make_statement(pattern, "github_issue")
        row = {
            "github_issue_id": f"ghissue_{scale}_{i+1:05d}",
            "repository": "clouddesk-labs/clouddesk-demo",
            "issue_number": i + 1,
            "title": f"{topic.replace('_', ' ').title()} regression or limitation",
            "body": body + " Reproduction steps and screenshots are omitted because this is synthetic issue-tracker evidence.",
            "comments": [
                {"author_role": "maintainer", "comment": "Thanks for the report. We are linking this to the product area and checking related support volume."},
                {"author_role": "customer", "comment": make_statement(pattern, "github_comment", positive=RNG.random() < 0.18)},
            ],
            "labels": [area, topic, RNG.choice(["bug", "enhancement", "needs-triage", "customer-reported"])],
            "state": RNG.choice(["open", "closed", "open", "triaged"]),
            "milestone": RNG.choice(["3.3 stabilization", "enterprise onboarding", "mobile reliability", None]),
            "created_at": iso(dt),
            "closed_at": iso(dt + timedelta(days=RNG.randint(3, 90))) if RNG.random() < 0.35 else None,
            "product_area": area,
            "account_id": acct["account_id"],
            "customer_id": None,
            "pattern_id": pattern,
            "source_url": None,
            "data_origin": "synthetic",
            "generation_batch_id": batch,
            "schema_version": SCHEMA_VERSION,
        }
        github_issues.append(add_hash(row))
    usage = []
    for i in range(counts["usage"]):
        acct = RNG.choice(accounts); dt = date_for_index(i, True); rel = release_for_date(releases, dt)
        enterprise = acct["segment"] in ["enterprise", "mid_market"]
        post = dt >= datetime(2024, 9, 1, tzinfo=UTC)
        row = {
            "usage_summary_id": f"usage_{scale}_{i+1:05d}", "account_id": acct["account_id"], "period_start": iso(dt), "period_end": iso(dt + timedelta(days=30)),
            "active_users": RNG.randint(1, max(3, int(acct["employee_count"]) // 4)), "invited_users": RNG.randint(1, max(5, int(acct["employee_count"]) // 2)),
            "completed_onboarding": RNG.random() > (0.35 if acct["segment"] == "enterprise" else 0.12), "sso_enabled": acct["subscription_plan"] in ["business", "enterprise"] and RNG.random() < 0.75,
            "bulk_import_attempts": RNG.randint(0, 6 if enterprise else 2), "bulk_import_failures": RNG.randint(0, 3 if enterprise else 1),
            "csv_export_attempts": RNG.randint(0, 80), "csv_export_failures": RNG.randint(0, 10 if enterprise else 3),
            "mobile_sessions": RNG.randint(0, 500), "notification_delivery_rate": round(RNG.uniform(0.88 if post else 0.94, 0.999), 4),
            "integration_error_count": RNG.randint(0, 18), "support_ticket_count": RNG.randint(0, 9), "product_version": rel["version"],
            "data_origin": "synthetic", "generation_batch_id": batch, "schema_version": SCHEMA_VERSION,
        }
        usage.append(row)
    return {"accounts": accounts, "customers": customers, "opportunities": opportunities, "support": support, "sales": sales, "reviews": reviews, "churn": churn, "features": features, "interviews": interviews, "github_issues": github_issues, "usage": usage}


def persist_synthetic(bundle, prefix=""):
    write_csv(DATASET_ROOT / f"data/synthetic/structured/{prefix}accounts.csv", bundle["accounts"], list(bundle["accounts"][0]))
    write_csv(DATASET_ROOT / f"data/synthetic/structured/{prefix}customers.csv", bundle["customers"], list(bundle["customers"][0]))
    write_csv(DATASET_ROOT / f"data/synthetic/structured/{prefix}opportunities.csv", bundle["opportunities"], list(bundle["opportunities"][0]))
    write_jsonl(DATASET_ROOT / f"data/synthetic/raw/{prefix}support_tickets.jsonl", bundle["support"])
    write_jsonl(DATASET_ROOT / f"data/synthetic/raw/{prefix}sales_call_excerpts.jsonl", bundle["sales"])
    write_jsonl(DATASET_ROOT / f"data/synthetic/raw/{prefix}product_reviews.jsonl", bundle["reviews"])
    write_jsonl(DATASET_ROOT / f"data/synthetic/raw/{prefix}churn_feedback.jsonl", bundle["churn"])
    write_jsonl(DATASET_ROOT / f"data/synthetic/raw/{prefix}feature_requests.jsonl", bundle["features"])
    write_jsonl(DATASET_ROOT / f"data/synthetic/raw/{prefix}customer_interviews.jsonl", bundle["interviews"])
    write_jsonl(DATASET_ROOT / f"data/synthetic/raw/{prefix}github_issues.jsonl", bundle["github_issues"])
    write_jsonl(DATASET_ROOT / f"data/synthetic/structured/{prefix}product_usage_summaries.jsonl", bundle["usage"])


def feedback_items_from_synthetic(bundle):
    items = []
    channel_map = [
        ("support", "ticket_id", "description", "support_ticket"),
        ("sales", "call_excerpt_id", "excerpt", "sales_call"),
        ("reviews", "review_id", "review_text", "product_review"),
        ("churn", "churn_feedback_id", "comment", "churn_feedback"),
        ("features", "request_id", "problem_description", "feature_request"),
        ("interviews", "interview_excerpt_id", "response", "customer_interview"),
        ("github_issues", "github_issue_id", "body", "github_issue"),
    ]
    accounts = {a["account_id"]: a for a in bundle["accounts"]}
    for key, id_field, text_field, source_type in channel_map:
        for r in bundle[key]:
            text = r.get(text_field) or r.get("requested_outcome") or ""
            items.append({
                "feedback_id": deterministic_id("fb", source_type, r[id_field]),
                "source_id": deterministic_id("source", "synthetic", source_type),
                "source_name": "CloudDesk synthetic corpus",
                "source_type": source_type,
                "source_external_id": r[id_field],
                "raw_text": text,
                "clean_text": clean_text(text),
                "account_id": r.get("account_id"),
                "customer_id": r.get("customer_id"),
                "customer_segment": accounts[r["account_id"]]["segment"],
                "product_version": r.get("product_version"),
                "created_at": r.get("created_at") or r.get("call_date") or r.get("churn_date") or r.get("interview_date"),
                "product_area": r.get("product_area") or r.get("ticket_category"),
                "pattern_id": r.get("pattern_id"),
                "data_origin": "synthetic",
                "generation_batch_id": r["generation_batch_id"],
                "content_hash": stable_hash(text),
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
            })
    return items


def extract_atoms(items, limit=None):
    atoms = []
    for item in (items[:limit] if limit else items):
        text = item["clean_text"]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 12]
        for j, sent in enumerate(sentences[:3]):
            area = item.get("product_area") or "other"
            pattern = item.get("pattern_id") or ""
            if pattern:
                area, topic = pattern_area_topic(pattern)
            else:
                topic = area
            lower = sent.lower()
            if any(x in lower for x in ["helpful", "quickly", "great", "reliable", "solid"]):
                ftype, sentiment, severity, role = "praise", "positive", "low", "opposing" if pattern and pattern != "positive_support_experience" else "supporting"
            elif "?" in sent:
                ftype, sentiment, severity, role = "purchase_question", "neutral", "low", "contextual"
            elif any(x in lower for x in ["upgrade", "price", "plan", "subscription"]):
                ftype, sentiment, severity, role = "pricing_complaint", "negative", "medium", "supporting"
            elif any(x in lower for x in ["request", "need", "would save", "cannot view", "offline"]):
                ftype, sentiment, severity, role = "feature_request", "negative", "medium", "supporting"
            else:
                ftype, sentiment, severity, role = "usability_problem", "negative", RNG.choice(["medium", "high"]), "supporting"
            start = text.find(sent)
            atoms.append({
                "atom_id": deterministic_id("atom", item["feedback_id"], j, stable_hash(sent)[:8]),
                "feedback_id": item["feedback_id"],
                "statement": sent,
                "feedback_type": ftype,
                "product_area": area,
                "topic": topic,
                "subtopic": topic,
                "sentiment": sentiment,
                "severity": severity,
                "customer_impact": "Workflow delay or administrative effort." if sentiment != "positive" else "Reduced support burden or increased trust.",
                "root_cause": "See linked source evidence; root cause is inferred, not final.",
                "requested_outcome": "Complete the workflow without extra support." if ftype == "feature_request" else None,
                "workaround": "manual process" if "workaround" in lower else None,
                "competitor_mentioned": None,
                "product_version": item.get("product_version"),
                "customer_segment": item.get("customer_segment"),
                "evidence_role": role,
                "is_firsthand_experience": True,
                "claim_type": "firsthand",
                "extraction_confidence": 0.72 if item.get("data_origin") == "synthetic" else 0.52,
                "source_span_start": start,
                "source_span_end": start + len(sent) if start >= 0 else None,
                "extractor_model": "deterministic_rules_v1",
                "extraction_prompt_version": PROMPT_VERSION,
                "extraction_input_hash": stable_hash(item["clean_text"]),
                "human_review_status": "unreviewed_suggestion",
            })
    return atoms


def clusters(atoms):
    rows = []
    for a in atoms:
        rows.append({
            "incident_cluster_id": deterministic_id("incident", a["product_area"], a["topic"], a.get("customer_segment"), a["statement"][:35]),
            "duplicate_cluster_id": deterministic_id("dup", a["statement"].lower()[:80]),
            "atom_id": a["atom_id"],
            "feedback_id": a["feedback_id"],
            "thread_id": None,
            "is_original_report": True,
            "is_independent_customer_report": True,
        })
    return rows


def evaluation_files(atoms, items):
    diverse = atoms[:220]
    gold = []
    for a in diverse:
        gold.append({
            "feedback_id": a["feedback_id"],
            "expected_atoms": [a["statement"]],
            "expected_feedback_types": [a["feedback_type"]],
            "expected_product_areas": [a["product_area"]],
            "expected_topics": [a["topic"]],
            "expected_sentiments": [a["sentiment"]],
            "expected_severities": [a["severity"]],
            "expected_claim_types": [a["claim_type"]],
            "expected_product_identity": "CloudDesk synthetic",
            "annotator_id": None,
            "annotation_status": "unverified_suggestion_pending_human_review",
            "annotator_notes": "Machine-generated suggestion. Keep outside normal vector index.",
        })
    write_jsonl(DATASET_ROOT / "data/evaluation/gold_atom_annotations.jsonl", gold)
    questions = [
        ("enterprise_onboarding", "Why are enterprise administrators struggling with onboarding?", "root_cause", False),
        ("large_export_failure", "Which reporting problems are most common among mid-market and enterprise accounts?", "segment_comparison", False),
        ("release_3_2_notification_regression", "Did notification complaints increase after version 3.2.0?", "release_impact", False),
        ("enterprise_onboarding", "What evidence contradicts the claim that all enterprise customers struggle with SSO?", "counterevidence", False),
        ("integration_reliability", "Which customer complaints are associated with delayed or lost sales opportunities?", "revenue_impact", False),
        ("insufficient_evidence", "What do customers think about cryptocurrency payments?", "no_answer", True),
    ]
    by_pattern = defaultdict(list)
    for a in atoms:
        for pid in ["enterprise_onboarding", "large_export_failure", "release_3_2_notification_regression", "integration_reliability", "smb_pricing_concern", "search_complaints"]:
            if pid.split("_")[0] in a["topic"] or pid in a["atom_id"]:
                by_pattern[pid].append(a)
    retrieval = []
    expanded = []
    for i in range(130):
        base = questions[i % len(questions)]
        expanded.append(base)
    for i, (pid, q, qtype, abstain) in enumerate(expanded):
        expected = [] if abstain else atoms[(i * 7) % max(1, len(atoms)):(i * 7) % max(1, len(atoms)) + 4]
        retrieval.append({
            "question_id": f"q_{i+1:03d}", "question": q if i < len(questions) else f"{q} ({i+1})",
            "question_type": qtype, "filters": {"customer_segment": ["enterprise"] if "enterprise" in q else None},
            "required_topics": [] if abstain else [e["topic"] for e in expected[:2]],
            "optional_topics": [] if abstain else [e["topic"] for e in expected[2:]],
            "forbidden_topics": ["cryptocurrency_payments", "virtual_reality_workspaces", "drone_integrations", "blockchain_identity"] if abstain else [],
            "expected_segments": [] if abstain else list({e.get("customer_segment") for e in expected if e.get("customer_segment")}),
            "expected_products": ["CloudDesk"],
            "expected_atom_ids": [e["atom_id"] for e in expected],
            "expected_feedback_ids": list({e["feedback_id"] for e in expected}),
            "expected_structured_metrics": {"requires_sql_counts": qtype in ["frequency", "release_impact", "revenue_impact"]},
            "expected_answer_summary": "Abstain due to intentionally absent evidence." if abstain else "Answer must cite linked atoms, include segment/context, and separate evidence from recommendation.",
            "must_include_counterevidence": qtype in ["counterevidence", "root_cause", "release_impact"],
            "should_abstain": abstain,
            "difficulty": RNG.choice(["easy", "medium", "hard"]),
            "review_status": "unverified_suggestion_pending_human_review",
        })
    write_jsonl(DATASET_ROOT / "data/evaluation/retrieval_questions.jsonl", retrieval)
    write_jsonl(DATASET_ROOT / "data/evaluation/abstention_questions.jsonl", [r for r in retrieval if r["should_abstain"]])
    cases = [{
        "release_id": "rel_3_2_0",
        "pre_release_window": {"start": "2024-06-01T00:00:00Z", "end": "2024-08-31T23:59:59Z"},
        "post_release_window": {"start": "2024-09-01T00:00:00Z", "end": "2024-11-30T23:59:59Z"},
        "expected_topics": ["delayed_notifications", "duplicate_notifications", "time_zone_handling"],
        "expected_direction": "increase",
        "expected_minimum_change": "2x mention rate",
        "affected_segments": ["enterprise", "mid_market", "agency"],
        "counterevidence": ["not all users or regions are affected"],
        "causality_warning_required": True,
    }]
    write_jsonl(DATASET_ROOT / "data/evaluation/release_impact_cases.jsonl", cases)
    rubric = {k: {"scale": "0-2", "description": k.replace("_", " ")} for k in [
        "correct_sql_counts", "correct_segment_filtering", "correct_product_filtering", "required_topic_coverage", "citation_presence",
        "citation_validity", "citation_entailment", "counterevidence_inclusion", "unsupported_claim_rate", "recommendation_fact_separation",
        "uncertainty_statement", "abstention_correctness", "duplicate_evidence_rate", "source_diversity",
    ]}
    write_json(DATASET_ROOT / "data/evaluation/answer_quality_rubric.json", {"rubric_version": "answer_quality_v1", "normal_vector_index_exclusion": True, "criteria": rubric})


def schema_files():
    common = {"data_origin": ["scraped", "api_collected", "synthetic", "manually_created", "inferred"], "validation_status": ["accepted", "quarantined_wrong_product", "quarantined_ambiguous_product", "quarantined_off_topic", "quarantined_missing_context"]}
    schemas = {
        "feedback_item.schema.json": {"required": ["feedback_id", "source_type", "raw_text", "data_origin", "content_hash"], "properties": common},
        "feedback_atom.schema.json": {"required": ["atom_id", "feedback_id", "statement", "feedback_type", "product_area", "evidence_role", "human_review_status"], "properties": common},
        "support_ticket.schema.json": {"required": ["ticket_id", "account_id", "customer_id", "description", "created_at", "data_origin", "content_hash"]},
        "retrieval_question.schema.json": {"required": ["question_id", "question", "question_type", "expected_atom_ids", "should_abstain", "review_status"]},
    }
    for name, data in schemas.items():
        write_json(DATASET_ROOT / "schemas" / name, {"$schema": "https://json-schema.org/draft/2020-12/schema", "schema_version": SCHEMA_VERSION, **data})


def public_reference_files():
    checked = now_iso()
    products = [
        {"product_id": "amazon_echo_dot", "product_name": "Amazon Echo Dot", "brand": "Amazon", "category": "smart speaker", "source_url": "provided_csv_sample", "retrieved_at": checked},
        {"product_id": "sony_wh1000xm5", "product_name": "Sony WH-1000XM5", "brand": "Sony", "category": "headphones", "source_url": "provided_csv_sample", "retrieved_at": checked},
        {"product_id": "wyze_cam_v3", "product_name": "Wyze Cam v3", "brand": "Wyze", "category": "security camera", "source_url": "provided_csv_sample", "retrieved_at": checked},
    ]
    write_csv(DATASET_ROOT / "data/public/reference/products.csv", products, list(products[0]))
    write_csv(DATASET_ROOT / "data/public/reference/product_features.csv", [{"product_id": p["product_id"], "feature_name": "unknown_pending_verified_public_source", "source_url": "not_collected", "retrieved_at": checked} for p in products])
    write_csv(DATASET_ROOT / "data/public/reference/product_versions.csv", [{"product_id": p["product_id"], "version": "unknown", "version_type": "unknown", "source_url": "not_collected", "retrieved_at": checked} for p in products])
    write_csv(DATASET_ROOT / "data/public/reference/subscription_plans.csv", [{"product_id": p["product_id"], "plan_name": "unknown_pending_verified_public_source", "source_url": "not_collected", "retrieved_at": checked} for p in products])
    write_jsonl(DATASET_ROOT / "data/public/reference/release_notes.jsonl", [])
    write_jsonl(DATASET_ROOT / "data/public/reference/documentation.jsonl", [])
    registry = []
    for p in products:
        registry.append({"source_id": deterministic_id("source", p["product_id"], "provided_csv"), "product_id": p["product_id"], "source_name": "Provided Reddit CSV sample", "source_type": "reddit_discussion", "base_url": "provided_local_csv", "collection_method": "provided_file", "api_available": "not_applicable", "terms_reviewed": "not_reviewed_current_turn", "robots_reviewed": "not_reviewed_current_turn", "rate_limit": "not_applicable", "license_notes": "User-provided local sample; publication requires review.", "collection_status": "partial", "failure_reason": "", "last_checked_at": checked})
        for source_name in ["manufacturer_support_or_docs", "app_store_reviews", "public_product_review_dataset"]:
            registry.append({"source_id": deterministic_id("source", p["product_id"], source_name), "product_id": p["product_id"], "source_name": source_name, "source_type": source_name, "base_url": "not_collected", "collection_method": "planned_lawful_source_review", "api_available": "unknown", "terms_reviewed": "not_reviewed_current_turn", "robots_reviewed": "not_reviewed_current_turn", "rate_limit": "unknown", "license_notes": "Pending lawful source review before collection.", "collection_status": "unavailable", "failure_reason": "No live collection performed in this run.", "last_checked_at": checked})
    write_csv(DATASET_ROOT / "data/public/source_registry.csv", registry, list(registry[0]))


def validate(bundle, atoms, public_normalized, public_quarantined):
    errors = []
    warnings = []
    account_ids = {a["account_id"] for a in bundle["accounts"]}
    customer_ids = {c["customer_id"] for c in bundle["customers"]}
    for group in ["support", "reviews", "features", "interviews", "churn", "github_issues"]:
        for r in bundle[group]:
            if r.get("account_id") not in account_ids:
                errors.append(f"{group}: invalid account {r.get('account_id')}")
            if r.get("customer_id") and r.get("customer_id") not in customer_ids:
                errors.append(f"{group}: invalid customer {r.get('customer_id')}")
            if r.get("data_origin") == "synthetic" and r.get("source_url"):
                errors.append(f"{group}: synthetic record has URL")
    if not atoms:
        errors.append("No feedback atoms generated.")
    if not public_normalized:
        warnings.append("No public records accepted after product identity validation.")
    for item in public_normalized:
        if item["data_origin"] in ["scraped", "api_collected"] and not item.get("source_url"):
            errors.append(f"Public record missing source_url: {item['feedback_id']}")
    eval_paths = [str(p).replace("\\", "/") for p in (DATASET_ROOT / "data/evaluation").glob("*")]
    write_json(DATASET_ROOT / "data/manifests/index_exclusion_manifest.json", {
        "excluded_from_normal_vector_index": ["data/evaluation/**", "data/synthetic/reference/hidden_patterns.json"],
        "reason": "Evaluation answer keys and hidden patterns must not be retrievable during normal RAG questions.",
    })
    write_json(DATASET_ROOT / "data/manifests/vector_index_include_manifest.json", {
        "include": ["data/processed/feedback_items.jsonl", "data/processed/feedback_atoms.jsonl", "data/synthetic/reference/product_documentation.jsonl"],
        "exclude": ["data/evaluation/**", "data/synthetic/reference/hidden_patterns.json"],
    })
    metrics = {
        "critical_error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings,
        "record_counts": {
            "accounts": len(bundle["accounts"]), "customers": len(bundle["customers"]), "support_tickets": len(bundle["support"]),
            "sales_call_excerpts": len(bundle["sales"]), "product_reviews": len(bundle["reviews"]), "churn_feedback": len(bundle["churn"]),
            "feature_requests": len(bundle["features"]), "customer_interviews": len(bundle["interviews"]), "github_issues": len(bundle["github_issues"]),
            "opportunities": len(bundle["opportunities"]), "usage_summaries": len(bundle["usage"]),
            "public_normalized": len(public_normalized), "public_quarantined": len(public_quarantined), "feedback_atoms": len(atoms),
        },
        "evaluation_leakage_check": {"excluded_paths": eval_paths, "normal_index_excludes_evaluation": True},
    }
    write_json(DATASET_ROOT / "reports/data_quality_metrics.json", metrics)
    md = ["# Data Quality Report", "", f"Generated at: {now_iso()}", "", f"Critical errors: {len(errors)}", f"Warnings: {len(warnings)}", ""]
    if errors:
        md += ["## Critical Errors"] + [f"- {e}" for e in errors[:100]] + [""]
    if warnings:
        md += ["## Warnings"] + [f"- {w}" for w in warnings] + [""]
    md += ["## Evaluation Leakage", "- `data/evaluation/**` is excluded from the normal vector index include manifest.", "- `data/synthetic/reference/hidden_patterns.json` is excluded from normal retrieval.", ""]
    md += ["## Counts", "```json", json.dumps(metrics["record_counts"], indent=2), "```", ""]
    (DATASET_ROOT / "reports/data_quality_report.md").write_text("\n".join(md), encoding="utf-8")
    return metrics


def manifests(bundle, public_normalized, public_quarantined):
    counts = {
        "accounts": len(bundle["accounts"]), "customers": len(bundle["customers"]), "support_tickets": len(bundle["support"]),
        "sales_call_excerpts": len(bundle["sales"]), "product_reviews": len(bundle["reviews"]), "churn_feedback": len(bundle["churn"]),
        "feature_requests": len(bundle["features"]), "customer_interviews": len(bundle["interviews"]), "github_issues": len(bundle["github_issues"]),
        "opportunities": len(bundle["opportunities"]), "product_usage_summaries": len(bundle["usage"]),
        "public_feedback_accepted": len(public_normalized), "public_feedback_quarantined": len(public_quarantined),
    }
    segs = Counter(a["segment"] for a in bundle["accounts"])
    dataset_manifest = {
        "dataset_version": "voc_clouddesk_public_pilot_v1",
        "created_at": now_iso(), "updated_at": now_iso(),
        "schema_versions": {"default": SCHEMA_VERSION}, "prompt_versions": {"default": PROMPT_VERSION},
        "record_counts": counts,
        "source_counts": {"synthetic": sum(v for k, v in counts.items() if not k.startswith("public")), "public_csv": counts["public_feedback_accepted"] + counts["public_feedback_quarantined"]},
        "date_ranges": {"synthetic": {"start": "2024-02-01T00:00:00Z", "end": "2025-12-05T00:00:00Z"}},
        "product_counts": {"CloudDesk": sum(v for k, v in counts.items() if k not in ["public_feedback_accepted", "public_feedback_quarantined"])},
        "segment_counts": dict(segs),
        "channel_counts": {"support": counts["support_tickets"], "sales": counts["sales_call_excerpts"], "reviews": counts["product_reviews"], "churn": counts["churn_feedback"], "feature_requests": counts["feature_requests"], "interviews": counts["customer_interviews"], "github_issues": counts["github_issues"]},
        "known_limitations": ["Gold annotations are unverified suggestions until human review.", "Public supplementary reference facts were not live-collected in this run.", "Rule-based atom extraction is a pilot baseline, not an LLM-reviewed final."],
        "reference_requirements_satisfied": {
            "voice_of_customer_intelligence_docx": {
                "support_tickets": "met_or_exceeded",
                "sales_call_excerpts": "met_or_exceeded",
                "churn_comments": "met",
                "customer_reviews": "met",
                "feature_requests": "met",
                "release_notes": "met_or_exceeded",
                "product_document_sections": "met_or_exceeded",
                "crm_accounts": "met",
                "github_issues": "covered_by_synthetic_issue_tracker_records",
            },
            "sales_and_retail_docx": {
                "support_tickets": "met",
                "sales_call_transcripts": "met",
                "product_reviews": "met_or_exceeded",
                "crm_deal_records": "met",
                "synthetic_labeling_rule": "met",
            },
        },
        "collection_failures": ["No live public collection attempted; source registry records planned lawful source review.", "GitHub issue coverage is synthetic CloudDesk issue-tracker data, not scraped public GitHub data."],
        "synthetic_generation_batches": [BATCH_PILOT, BATCH_FULL],
        "manual_annotation_status": "pending_human_review",
        "hash_algorithm": "sha256",
        "license_notes": "Synthetic CloudDesk data is fictional. Public CSV samples require source/license review before publishing.",
    }
    write_json(DATASET_ROOT / "data/manifests/dataset_manifest.json", dataset_manifest)
    write_jsonl(DATASET_ROOT / "data/manifests/source_manifest.jsonl", [{"source_id": deterministic_id("source", "synthetic", "CloudDesk"), "source_name": "CloudDesk synthetic generation", "data_origin": "synthetic"}, {"source_id": deterministic_id("source", "synthetic", "CloudDesk GitHub issues"), "source_name": "CloudDesk synthetic issue tracker", "data_origin": "synthetic"}, {"source_id": deterministic_id("source", "public_csv"), "source_name": "Provided public feedback CSV samples", "data_origin": "scraped"}])
    write_jsonl(DATASET_ROOT / "data/manifests/processing_manifest.jsonl", [{"processed_at": now_iso(), "step": "full_pipeline", "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION, "hash_algorithm": "sha256"}])


def reports(audits, metrics):
    contamination = []
    for a in audits:
        contamination.append(f"- `{a['file_name']}`: accepted {a['records_matching_target_product']} / {a['record_count']}; wrong-product {a['records_mentioning_another_product']}; statuses {a['validation_status_counts']}.")
    missing = []
    for a in audits:
        absent = [k for k, v in a["missing_value_percentages"].items() if v == 100]
        missing.append(f"- `{a['file_name']}` lacks normalized product identity, parent IDs, thread context fields, ratings, fetched_at, source metadata, validation status, and provenance hashes. Fully empty supplied columns: {absent or 'none'}.")
    pilot_config = {
        "reference_target_counts": {
            "support_tickets": 3000,
            "sales_call_excerpts": 1000,
            "product_reviews": 1000,
            "churn_feedback": 300,
            "feature_requests": 500,
            "customer_interviews": 200,
            "github_issues": 500,
            "opportunities": 2000,
            "crm_accounts": 500,
            "release_notes": 20,
            "product_document_sections": 100,
        },
        "pilot_counts": {"accounts": 25, "customers": 50, "support_tickets": 50, "product_reviews": 25, "sales_call_excerpts": 20, "churn_feedback": 15, "feature_requests": 20, "customer_interviews": 10, "github_issues": 20},
        "full_counts_generated": metrics["record_counts"],
        "batching": ["pilot written with pilot_ prefix", "full corpus written to canonical synthetic paths"],
        "validation_gate": "Critical validation errors must be zero before treating generated artifacts as ready for RAG experiments.",
    }
    md = [
        "# Final Dataset Report", "",
        f"Generated at: {now_iso()}", "",
        "## What Was Created",
        "- Stage 1 audit reports for the three provided Reddit CSV samples.",
        "- Stage 2 CloudDesk reference profile, taxonomy, pricing, releases, documentation, and hidden-pattern ground truth.",
        "- Stage 3 pilot files with `pilot_` prefixes.",
        "- Final-scale synthetic CloudDesk corpus in canonical paths, sized to satisfy the reference target counts.",
        "- Synthetic issue-tracker records covering the GitHub-issues source type without pretending live GitHub data was scraped.",
        "- Public CSV normalized and quarantined records with product-identity validation.",
        "- Rule-based atom suggestions, incident clusters, retrieval questions, release-impact cases, abstention questions, and answer-quality rubric.",
        "- Vector-index include and exclusion manifests that keep evaluation answer keys outside normal retrieval.",
        "",
        "## Product Contamination Findings",
        *contamination,
        "",
        "## Missing-Field Analysis",
        *missing,
        "",
        "## Proposed Normalized Schemas",
        "- See `schemas/feedback_item.schema.json`, `schemas/feedback_atom.schema.json`, `schemas/support_ticket.schema.json`, and `schemas/retrieval_question.schema.json`.",
        "- Public feedback preserves raw text, clean text, source URL, hashed author, product-identity decision, validation status, and content hashes.",
        "- Synthetic records keep account/customer links for exact SQL counts while raw textual evidence remains separately addressable.",
        "",
        "## Pilot Generation Configuration",
        "```json",
        json.dumps(pilot_config, indent=2),
        "```",
        "",
        "## Risks, Assumptions, And Limitations",
        "- The public CSVs are user-provided samples; this run did not perform live scraping or current terms/robots review.",
        "- GitHub issue coverage is generated CloudDesk issue-tracker data, clearly labeled synthetic, rather than scraped public GitHub records.",
        "- Public product feature/version/plan enrichment remains placeholder-only until a lawful source review and collection pass is performed.",
        "- Gold annotations and retrieval ground truth are machine-generated suggestions and remain pending human approval.",
        "- Rule-based extraction is useful for bootstrapping but should be replaced or compared with human-reviewed LLM extraction.",
        "- Evaluation files and hidden patterns are intentionally excluded from normal RAG indexing to avoid answer-key leakage.",
        "",
        "## Data Quality",
        f"- Critical validation errors: {metrics['critical_error_count']}",
        f"- Warnings: {metrics['warning_count']}",
        "- See `reports/data_quality_report.md` and `reports/data_quality_metrics.json`.",
    ]
    (DATASET_ROOT / "reports/final_dataset_report.md").write_text("\n".join(md), encoding="utf-8")
    config = {
        "stage_order": ["audit", "schema_definition", "pilot_generation", "validation", "scaled_synthetic_generation", "public_normalization", "evaluation_ground_truth_suggestions"],
        "human_review_checkpoints": ["final product taxonomy", "gold atom annotations", "retrieval ground truth", "ambiguous public records", "publication of public data"],
        "normal_vector_index_exclusions": ["data/evaluation/**", "data/synthetic/reference/hidden_patterns.json"],
    }
    write_json(DATASET_ROOT / "docs/project_configuration.json", config)


def main():
    ensure_dirs()
    doc_text = extract_docx_text(SOURCE_INPUTS / "Voice of customer intelligence.docx")
    (DATASET_ROOT / "docs/reference_document_excerpt.txt").write_text(doc_text[:12000], encoding="utf-8")
    audits, public_normalized, public_quarantined = audit_public_csvs()
    releases = reference_artifacts()
    schema_files()
    public_reference_files()
    pilot = generate_synthetic(25, 50, "pilot", releases)
    persist_synthetic(pilot, "pilot_")
    full = generate_synthetic(500, 1250, "full", releases)
    persist_synthetic(full)
    items = feedback_items_from_synthetic(full) + public_normalized
    write_jsonl(DATASET_ROOT / "data/processed/feedback_items.jsonl", items)
    atoms = extract_atoms(items)
    write_jsonl(DATASET_ROOT / "data/processed/feedback_atoms.jsonl", atoms)
    write_jsonl(DATASET_ROOT / "data/processed/incident_clusters.jsonl", clusters(atoms))
    write_jsonl(DATASET_ROOT / "data/processed/opportunity_evidence.jsonl", [{"opportunity_id": deterministic_id("opp_evidence", a["topic"]), "atom_id": a["atom_id"], "evidence_role": a["evidence_role"], "relevance_score": round(a["extraction_confidence"], 2)} for a in atoms[:1000]])
    evaluation_files(atoms, items)
    metrics = validate(full, atoms, public_normalized, public_quarantined)
    manifests(full, public_normalized, public_quarantined)
    reports(audits, metrics)
    print(json.dumps({"status": "ok", "critical_errors": metrics["critical_error_count"], "counts": metrics["record_counts"]}, indent=2))


if __name__ == "__main__":
    main()
