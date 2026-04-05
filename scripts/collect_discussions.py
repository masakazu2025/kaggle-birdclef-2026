"""Collect BirdCLEF 2026 discussions from Kaggle and save to docs/discussions/.

Usage:
    poetry run python scripts/collect_discussions.py

Output:
    docs/discussions/raw/{YYYYMMDD}_{slug}_{topic_id}.md  ... 生データ（本文＋コメント）
    docs/discussions/summary/                             ... 要約（Claude Codeとの会話で別途生成）
    docs/discussions/index.md                             ... 一覧テーブル
"""

import ast
import json
import random
import re
import sys
import time
from datetime import datetime, timezone

import _compat  # noqa: F401  Windows UTF-8互換
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.search.types.search_api_service import (
    ApiSearchDiscussionsFilters,
    ListEntitiesFilters,
    ListEntitiesRequest,
)
from kagglesdk.discussions.types.search_discussions import (
    SearchDiscussionsSourceType,
    WriteUpInclusionType,
)

def _sleep() -> None:
    """ランダムなウェイト（SLEEP_BASE ± jitter）を入れてサーバー負荷を分散する。"""
    duration = SLEEP_BASE + random.uniform(0, SLEEP_JITTER)
    time.sleep(duration)


COMPETITION = "birdclef-2026"
DISCUSSIONS_DIR = Path("docs/discussions")
RAW_DIR = DISCUSSIONS_DIR / "raw"
SUMMARY_DIR = DISCUSSIONS_DIR / "summary"
INDEX_FILE = DISCUSSIONS_DIR / "index.md"
SLEEP_BASE = 2.0   # ベース待機秒数
SLEEP_JITTER = 2.0  # ランダム上乗せ上限（秒）
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Kaggle API helpers
# ---------------------------------------------------------------------------

def get_client() -> KaggleClient:
    creds = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))
    return KaggleClient(username=creds["username"], password=creds["key"])


def _parse_nested(value) -> dict:
    """to_dict() の値が str の場合に dict へ変換する。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return {}
    return {}


def _get_message(doc: dict) -> str:
    disc = _parse_nested(doc.get("discussionDocument", {}))
    return disc.get("messageStripped", "")


def _get_owner(doc: dict) -> str:
    owner = _parse_nested(doc.get("ownerUser", {}))
    return owner.get("displayName", "unknown")


def _get_topic_id_from_url(doc: dict) -> str | None:
    """discussionDocument.newCommentUrl から topic_id を抽出する。"""
    disc = _parse_nested(doc.get("discussionDocument", {}))
    url = disc.get("newCommentUrl", "")
    m = re.search(r"/discussion/(\d+)", url)
    return m.group(1) if m else None


def fetch_all_topics(client: KaggleClient) -> list[dict]:
    """全 TOPIC を取得してリストで返す。"""
    disc_filter = ApiSearchDiscussionsFilters()
    disc_filter.source_type = SearchDiscussionsSourceType.SEARCH_DISCUSSIONS_SOURCE_TYPE_COMPETITION
    disc_filter.write_up_inclusion_type = WriteUpInclusionType.WRITE_UP_INCLUSION_TYPE_EXCLUDE

    filters = ListEntitiesFilters()
    filters.query = COMPETITION
    filters.discussion_filters = disc_filter

    topics = []
    page_token = ""
    while True:
        req = ListEntitiesRequest()
        req.filters = filters
        req.page_size = 100
        req.page_token = page_token

        resp = client.search.search_api_client.list_entities(req)
        for doc in resp.documents:
            d = doc.to_dict()
            if d.get("documentType") == "TOPIC":
                topics.append(d)

        next_token = resp.next_page_token
        if not next_token:
            break
        page_token = next_token
        _sleep()

    return topics


def _list_entities_with_retry(client: KaggleClient, req: ListEntitiesRequest):
    """500エラー時にリトライするラッパー。"""
    import requests as req_lib
    for attempt in range(MAX_RETRIES):
        try:
            return client.search.search_api_client.list_entities(req)
        except req_lib.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 500 and attempt < MAX_RETRIES - 1:
                wait = SLEEP_BASE * (attempt + 2) + random.uniform(0, SLEEP_JITTER)
                print(f"  → 500 error, retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
    return None  # unreachable


def fetch_comments(client: KaggleClient, topic: dict) -> list[dict]:
    """タイトルキーワードで検索し、topic_id で照合してコメントを返す。"""
    topic_id = str(topic["id"])
    # 英数字のみのキーワード4語に絞る（特殊文字でAPIが500になるのを防ぐ）
    raw_words = topic["title"].split()
    clean_words = [re.sub(r"[^\w]", "", w) for w in raw_words if re.sub(r"[^\w]", "", w)]
    query = " ".join(clean_words[:4])

    disc_filter = ApiSearchDiscussionsFilters()
    disc_filter.source_type = SearchDiscussionsSourceType.SEARCH_DISCUSSIONS_SOURCE_TYPE_COMPETITION

    filters = ListEntitiesFilters()
    filters.query = query
    filters.discussion_filters = disc_filter

    comments = []
    page_token = ""
    while True:
        req = ListEntitiesRequest()
        req.filters = filters
        req.page_size = 100
        req.page_token = page_token

        try:
            resp = _list_entities_with_retry(client, req)
        except Exception as e:
            print(f"  → comment fetch failed: {e}. skipping comments.")
            break

        for doc in resp.documents:
            d = doc.to_dict()
            if d.get("documentType") != "COMMENT":
                continue
            if _get_topic_id_from_url(d) != topic_id:
                continue
            comments.append(d)

        next_token = resp.next_page_token
        if not next_token:
            break
        page_token = next_token
        _sleep()

    # 投稿日時順に並べる
    comments.sort(key=lambda d: d.get("createTime", ""))
    return comments


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")[:50]


def make_raw_filename(topic: dict) -> str:
    date = topic.get("createTime", "")[:10].replace("-", "")
    slug = slugify(topic.get("title", "untitled"))
    topic_id = topic["id"]
    return f"{date}_{slug}_{topic_id}.md"


def read_frontmatter_value(path: Path, key: str) -> str | None:
    """ファイルの YAML frontmatter から指定キーの値を返す。"""
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return None
    end = content.find("\n---\n", 3)
    if end == -1:
        return None
    for line in content[3:end].splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return None


def write_raw_file(path: Path, topic: dict, comments: list[dict]) -> None:
    topic_id = topic["id"]
    title = topic.get("title", "")
    create_time = topic.get("createTime", "")
    update_time = topic.get("updateTime", "")
    votes = topic.get("votes", 0)
    owner = _get_owner(topic)
    url = f"https://www.kaggle.com/competitions/{COMPETITION}/discussion/{topic_id}"
    body = _get_message(topic)

    lines = [
        "---",
        f"topic_id: {topic_id}",
        f"update_time: {update_time}",
        "---",
        "",
        f"# {title}",
        "",
        f"- **原文URL**: {url}",
        f"- **投稿日時**: {create_time}",
        f"- **更新日時**: {update_time}",
        f"- **Votes**: {votes}",
        f"- **投稿者**: {owner}",
        "",
        "## 本文",
        "",
        body,
        "",
    ]

    if comments:
        lines += ["## コメント", ""]
        for i, c in enumerate(comments, 1):
            c_votes = c.get("votes", 0)
            c_owner = _get_owner(c)
            c_time = c.get("createTime", "")[:16].replace("T", " ")
            c_body = _get_message(c)
            lines += [
                f"### コメント{i}（votes: {c_votes} / {c_owner} / {c_time}）",
                "",
                c_body,
                "",
            ]

    path.write_text("\n".join(lines), encoding="utf-8")


def is_summary_needed(raw_path: Path, summary_path: Path) -> bool:
    """summaryが存在しない、またはrawが更新されていれば True。"""
    if not summary_path.exists():
        return True
    raw_update = read_frontmatter_value(raw_path, "update_time")
    summary_target = read_frontmatter_value(summary_path, "target_update_time")
    if not raw_update or not summary_target:
        return True
    return raw_update > summary_target


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(topics_meta: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    lines = [
        "# BirdCLEF 2026 Discussion Index",
        "",
        f"最終更新: {now}",
        "",
        "| 日付 | タイトル | Votes | 最終更新日時 | 要約待ち | リンク |",
        "|------|----------|-------|------------|---------|--------|",
    ]

    for m in sorted(topics_meta, key=lambda x: -x["votes"]):
        date = m["create_time"][:10]
        title = m["title"]
        votes = m["votes"]
        update_time = m["update_time"][:16].replace("T", " ") + "Z"
        flag = "✓" if m["summary_needed"] else ""
        raw_link = f"[raw](raw/{m['filename']})"
        summary_path = SUMMARY_DIR / m["filename"]
        summary_link = f"[summary](summary/{m['filename']})" if summary_path.exists() else ""
        links = " ".join(filter(None, [raw_link, summary_link]))
        lines.append(f"| {date} | {title} | {votes} | {update_time} | {flag} | {links} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    client = get_client()

    print("Fetching topics...")
    topics = fetch_all_topics(client)
    print(f"Found {len(topics)} topics.")

    # 既存rawファイルを topic_id → Path でマッピング
    existing_raw: dict[int, Path] = {}
    for f in RAW_DIR.glob("*.md"):
        m = re.search(r"_(\d+)\.md$", f.name)
        if m:
            existing_raw[int(m.group(1))] = f

    topics_meta = []

    for i, topic in enumerate(topics, 1):
        topic_id = int(topic["id"])
        title = topic.get("title", "")
        api_update_time = topic.get("updateTime", "")

        print(f"[{i}/{len(topics)}] {title[:60]}")

        # rawファイルのパスを決定
        if topic_id in existing_raw:
            raw_path = existing_raw[topic_id]
        else:
            raw_path = RAW_DIR / make_raw_filename(topic)

        # 更新要否チェック
        needs_raw_update = True
        if raw_path.exists():
            stored = read_frontmatter_value(raw_path, "update_time")
            if stored and stored >= api_update_time:
                needs_raw_update = False
                print("  → skip (up to date)")

        if needs_raw_update:
            print("  → fetching comments...")
            comments = fetch_comments(client, topic)
            print(f"  → {len(comments)} comments. writing...")
            write_raw_file(raw_path, topic, comments)
            _sleep()

        summary_path = SUMMARY_DIR / raw_path.name
        topics_meta.append({
            "title": title,
            "votes": int(topic.get("votes", 0)),
            "create_time": topic.get("createTime", ""),
            "update_time": api_update_time,
            "filename": raw_path.name,
            "summary_needed": is_summary_needed(raw_path, summary_path),
        })

    print("Building index.md...")
    INDEX_FILE.write_text(build_index(topics_meta), encoding="utf-8")
    print(f"Done. {len(topics_meta)} topics written to index.md.")


if __name__ == "__main__":
    main()
