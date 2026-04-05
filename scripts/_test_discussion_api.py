"""discussionDocumentの全フィールドを確認する。動作確認後に削除する。"""
import json
import time
from kagglesdk import KaggleClient
from kagglesdk.search.types.search_api_service import (
    ListEntitiesRequest, ListEntitiesFilters, ApiSearchDiscussionsFilters,
)
from kagglesdk.discussions.types.search_discussions import (
    SearchDiscussionsSourceType, WriteUpInclusionType,
)

creds_raw = json.load(open("C:/Users/Owner/.kaggle/kaggle.json"))
client = KaggleClient(username=creds_raw["username"], password=creds_raw["key"])

disc_filter = ApiSearchDiscussionsFilters()
disc_filter.source_type = SearchDiscussionsSourceType.SEARCH_DISCUSSIONS_SOURCE_TYPE_COMPETITION
disc_filter.write_up_inclusion_type = WriteUpInclusionType.WRITE_UP_INCLUSION_TYPE_EXCLUDE

filters = ListEntitiesFilters()
filters.query = "birdclef-2026"
filters.discussion_filters = disc_filter

req = ListEntitiesRequest()
req.filters = filters
req.page_size = 20

resp = client.search.search_api_client.list_entities(req)

# TOPICのdiscussionDocumentを全部表示
for doc in resp.documents:
    d = doc.to_dict()
    if d.get("documentType") != "TOPIC":
        continue
    disc = d.get("discussionDocument", {})
    if isinstance(disc, str):
        disc = eval(disc)
    print(f"[votes={d['votes']}] {d['title']}")
    print(f"  discussionDocument keys: {list(disc.keys()) if disc else 'N/A'}")
    print(f"  full: {str(disc)[:200]}")
    print()
    break  # まず1件だけ確認
