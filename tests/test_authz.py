from app.db import MetadataStore
from app.models import ChunkRecord, UserContext


def test_user_can_access_public_chunk(tmp_path):
    store = MetadataStore(db_path=str(tmp_path / "rag.db"))
    chunk = ChunkRecord(
        chunk_id="1",
        source="policy.pdf",
        title="policy",
        text="hello",
        extra_metadata={"visibility": "public"},
    )
    store.insert_chunks([chunk])

    user = UserContext(user_id="alice", roles=[])
    saved = store.get_chunk_by_id("1")
    assert saved is not None
    assert store.user_can_access_chunk(saved, user) is True


def test_user_cannot_access_private_chunk_without_permission(tmp_path):
    store = MetadataStore(db_path=str(tmp_path / "rag.db"))
    chunk = ChunkRecord(
        chunk_id="2",
        source="contract.pdf",
        title="contract",
        text="secret",
        extra_metadata={"visibility": "private", "allowed_users": ["bob"]},
    )
    store.insert_chunks([chunk])

    user = UserContext(user_id="alice", roles=[])
    saved = store.get_chunk_by_id("2")
    assert saved is not None
    assert store.user_can_access_chunk(saved, user) is False