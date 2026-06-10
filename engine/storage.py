"""Disk + SQLite storage for the memory engine.

Owns: file helpers (atomic writes), the inverted-index schema and its
migrations, chunk persistence, document ingestion, the incremental chat
buffer, upload management with content-hash dedup, and the single-instance
lock.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from runtime_paths import default_memory_dir, user_app_support_root
from engine.textutil import tokenize

SCHEMA_VERSION = 2

VAULT_HEADER = "# Vault (Long-tail)\n\n> 这里记录 Vault 的索引摘要；原文分块在 chunks/ 中。\n\n"


# -------------------------- file helpers --------------------------


def now_ts() -> int:
    return int(time.time())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    """Atomic write: a crash mid-write must never truncate memory files."""

    ensure_dir(os.path.dirname(path))
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".evermate-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def append_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9一-鿿._\- ()]", "_", name)
    return name[:180] if len(name) > 180 else name


def can_write_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_path = os.path.join(path, ".evermate_write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False


def resolve_memory_dir(raw: str) -> str:
    """Pick a writable memory root: configured path, else user app-data."""

    raw = (raw or "").strip()
    preferred = os.path.abspath(raw or default_memory_dir())
    fallback = str(user_app_support_root() / "memory")

    if can_write_dir(preferred):
        return preferred
    if can_write_dir(fallback):
        return os.path.abspath(fallback)
    return preferred


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class InstanceLock:
    """Advisory single-instance lock on the memory directory.

    Two app instances sharing one memory root race on buffer.txt, the
    refresh counter, and rebuilds (which delete the DB under the other
    instance). Acquire this before opening the UI.
    """

    def __init__(self, memory_dir: str):
        self.path = os.path.join(memory_dir, ".evermate.lock")
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        import fcntl

        ensure_dir(os.path.dirname(self.path))
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


# -------------------------- store --------------------------


class MemoryStore:
    """Chunk-on-disk + SQLite inverted index (terms / postings / chunks)."""

    def __init__(self, memory_dir: str, chunk_chars: int = 2800):
        self.memory_dir = memory_dir
        self.chunk_chars = int(chunk_chars)

        self.db_path = os.path.join(memory_dir, "index.sqlite")
        self.chunks_dir = os.path.join(memory_dir, "chunks")
        self.uploads_dir = os.path.join(memory_dir, "uploads")
        self.buffer_path = os.path.join(memory_dir, "buffer.txt")
        self.chat_log_path = os.path.join(memory_dir, "chat_log.txt")
        self.vault_md_path = os.path.join(memory_dir, "03_vault.md")

        ensure_dir(memory_dir)
        ensure_dir(self.chunks_dir)
        ensure_dir(self.uploads_dir)

        self.conn = self._open_db()
        # (chunk count, avg doc len) is needed on every retrieve; cache it
        # instead of a COUNT/AVG table scan per query.
        self._stats_cache: Optional[Tuple[int, float]] = None

    # ---------------- schema ----------------

    def _open_db(self) -> sqlite3.Connection:
        # check_same_thread=False: the GUI runs all engine work on a single
        # background worker (writes are serialized there); the main thread
        # only does short status reads. System SQLite is built thread-safe
        # (serialized), which covers that overlap.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                source TEXT,
                created_at INTEGER,
                char_len INTEGER,
                doc_len INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS terms (
                term TEXT PRIMARY KEY,
                df INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS postings (
                term TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (term, chunk_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS imported_files (
                content_hash TEXT PRIMARY KEY,
                name TEXT,
                imported_at INTEGER,
                ingested INTEGER DEFAULT 0
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);")
        conn.commit()
        self._migrate(conn)
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        row = cur.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row["value"]) if row else 1
        if version < 2:
            # v1 carried idx_postings_term, redundant with the (term, chunk_id)
            # primary key — it only doubled insert cost.
            cur.execute("DROP INDEX IF EXISTS idx_postings_term;")
            version = 2
        cur.execute(
            "INSERT INTO meta(key,value) VALUES ('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(version),),
        )
        conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ---------------- meta ----------------

    def meta_get_int(self, key: str, default: int = 0) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return default

    def meta_set_int(self, key: str, value: int) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(int(value))),
        )
        self.conn.commit()

    # ---------------- stats ----------------

    def count_chunks(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def count_terms(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM terms").fetchone()
        return int(row["n"]) if row else 0

    def corpus_stats(self) -> Tuple[int, float]:
        """(chunk count, average doc length), cached between writes."""

        if self._stats_cache is None:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n, AVG(doc_len) AS avgdl FROM chunks"
            ).fetchone()
            n = int(row["n"]) if row else 0
            avgdl = float(row["avgdl"] or 1.0) if row else 1.0
            self._stats_cache = (n, avgdl)
        return self._stats_cache

    def _invalidate_stats(self) -> None:
        self._stats_cache = None

    # ---------------- chunk persistence ----------------

    def add_chunk(self, text: str, source: str, _vault_buffer: Optional[List[str]] = None) -> int:
        """Persist one chunk + index rows. Returns chunk id or -1.

        When `_vault_buffer` is given (bulk ingestion), the vault summary
        line is buffered instead of appended to disk per chunk, and the
        caller owns the transaction commit.
        """

        text = text.strip()
        if not text:
            return -1

        tokens = tokenize(text)
        if not tokens:
            return -1

        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        created_at = now_ts()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO chunks(path, source, created_at, char_len, doc_len) VALUES (?,?,?,?,?)",
            ("", source, created_at, len(text), len(tokens)),
        )
        chunk_id = int(cur.lastrowid)

        rel_path = os.path.join("chunks", f"{chunk_id:08d}.txt")
        write_text(os.path.join(self.memory_dir, rel_path), text)
        cur.execute("UPDATE chunks SET path=? WHERE id=?", (rel_path, chunk_id))

        cur.executemany(
            "INSERT INTO postings(term, chunk_id, tf) VALUES (?,?,?)",
            [(term, chunk_id, int(freq)) for term, freq in tf.items()],
        )
        cur.executemany(
            "INSERT INTO terms(term, df) VALUES (?,1) ON CONFLICT(term) DO UPDATE SET df=df+1",
            [(term,) for term in tf],
        )

        stamp = time.strftime("%Y-%m-%d", time.localtime(created_at))
        preview = text.replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:200] + "…"
        vault_line = f"### Chunk {chunk_id:08d} ({stamp}) [{source}]\n\n{preview}\n\n"

        if _vault_buffer is not None:
            _vault_buffer.append(vault_line)
        else:
            self.conn.commit()
            append_text(self.vault_md_path, vault_line)

        self._invalidate_stats()
        return chunk_id

    def chunk_text_by_id(self, chunk_id: int) -> str:
        row = self.conn.execute(
            "SELECT path FROM chunks WHERE id=?", (int(chunk_id),)
        ).fetchone()
        if not row:
            return ""
        abs_path = os.path.join(self.memory_dir, str(row["path"]))
        return read_text(abs_path)

    def chunk_row_by_id(self, chunk_id: int):
        return self.conn.execute(
            "SELECT id, source, path, created_at FROM chunks WHERE id=?", (int(chunk_id),)
        ).fetchone()

    def recent_chunks_text(self, max_chunks: int = 12, max_chars: int = 12000) -> str:
        rows = self.conn.execute(
            "SELECT id, path FROM chunks ORDER BY id DESC LIMIT ?", (int(max_chunks),)
        ).fetchall()
        texts: List[str] = []
        total = 0
        for r in rows:
            t = read_text(os.path.join(self.memory_dir, str(r["path"]))).strip()
            if not t:
                continue
            if total + len(t) > max_chars:
                t = t[: max(0, max_chars - total)]
            texts.append(t)
            total += len(t)
            if total >= max_chars:
                break
        return "\n\n".join(reversed(texts))

    # ---------------- ingestion ----------------

    def ingest_file(self, path: str, source: str) -> int:
        """Ingest one document inside a single transaction with buffered
        vault writes — bulk imports used to pay one commit per chunk."""

        ext = os.path.splitext(path)[1].lower()
        vault_buffer: List[str] = []
        made = 0
        try:
            if ext == ".txt":
                made = self._ingest_txt_stream(path, source, vault_buffer)
            elif ext == ".docx":
                made = self._ingest_docx(path, source, vault_buffer)
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        if vault_buffer:
            append_text(self.vault_md_path, "".join(vault_buffer))
        return made

    def _ingest_txt_stream(self, path: str, source: str, vault_buffer: List[str]) -> int:
        made = 0
        acc: List[str] = []
        acc_len = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line:
                    continue
                acc.append(line)
                acc_len += len(line)
                if acc_len >= self.chunk_chars:
                    if self.add_chunk("".join(acc), source, vault_buffer) >= 0:
                        made += 1
                    acc, acc_len = [], 0
        if acc and self.add_chunk("".join(acc), source, vault_buffer) >= 0:
            made += 1
        return made

    def _ingest_docx(self, path: str, source: str, vault_buffer: List[str]) -> int:
        try:
            from docx import Document  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "解析 .docx 需要 python-docx。请先安装：pip install python-docx"
            ) from e

        doc = Document(path)
        made = 0
        acc: List[str] = []
        acc_len = 0
        for para in doc.paragraphs:
            t = (para.text or "").strip()
            if not t:
                continue
            line = t + "\n"
            acc.append(line)
            acc_len += len(line)
            if acc_len >= self.chunk_chars:
                if self.add_chunk("".join(acc), source, vault_buffer) >= 0:
                    made += 1
                acc, acc_len = [], 0
        if acc and self.add_chunk("".join(acc), source, vault_buffer) >= 0:
            made += 1
        return made

    def ingest_text(self, text: str, source: str) -> int:
        if not text:
            return 0
        vault_buffer: List[str] = []
        made = 0
        start = 0
        n = len(text)
        try:
            while start < n:
                end = min(n, start + self.chunk_chars)
                piece = text[start:end]
                if end < n:
                    nl = piece.rfind("\n")
                    if nl >= int(self.chunk_chars * 0.5):
                        end = start + nl
                        piece = text[start:end]
                if self.add_chunk(piece, source, vault_buffer) >= 0:
                    made += 1
                start = end
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        if vault_buffer:
            append_text(self.vault_md_path, "".join(vault_buffer))
        return made

    # ---------------- uploads ----------------

    def import_files(self, file_paths: List[str]) -> List[str]:
        """Copy new files into uploads/, skipping content we already have.

        Re-importing the same document used to silently double the corpus.
        """

        stored: List[str] = []
        for p in file_paths:
            if not p or not os.path.exists(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext not in (".txt", ".docx"):
                continue

            digest = file_sha256(p)
            row = self.conn.execute(
                "SELECT name FROM imported_files WHERE content_hash=?", (digest,)
            ).fetchone()
            if row:
                continue

            base = safe_filename(os.path.basename(p))
            dst = os.path.join(self.uploads_dir, base)
            if os.path.exists(dst):
                stem, ext2 = os.path.splitext(base)
                dst = os.path.join(self.uploads_dir, f"{stem}_{now_ts()}{ext2}")
            shutil.copy2(p, dst)
            self.conn.execute(
                "INSERT INTO imported_files(content_hash, name, imported_at, ingested) VALUES (?,?,?,0)",
                (digest, os.path.basename(dst), now_ts()),
            )
            self.conn.commit()
            stored.append(dst)
        return stored

    def mark_ingested(self, stored_path: str) -> None:
        self.conn.execute(
            "UPDATE imported_files SET ingested=1 WHERE name=?",
            (os.path.basename(stored_path),),
        )
        self.conn.commit()

    def list_uploads(self) -> List[str]:
        if not os.path.exists(self.uploads_dir):
            return []
        files = [os.path.join(self.uploads_dir, f) for f in os.listdir(self.uploads_dir)]
        files = [
            f
            for f in files
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in (".txt", ".docx")
        ]
        files.sort(key=lambda p: os.path.getmtime(p))
        return files

    def delete_upload(self, stored_path: str) -> bool:
        """Remove an uploaded document (its chunks go away on next rebuild)."""

        if os.path.dirname(os.path.abspath(stored_path)) != os.path.abspath(self.uploads_dir):
            return False
        if not os.path.exists(stored_path):
            return False
        try:
            digest = file_sha256(stored_path)
            self.conn.execute("DELETE FROM imported_files WHERE content_hash=?", (digest,))
            self.conn.commit()
        except OSError:
            pass
        os.remove(stored_path)
        return True

    # ---------------- destructive ops ----------------

    def reset_index(self) -> None:
        """Drop the index (with a .bak of the old DB) ahead of a rebuild."""

        self.close()
        for fn in os.listdir(self.chunks_dir) if os.path.exists(self.chunks_dir) else []:
            fp = os.path.join(self.chunks_dir, fn)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except OSError:
                pass
        write_text(self.vault_md_path, VAULT_HEADER)
        if os.path.exists(self.db_path):
            shutil.copy2(self.db_path, self.db_path + ".bak")
            os.remove(self.db_path)
        for suffix in ("-wal", "-shm"):
            side = self.db_path + suffix
            if os.path.exists(side):
                try:
                    os.remove(side)
                except OSError:
                    pass
        self.conn = self._open_db()
        self._invalidate_stats()

    def clear_chat_history(self) -> None:
        """Forget chat-derived memory: log, buffer, and chat-source chunks."""

        write_text(self.chat_log_path, "")
        write_text(self.buffer_path, "")

    def wipe_all(self) -> None:
        """Forget everything: index, chunks, uploads, logs."""

        self.close()
        for sub in (self.chunks_dir, self.uploads_dir):
            if os.path.exists(sub):
                shutil.rmtree(sub, ignore_errors=True)
            ensure_dir(sub)
        for f in (self.buffer_path, self.chat_log_path):
            write_text(f, "")
        write_text(self.vault_md_path, VAULT_HEADER)
        for suffix in ("", "-wal", "-shm", ".bak"):
            side = self.db_path + suffix
            if os.path.exists(side):
                try:
                    os.remove(side)
                except OSError:
                    pass
        self.conn = self._open_db()
        self._invalidate_stats()
