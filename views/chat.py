# views/chat.py

from __future__ import annotations

import html
import os
import queue
import threading
import time
import traceback

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from i18n_qt import tr
from memory_manager import MemoryManager
from models_config import TARGET_MODELS, is_local_model, resolve_installed_model
from ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    chat_stream,
    list_models,
)


PERSONA_NAMES = {
    "zh": {"listener": "温柔倾听者", "buddy": "聊天搭子", "writer": "创意作家"},
    "en": {"listener": "Gentle Listener", "buddy": "Chat Buddy", "writer": "Creative Writer"},
}


PERSONA_PROMPTS = {
    "zh": {
        "listener": "您是一位温柔体贴的本地 AI 朋友“小枢”。先用一句话复述要点，再用 1–2 句表达理解，然后给出不超过 2 条可操作建议。语气温柔，输出 4–8 句。",
        "buddy": "您是一位轻松幽默、实用主义的聊天搭子“小枢”。先确认目标，再给出简洁可执行的步骤清单。每条不超过 20 字。",
        "writer": "您是一位善于结构化表达的创意作家“小枢”。用小标题 + 列表组织内容，重点在逻辑清晰与表达精炼。",
    },
    "en": {
        "listener": "You are EverMate, a gentle and caring local AI friend. Restate the key point in one sentence, show understanding in 1–2 sentences, then give at most 2 actionable suggestions. Keep a warm tone, 4–8 sentences total.",
        "buddy": "You are EverMate, a relaxed, pragmatic chat buddy. Confirm the goal first, then give a short, actionable checklist. Keep each item under 12 words.",
        "writer": "You are EverMate, a creative writer who excels at structured expression. Organize with small headings and lists; prioritize clear logic and concise wording.",
    },
}


MAX_SESSION_MESSAGES = 12  # keep last 6 turns (user+assistant)

RECOMMENDED_MODEL_LABELS = {
    "deepseek_qwen3_8b": "DeepSeek R1 8B · Easy",
    "qwen3_30b_a3b": "Qwen3 30B-A3B · Fast",
    "deepseek_r1_70b": "DeepSeek R1 70B · Stable",
    "gpt_oss_120b": "gpt-oss 120B · Best",
}


def _is_recommended_model_key(value: str) -> bool:
    return any(m["key"] == value for m in TARGET_MODELS)


def _is_covered_by_recommended_model(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    if not name:
        return False
    for target in TARGET_MODELS:
        for candidate in target.get("candidates", []):
            if name.startswith(candidate.lower()):
                return True
    return False


def _escape_text(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


class InstanceLockedError(RuntimeError):
    """Another EverMate instance owns the memory directory."""


class EngineWorker(QThread):
    """Single persistent worker that serializes every engine / LLM job.

    The UI thread never talks to Ollama and never runs indexing; it submits
    closures here and reacts to signals. One worker means SQLite writes are
    naturally serialized.
    """

    chunk = Signal(str)
    done = Signal(str, object)  # job kind, result
    failed = Signal(str, str, str)  # job kind, error kind, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: "queue.Queue" = queue.Queue()
        # Cooperative cancellation: long-running jobs (the streaming loop)
        # check this between steps so the app can shut down cleanly.
        self.cancel = threading.Event()

    def submit(self, kind: str, fn) -> None:
        self._queue.put((kind, fn))

    def stop(self) -> None:
        self.cancel.set()
        self._queue.put(None)

    def run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, fn = item
            try:
                result = fn(self.chunk.emit)
            except OllamaConnectionError:
                self.failed.emit(kind, "connection", "")
            except OllamaModelNotFoundError as e:
                self.failed.emit(kind, "model_missing", getattr(e, "model", "") or str(e))
            except OllamaError as e:
                self.failed.emit(kind, "ollama", str(e))
            except Exception as e:
                traceback.print_exc()
                self.failed.emit(kind, "engine", str(e))
            else:
                self.done.emit(kind, result)


class ComposerEdit(QTextEdit):
    send_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("ComposerEdit")
        self.setAcceptRichText(False)
        self.setTabChangesFocus(True)
        self.setFixedHeight(92)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self.send_requested.emit()
            return
        super().keyPressEvent(event)


class DropArea(QLabel):
    files_dropped = Signal(list)

    def __init__(self, text: str):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(82)
        self.setWordWrap(True)
        self.setObjectName("DropArea")

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            paths = self._expand_paths([u.toLocalFile() for u in md.urls()])
            if paths:
                self.setProperty("dragover", True)
                self._repolish()
                event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setProperty("dragover", False)
        self._repolish()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.setProperty("dragover", False)
        self._repolish()
        md = event.mimeData()
        if not md.hasUrls():
            return
        paths = self._expand_paths([u.toLocalFile() for u in md.urls()])
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()

    @staticmethod
    def _expand_paths(raw_paths: list[str]) -> list[str]:
        """Accept .txt/.docx files, and folders containing them (one level)."""

        out: list[str] = []
        for p in raw_paths:
            if not p:
                continue
            if os.path.isdir(p):
                for name in sorted(os.listdir(p)):
                    fp = os.path.join(p, name)
                    if os.path.isfile(fp) and fp.lower().endswith((".txt", ".docx")):
                        out.append(fp)
            elif p.lower().endswith((".txt", ".docx")):
                out.append(p)
        return out

    def _repolish(self):
        self.style().unpolish(self)
        self.style().polish(self)


class UploadsDialog(QDialog):
    """List imported documents; deleting one rebuilds memory without it."""

    delete_requested = Signal(str)

    def __init__(self, lang: str, uploads: list[str], parent=None):
        super().__init__(parent)
        self.lang = lang
        self.setWindowTitle(tr(lang, "uploads_dialog_title"))
        self.setMinimumSize(420, 320)

        layout = QVBoxLayout(self)
        self.listw = QListWidget()
        self.listw.setAccessibleName(tr(lang, "uploads_dialog_title"))
        for path in uploads:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            self.listw.addItem(item)
        if not uploads:
            self.listw.addItem(QListWidgetItem(tr(lang, "no_uploads")))
            self.listw.setEnabled(False)
        layout.addWidget(self.listw, 1)

        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton(tr(lang, "delete"))
        self.delete_btn.setObjectName("SecondaryButton")
        self.delete_btn.setEnabled(bool(uploads))
        self.delete_btn.clicked.connect(self._on_delete)
        close_btn = QPushButton(tr(lang, "close"))
        close_btn.setObjectName("TertiaryButton")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.delete_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _on_delete(self):
        item = self.listw.currentItem()
        if not item:
            return
        path = item.data(Qt.UserRole)
        if not path:
            return
        name = os.path.basename(path)
        if (
            QMessageBox.question(
                self,
                tr(self.lang, "manage_uploads"),
                tr(self.lang, "delete_upload_confirm", name=name),
            )
            == QMessageBox.Yes
        ):
            self.delete_requested.emit(path)
            self.accept()


class ChatPage(QWidget):
    state_changed = Signal()

    def __init__(self, on_change_lang=None, on_change_theme=None):
        super().__init__()
        self.on_change_lang = on_change_lang
        self.on_change_theme = on_change_theme

        self.lang = "en"
        self.theme = "dark"
        self.persona = "buddy"
        self.session_messages: list[dict] = []
        self.pending_files: list[str] = []
        self.memory_panel_visible = False
        self._chat_messages_html: list[str] = []
        self._animations: list[QPropertyAnimation] = []

        self._chat_busy = False
        self._inflight_text = ""
        self._stream_started = False
        self._stream_text = ""
        self._stream_start_pos = -1
        self._ollama_reachable: bool | None = None
        self._active_jobs: dict[str, str] = {}
        self._installed_models: list[str] = []

        self.mm = MemoryManager()
        # Single-instance guard must hold BEFORE the worker starts and before
        # anything heavy touches the shared store.
        if not self.mm.acquire_instance_lock():
            self.mm.close()
            raise InstanceLockedError()
        self.mm.ui_lang = self.lang

        self.worker = EngineWorker(self)
        self.worker.chunk.connect(self._on_stream_chunk)
        self.worker.done.connect(self._on_job_done)
        self.worker.failed.connect(self._on_job_failed)
        self.worker.start()

        self._build_ui()
        self._submit_model_scan()

    def shutdown(self):
        """Stop the worker cooperatively, then close the store.

        The streaming loop polls the cancel event, so even a mid-reply quit
        converges quickly; the generous waits cover rebuilds of large
        corpora. The connection is only closed once the worker is done —
        yanking SQLite from under a live thread (or destroying a running
        QThread) crashes Qt 6.
        """

        self.worker.stop()
        finished = self.worker.wait(15000) or self.worker.wait(45000)
        if not finished:
            self.worker.terminate()
            self.worker.wait(2000)
        self.mm.close()

    # ---------------- UI construction ----------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(304)
        sidebar_outer = QVBoxLayout(self.sidebar)
        sidebar_outer.setContentsMargins(0, 0, 0, 0)
        sidebar_outer.setSpacing(0)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("SidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setFrameShape(QFrame.NoFrame)
        self.sidebar_content = QWidget()
        self.sidebar_content.setObjectName("SidebarContent")
        sidebar_layout = QVBoxLayout(self.sidebar_content)
        sidebar_layout.setContentsMargins(18, 18, 18, 18)
        sidebar_layout.setSpacing(12)
        self.sidebar_scroll.setWidget(self.sidebar_content)
        sidebar_outer.addWidget(self.sidebar_scroll)

        brand = QLabel("EverMate")
        brand.setObjectName("BrandTitle")
        self.tagline = QLabel("")
        self.tagline.setObjectName("MutedLabel")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(self.tagline)

        self.model_label = QLabel("")
        self.model_label.setObjectName("FieldLabel")
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ControlCombo")
        self.model_combo.setAccessibleName("Model")
        self._populate_model_combo([])
        self.model_combo.currentIndexChanged.connect(self._on_model_change)
        sidebar_layout.addWidget(self.model_label)
        sidebar_layout.addWidget(self.model_combo)

        self.persona_label = QLabel(tr(self.lang, "select_persona"))
        self.persona_label.setObjectName("FieldLabel")
        self.persona_combo = QComboBox()
        self.persona_combo.setObjectName("ControlCombo")
        self.persona_combo.setAccessibleName("Persona")
        self._fill_personas()
        self.persona_combo.currentIndexChanged.connect(self._on_persona_change)
        sidebar_layout.addWidget(self.persona_label)
        sidebar_layout.addWidget(self.persona_combo)

        locale_row = QHBoxLayout()
        locale_row.setSpacing(10)
        lang_box = QVBoxLayout()
        lang_box.setSpacing(6)
        self.lang_label = QLabel("")
        self.lang_label.setObjectName("FieldLabel")
        self.lang_combo = QComboBox()
        self.lang_combo.setObjectName("CompactCombo")
        self.lang_combo.setAccessibleName("Language")
        self.lang_combo.addItem("中文", "zh")
        self.lang_combo.addItem("English", "en")
        self._set_combo_by_data(self.lang_combo, self.lang)
        self.lang_combo.currentIndexChanged.connect(self._on_lang_change)
        lang_box.addWidget(self.lang_label)
        lang_box.addWidget(self.lang_combo)
        theme_box = QVBoxLayout()
        theme_box.setSpacing(6)
        self.theme_label = QLabel("")
        self.theme_label.setObjectName("FieldLabel")
        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("CompactCombo")
        self.theme_combo.setAccessibleName("Theme")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self._set_combo_by_data(self.theme_combo, self.theme)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_change)
        theme_box.addWidget(self.theme_label)
        theme_box.addWidget(self.theme_combo)
        locale_row.addLayout(lang_box)
        locale_row.addLayout(theme_box)
        sidebar_layout.addLayout(locale_row)

        self.memory_card = QFrame()
        self.memory_card.setObjectName("MemoryCard")
        self.memory_card.setMinimumHeight(166)
        card_layout = QVBoxLayout(self.memory_card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)
        self.memory_title = QLabel("")
        self.memory_title.setObjectName("PanelTitle")
        card_layout.addWidget(self.memory_title)

        stats = QGridLayout()
        stats.setHorizontalSpacing(10)
        stats.setVerticalSpacing(8)
        self.chunk_value = self._stat_value_label()
        self.term_value = self._stat_value_label()
        self.upload_value = self._stat_value_label()
        self.last_value = self._stat_value_label()
        self.chunk_caption = self._stat_caption_label("")
        self.term_caption = self._stat_caption_label("")
        self.upload_caption = self._stat_caption_label("")
        self.last_caption = self._stat_caption_label("")
        stats.addWidget(self.chunk_caption, 0, 0)
        stats.addWidget(self.chunk_value, 1, 0)
        stats.addWidget(self.term_caption, 0, 1)
        stats.addWidget(self.term_value, 1, 1)
        stats.addWidget(self.upload_caption, 2, 0)
        stats.addWidget(self.upload_value, 3, 0)
        stats.addWidget(self.last_caption, 2, 1)
        stats.addWidget(self.last_value, 3, 1)
        card_layout.addLayout(stats)

        self.mem_status_label = QLabel("")
        self.mem_status_label.setObjectName("TinyMutedLabel")
        self.mem_status_label.setWordWrap(True)
        card_layout.addWidget(self.mem_status_label)
        sidebar_layout.addWidget(self.memory_card)

        self.drop_area = DropArea(tr(self.lang, "drop_here"))
        self.drop_area.setAccessibleName("Import files")
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        sidebar_layout.addWidget(self.drop_area)

        self.pending_label = QLabel("")
        self.pending_label.setObjectName("PendingLabel")
        self.pending_label.setWordWrap(True)
        sidebar_layout.addWidget(self.pending_label)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.build_btn = QPushButton(tr(self.lang, "build_memory"))
        self.build_btn.setObjectName("SecondaryButton")
        self.build_btn.setAccessibleName("Build memory")
        self.build_btn.clicked.connect(self._on_build_memory)
        self.analyze_btn = QPushButton(tr(self.lang, "analyze"))
        self.analyze_btn.setObjectName("SecondaryButton")
        self.analyze_btn.setAccessibleName("Analyze memory")
        self.analyze_btn.clicked.connect(self._on_analyze_memory)
        actions.addWidget(self.build_btn)
        actions.addWidget(self.analyze_btn)
        sidebar_layout.addLayout(actions)

        detail_row = QHBoxLayout()
        detail_row.setSpacing(8)
        self.toggle_mem_btn = QPushButton(tr(self.lang, "view_memory"))
        self.toggle_mem_btn.setObjectName("TertiaryButton")
        self.toggle_mem_btn.clicked.connect(self._toggle_memory)
        self.clear_pending_btn = QPushButton("")
        self.clear_pending_btn.setObjectName("TertiaryButton")
        self.clear_pending_btn.clicked.connect(self._clear_pending_files)
        detail_row.addWidget(self.toggle_mem_btn)
        detail_row.addWidget(self.clear_pending_btn)
        sidebar_layout.addLayout(detail_row)

        # --- forgetting controls: a privacy-first app needs a forget button ---
        self.forget_title = QLabel("")
        self.forget_title.setObjectName("FieldLabel")
        sidebar_layout.addWidget(self.forget_title)
        forget_row1 = QHBoxLayout()
        forget_row1.setSpacing(8)
        self.forget_chat_btn = QPushButton("")
        self.forget_chat_btn.setObjectName("TertiaryButton")
        self.forget_chat_btn.setAccessibleName("Forget chat memory")
        self.forget_chat_btn.clicked.connect(self._on_forget_chat)
        self.manage_uploads_btn = QPushButton("")
        self.manage_uploads_btn.setObjectName("TertiaryButton")
        self.manage_uploads_btn.setAccessibleName("Manage imported documents")
        self.manage_uploads_btn.clicked.connect(self._on_manage_uploads)
        forget_row1.addWidget(self.forget_chat_btn)
        forget_row1.addWidget(self.manage_uploads_btn)
        sidebar_layout.addLayout(forget_row1)
        self.wipe_all_btn = QPushButton("")
        self.wipe_all_btn.setObjectName("TertiaryButton")
        self.wipe_all_btn.setAccessibleName("Wipe all memory")
        self.wipe_all_btn.clicked.connect(self._on_wipe_all)
        sidebar_layout.addWidget(self.wipe_all_btn)

        self.mem_group = QFrame()
        self.mem_group.setObjectName("MemoryDetails")
        self.mem_group.setMinimumHeight(168)
        mem_layout = QVBoxLayout(self.mem_group)
        mem_layout.setContentsMargins(10, 10, 10, 10)
        mem_layout.setSpacing(8)
        self.mem_view_title = QLabel(tr(self.lang, "memory_title"))
        self.mem_view_title.setObjectName("FieldLabel")
        self.mem_view = QTextEdit()
        self.mem_view.setObjectName("MemoryText")
        self.mem_view.setReadOnly(True)
        self.mem_view.setMinimumHeight(92)
        mem_layout.addWidget(self.mem_view_title)
        mem_layout.addWidget(self.mem_view, 1)
        self.mem_group.setVisible(False)
        sidebar_layout.addWidget(self.mem_group)
        sidebar_layout.addStretch(1)

        self.main_panel = QFrame()
        self.main_panel.setObjectName("MainPanel")
        main_layout = QVBoxLayout(self.main_panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("ChatHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 18, 20, 16)
        header_layout.setSpacing(10)
        header_text = QVBoxLayout()
        header_text.setSpacing(3)
        self.chat_title = QLabel("")
        self.chat_title.setObjectName("ChatTitle")
        self.chat_subtitle = QLabel("")
        self.chat_subtitle.setObjectName("MutedLabel")
        header_text.addWidget(self.chat_title)
        header_text.addWidget(self.chat_subtitle)
        header_layout.addLayout(header_text, 1)
        self.status_pill = QLabel("")
        self.status_pill.setObjectName("StatusPill")
        header_layout.addWidget(self.status_pill)
        main_layout.addWidget(header)

        self.chat_view = QTextBrowser()
        self.chat_view.setObjectName("ChatArea")
        self.chat_view.setAccessibleName("Chat transcript")
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.chat_view, 1)

        composer = QFrame()
        composer.setObjectName("Composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(14, 14, 14, 14)
        composer_layout.setSpacing(10)
        self.input_edit = ComposerEdit()
        self.input_edit.setAccessibleName("Message input")
        self.input_edit.setPlaceholderText(tr(self.lang, "input_placeholder"))
        self.input_edit.send_requested.connect(self._on_send)
        self.send_btn = QPushButton(tr(self.lang, "send"))
        self.send_btn.setObjectName("PrimaryButton")
        self.send_btn.setAccessibleName("Send message")
        self.send_btn.setFixedWidth(104)
        self.send_btn.setFixedHeight(44)
        self.send_btn.clicked.connect(self._on_send)
        composer_layout.addWidget(self.input_edit, 1)
        composer_layout.addWidget(self.send_btn)
        main_layout.addWidget(composer)

        root.addWidget(self.sidebar)
        root.addWidget(self.main_panel, 1)

        self._refresh_static_texts()
        self._refresh_pending_label()
        self._refresh_memory_status()
        self._render_all()
        QTimer.singleShot(0, self.play_intro_animation)

    def _stat_caption_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("StatCaption")
        return label

    def _stat_value_label(self) -> QLabel:
        label = QLabel("0")
        label.setObjectName("StatValue")
        return label

    def _fill_personas(self):
        current = self.persona
        self.persona_combo.blockSignals(True)
        self.persona_combo.clear()
        d = PERSONA_NAMES.get(self.lang, PERSONA_NAMES["zh"])
        self.persona_combo.addItem(d["listener"], "listener")
        self.persona_combo.addItem(d["buddy"], "buddy")
        self.persona_combo.addItem(d["writer"], "writer")
        self._set_combo_by_data(self.persona_combo, current)
        self.persona_combo.blockSignals(False)

    def _populate_model_combo(self, installed_models: list[str]):
        current = self.model_combo.currentData() if self.model_combo.count() else ""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in TARGET_MODELS:
            display_label = RECOMMENDED_MODEL_LABELS.get(m["key"], m["label"])
            self.model_combo.addItem(display_label, m["key"])
            self.model_combo.setItemData(self.model_combo.count() - 1, m["label"], Qt.ToolTipRole)
        extras = [
            name
            for name in installed_models
            if name and is_local_model(name) and not _is_covered_by_recommended_model(name)
        ]
        for name in extras:
            self.model_combo.addItem(f"Installed: {name}", name)
            self.model_combo.setItemData(self.model_combo.count() - 1, name, Qt.ToolTipRole)
        if current and not self._set_combo_by_data(self.model_combo, current):
            # A restored custom model that the async scan didn't list (yet, or
            # anymore) must not be silently replaced by the first entry.
            self.model_combo.addItem(f"Installed: {current}", current)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self.model_combo.blockSignals(False)
        self._on_model_change(self.model_combo.currentIndex())

    def _submit_model_scan(self):
        """Discover installed models off the UI thread."""

        def job(_emit):
            return list_models()

        self.worker.submit("model_scan", job)

    # ---------------- language / theme / persona ----------------

    def _on_lang_change(self, idx):
        self.lang = self.lang_combo.currentData()
        self.mm.ui_lang = self.lang
        if callable(self.on_change_lang):
            self.on_change_lang(self.lang)

        self._refresh_static_texts()
        self.persona_label.setText(tr(self.lang, "select_persona"))
        self._fill_personas()

        self.input_edit.setPlaceholderText(tr(self.lang, "input_placeholder"))
        self.mem_view_title.setText(tr(self.lang, "memory_title"))
        self.drop_area.setText(tr(self.lang, "drop_here"))
        self._refresh_pending_label()
        self._refresh_memory_status()

        self._render_all()
        if self.mem_group.isVisible():
            self._refresh_memory_panel()
        self.state_changed.emit()

    def _on_theme_change(self, idx):
        self.theme = self.theme_combo.currentData()
        if callable(self.on_change_theme):
            self.on_change_theme(self.theme)
        self._render_all()
        self.state_changed.emit()

    def _on_persona_change(self, idx):
        self.persona = self.persona_combo.currentData()
        self.state_changed.emit()

    def _on_model_change(self, idx):
        tip = self.model_combo.itemData(idx, Qt.ToolTipRole)
        self.model_combo.setToolTip(str(tip or self.model_combo.currentText()))

    def _refresh_static_texts(self):
        L = self.lang
        self.tagline.setText(tr(L, "tagline"))
        self.model_label.setText(tr(L, "model_label"))
        self.lang_label.setText(tr(L, "lang_label"))
        self.theme_label.setText(tr(L, "theme"))
        self.memory_title.setText(tr(L, "memory"))
        self.chunk_caption.setText(tr(L, "chunks"))
        self.term_caption.setText(tr(L, "terms"))
        self.upload_caption.setText(tr(L, "uploads"))
        self.last_caption.setText(tr(L, "last"))
        self.build_btn.setText(tr(L, "build_memory"))
        self.analyze_btn.setText(tr(L, "analyze"))
        self.toggle_mem_btn.setText(
            tr(L, "hide_memory") if self.mem_group.isVisible() else tr(L, "view_memory")
        )
        self.clear_pending_btn.setText(tr(L, "clear"))
        self.send_btn.setText(tr(L, "send"))
        self.chat_title.setText(tr(L, "chat"))
        self.chat_subtitle.setText(tr(L, "chat_subtitle"))
        self.forget_title.setText(tr(L, "forget_menu"))
        self.forget_chat_btn.setText(tr(L, "forget_chat"))
        self.manage_uploads_btn.setText(tr(L, "manage_uploads"))
        self.wipe_all_btn.setText(tr(L, "wipe_all"))
        if not bool(self.status_pill.property("busy")):
            self.status_pill.setText(tr(L, "ready"))

    # ---------------- memory panel / status ----------------

    def _toggle_memory(self):
        vis = not self.mem_group.isVisible()
        self.memory_panel_visible = vis
        self.mem_group.setVisible(vis)
        self._refresh_static_texts()
        if vis:
            self._refresh_memory_panel()
            self._fade_in_widget(self.mem_group, duration=180, start=0.0)

    def _refresh_memory_panel(self):
        try:
            self.mem_view.setPlainText(self.mm.debug_view())
        except Exception as e:
            self.mem_view.setPlainText(f"Memory error: {e}")
        self._refresh_memory_status()

    def _refresh_memory_status(self):
        try:
            status = self.mm.status_snapshot()
            last_ts = int(status.get("last_analyze_ts", 0) or 0)
            last_text = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(last_ts))
                if last_ts > 0
                else tr(self.lang, "memory_last_analyze_none")
            )
            memory_dir = str(status.get("memory_dir", "") or "")

            self.chunk_value.setText(str(status.get("chunks", 0)))
            self.term_value.setText(str(status.get("terms", 0)))
            self.upload_value.setText(str(status.get("uploads", 0)))
            self.last_value.setText(last_text)
            self.mem_status_label.setText(tr(self.lang, "memory_root_ready"))
            self.mem_status_label.setToolTip(memory_dir)
        except Exception:
            self.chunk_value.setText("-")
            self.term_value.setText("-")
            self.upload_value.setText("-")
            self.last_value.setText("-")
            self.mem_status_label.setText(tr(self.lang, "memory_status_error"))

    # ---------------- pending files ----------------

    def _on_files_dropped(self, paths: list[str]):
        for p in paths:
            if p not in self.pending_files:
                self.pending_files.append(p)
        self._refresh_pending_label()

    def _clear_pending_files(self):
        self.pending_files = []
        self._refresh_pending_label()

    def _refresh_pending_label(self):
        if not self.pending_files:
            self.pending_label.setText(tr(self.lang, "no_pending_files"))
            return
        names = [os.path.basename(p) or p for p in self.pending_files[:4]]
        suffix = ""
        if len(self.pending_files) > 4:
            suffix = f"\n+ {len(self.pending_files) - 4} {tr(self.lang, 'more_suffix')}"
        label = tr(self.lang, "pending_files")
        self.pending_label.setText(f"{label}:\n" + "\n".join(f"- {n}" for n in names) + suffix)

    # ---------------- memory jobs ----------------

    def _on_build_memory(self):
        mm = self.mm
        pending = list(self.pending_files)
        lang = self.lang
        model = self._resolved_model_quiet()
        self._job_started("build", tr(self.lang, "building"))

        def job(_emit):
            if pending:
                stored = mm.import_files(pending)
                if not stored:
                    return {"kind": "nothing_imported"}
                mm.ingest_new_uploads(stored)
                mm.analyze_memory(model=model, lang=lang)
                return {"kind": "ok", "stats": {
                    "chunks": mm.count_chunks(),
                    "terms": mm.count_terms(),
                    "uploads": len(mm.list_uploads()),
                }}
            return {"kind": "ok", "stats": mm.rebuild_memory()}

        self.worker.submit("build", job)

    def _on_analyze_memory(self):
        mm = self.mm
        lang = self.lang
        model = self._resolved_model_quiet()
        self._job_started("analyze", tr(self.lang, "analyzing"))

        def job(_emit):
            mm.analyze_memory(model=model, lang=lang)
            return {}

        self.worker.submit("analyze", job)

    def _on_forget_chat(self):
        if (
            QMessageBox.question(
                self, tr(self.lang, "forget_chat"), tr(self.lang, "forget_chat_confirm")
            )
            != QMessageBox.Yes
        ):
            return
        mm = self.mm
        self.session_messages = []
        self._chat_messages_html = []
        self._render_all()
        self._job_started("forget", tr(self.lang, "building"))
        self.worker.submit("forget", lambda _emit: mm.clear_chat_memory())
        self.state_changed.emit()

    def _on_manage_uploads(self):
        dlg = UploadsDialog(self.lang, self.mm.list_uploads(), self)
        dlg.delete_requested.connect(self._on_delete_upload)
        dlg.exec()

    def _on_delete_upload(self, path: str):
        mm = self.mm
        self._job_started("forget", tr(self.lang, "building"))
        self.worker.submit("forget", lambda _emit: mm.delete_upload(path))

    def _on_wipe_all(self):
        if (
            QMessageBox.question(
                self, tr(self.lang, "wipe_all"), tr(self.lang, "wipe_all_confirm")
            )
            != QMessageBox.Yes
        ):
            return
        mm = self.mm
        self.session_messages = []
        self._chat_messages_html = []
        self._render_all()
        self._job_started("forget", tr(self.lang, "building"))
        self.worker.submit("forget", lambda _emit: mm.wipe_all_memory())
        self.state_changed.emit()

    # ---------------- chat turn ----------------

    def _resolved_model_quiet(self) -> str | None:
        """Best-effort model name for background jobs; no dialogs, no network.

        Recommended keys resolve against the cached model scan, so Persona
        refresh honors the selected model even before the first chat turn.
        """

        choice_key = self.model_combo.currentData()
        if not choice_key:
            return None
        if _is_recommended_model_key(str(choice_key)):
            name, _ = resolve_installed_model(self._installed_models, str(choice_key))
            return name or None
        return str(choice_key)

    def _on_send(self):
        if self._chat_busy:
            return
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        choice_key = str(self.model_combo.currentData() or "")
        if not choice_key:
            return

        self._inflight_text = text
        self.input_edit.clear()
        self._append_message("user", tr(self.lang, "you"), text)
        self._chat_busy = True
        self._job_started("chat", tr(self.lang, "thinking"))
        self._begin_stream_bubble(self._assistant_display_name())

        mm = self.mm
        lang = self.lang
        style = PERSONA_PROMPTS.get(lang, PERSONA_PROMPTS["zh"]).get(self.persona, "")
        session = list(self.session_messages[-MAX_SESSION_MESSAGES:])
        recommended = _is_recommended_model_key(choice_key)

        def job(emit_chunk):
            installed = list_models()  # raises OllamaConnectionError if down
            if recommended:
                model, candidates = resolve_installed_model(installed, choice_key)
                if not model:
                    raise OllamaModelNotFoundError(candidates[0] if candidates else "<model>")
            else:
                model = choice_key
                if installed and not any(
                    m == model for m in installed if is_local_model(m)
                ):
                    raise OllamaModelNotFoundError(model)

            mm.preferred_model = model
            plan = mm.build_turn_plan(
                user_text=text,
                assistant_style=style,
                lang=lang,
                session_messages=session,
            )
            parts: list[str] = []
            cancel = self.worker.cancel
            for delta in chat_stream(list(plan.get("messages", [])), model=model):
                if cancel.is_set():
                    break
                parts.append(delta)
                emit_chunk(delta)
            reply = "".join(parts).strip()

            fallback_used = False
            if not reply and not cancel.is_set() and str(plan.get("mode")) == "recall":
                # The model produced nothing (e.g. truncated reasoning);
                # fall back to an honest extractive answer from evidence.
                reply = mm.render_fact_answer(
                    text, retrieved_bundle=dict(plan.get("bundle", {})), lang=lang
                )
                fallback_used = reply not in ("无法确定", "Unable to determine from memory.")
                if not fallback_used:
                    reply = ""

            append_error = ""
            if reply:
                try:
                    mm.append_turn(user_text=text, assistant_text=reply)
                except Exception as e:  # memory write failure must not eat the reply
                    append_error = str(e)
            return {
                "reply": reply,
                "fallback_used": fallback_used,
                "append_error": append_error,
                "refresh_due": bool(reply) and mm.refresh_due(),
                "model": model,
            }

        self.worker.submit("chat", job)

    def _assistant_display_name(self) -> str:
        return tr(self.lang, "assistant_name")

    def _submit_background_refresh(self):
        mm = self.mm
        lang = self.lang
        model = self._resolved_model_quiet() or mm.preferred_model

        def job(_emit):
            mm.analyze_memory(model=model, lang=lang)
            mm.mark_refreshed()
            return {}

        self._job_started("auto_analyze", tr(self.lang, "refreshing_memory"))
        self.worker.submit("auto_analyze", job)

    # ---------------- worker callbacks ----------------

    def _on_stream_chunk(self, delta: str):
        if not self._chat_busy:
            return
        self._stream_text += delta
        self._update_stream_bubble()

    def _on_job_done(self, kind: str, result: object):
        if kind == "model_scan":
            self._ollama_reachable = True
            self._installed_models = list(result or [])
            self._populate_model_combo(self._installed_models)
            return

        if kind == "chat":
            data = dict(result or {})
            reply = str(data.get("reply", ""))
            self._finish_stream_bubble(reply or tr(self.lang, "reply_truncated"))
            self._chat_busy = False
            self._job_finished("chat")

            if reply:
                self.session_messages.append({"role": "user", "content": self._inflight_text})
                self.session_messages.append({"role": "assistant", "content": reply})
                self.session_messages = self.session_messages[-MAX_SESSION_MESSAGES:]
            self._inflight_text = ""

            append_error = str(data.get("append_error", ""))
            if append_error:
                QMessageBox.warning(
                    self,
                    tr(self.lang, "memory_error_title"),
                    tr(self.lang, "memory_write_failed") + "\n" + append_error,
                )
            self._refresh_memory_status()
            if self.mem_group.isVisible():
                self._refresh_memory_panel()
            self.state_changed.emit()

            if bool(data.get("refresh_due")):
                self._submit_background_refresh()
            return

        if kind == "build":
            data = dict(result or {})
            self._job_finished("build")
            if data.get("kind") == "nothing_imported":
                QMessageBox.warning(
                    self, tr(self.lang, "build_done"), tr(self.lang, "import_nothing")
                )
            else:
                stats = dict(data.get("stats", {}))
                self.pending_files = []
                self._refresh_pending_label()
                QMessageBox.information(
                    self,
                    tr(self.lang, "build_done"),
                    tr(
                        self.lang,
                        "build_stats",
                        chunks=stats.get("chunks", 0),
                        terms=stats.get("terms", 0),
                        uploads=stats.get("uploads", 0),
                    ),
                )
            self._refresh_memory_status()
            if self.mem_group.isVisible():
                self._refresh_memory_panel()
            return

        if kind in ("analyze", "forget"):
            self._job_finished(kind)
            if kind == "analyze":
                QMessageBox.information(
                    self, tr(self.lang, "analysis_complete"), tr(self.lang, "done")
                )
            self._refresh_memory_status()
            if self.mem_group.isVisible():
                self._refresh_memory_panel()
            return

        if kind == "auto_analyze":
            self._job_finished("auto_analyze")
            self._refresh_memory_status()
            return

    def _on_job_failed(self, kind: str, error_kind: str, message: str):
        if kind == "model_scan":
            self._ollama_reachable = error_kind != "connection"
            return

        if kind == "chat":
            # Roll back the optimistic UI: remove the streaming bubble and the
            # user's bubble, and put the text back into the composer.
            self._chat_busy = False
            self._stream_started = False
            self._stream_text = ""
            if self._chat_messages_html:
                self._chat_messages_html.pop()  # user bubble
            self._render_all()
            self.input_edit.setPlainText(self._inflight_text)
            self._inflight_text = ""
        self._job_finished(kind)

        if error_kind == "connection":
            QMessageBox.warning(
                self, tr(self.lang, "ollama_down_title"), tr(self.lang, "ollama_down_body")
            )
        elif error_kind == "model_missing":
            QMessageBox.warning(
                self,
                tr(self.lang, "model_missing_title"),
                tr(self.lang, "model_missing_body", model=message or "<model>"),
            )
        elif error_kind == "ollama":
            QMessageBox.critical(self, tr(self.lang, "ollama_error_title"), message)
        else:
            QMessageBox.critical(self, tr(self.lang, "memory_error_title"), message)

    # ---------------- chat rendering ----------------

    def _message_fragment(self, role: str, name: str, text: str) -> str:
        role_class = "user" if role == "user" else "assistant"
        return (
            f'<div class="message-row {role_class}">'
            f'<div class="message-bubble {role_class}">'
            f'<div class="message-name">{_escape_text(name)}</div>'
            f'<div class="message-body">{_escape_text(text)}</div>'
            f"</div></div>"
        )

    def _append_message(self, role: str, name: str, text: str):
        fragment = self._message_fragment(role, name, text)
        first = not self._chat_messages_html
        self._chat_messages_html.append(fragment)
        if first:
            self._render_all()
        else:
            cursor = self.chat_view.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertHtml(fragment)
            self._scroll_to_bottom()

    def _begin_stream_bubble(self, name: str):
        self._stream_started = True
        self._stream_text = ""
        if not self._chat_messages_html:
            # ensure the empty-state is cleared before we take a position
            self._render_all()
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._stream_start_pos = cursor.position()
        cursor.insertHtml(self._message_fragment("assistant", name, "…"))
        self._scroll_to_bottom()

    def _update_stream_bubble(self):
        if not self._stream_started:
            return
        cursor = self.chat_view.textCursor()
        cursor.setPosition(min(self._stream_start_pos, self.chat_view.document().characterCount() - 1))
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertHtml(
            self._message_fragment("assistant", self._assistant_display_name(), self._stream_text or "…")
        )
        self._scroll_to_bottom()

    def _finish_stream_bubble(self, final_text: str):
        if not self._stream_started:
            return
        # Render the final text while the stream is still "started" —
        # _update_stream_bubble guards on that flag.
        self._stream_text = final_text
        self._update_stream_bubble()
        self._stream_started = False
        self._stream_text = ""
        self._chat_messages_html.append(
            self._message_fragment("assistant", self._assistant_display_name(), final_text)
        )

    def _scroll_to_bottom(self):
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _render_all(self):
        """Full re-render (theme/lang switches, restore, rollback)."""

        self.chat_view.document().setDefaultStyleSheet(self._chat_css())
        if self._chat_messages_html or self._stream_started:
            self.chat_view.setHtml("".join(self._chat_messages_html))
        else:
            title = tr(self.lang, "empty_title")
            body = tr(self.lang, "empty_body")
            self.chat_view.setHtml(
                f'<div class="empty-state"><h2>{_escape_text(title)}</h2><p>{_escape_text(body)}</p></div>'
            )
        if self._stream_started:
            # A reply is streaming: re-anchor the live bubble at the new end
            # of the document, otherwise the stale position corrupts the
            # transcript on the next chunk.
            cursor = self.chat_view.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._stream_start_pos = cursor.position()
            cursor.insertHtml(
                self._message_fragment(
                    "assistant", self._assistant_display_name(), self._stream_text or "…"
                )
            )
        self._scroll_to_bottom()

    def _chat_css(self) -> str:
        if self.theme == "dark":
            text = "#ecf4f0"
            muted = "#9ba9a3"
            title = "#f4fbf7"
            user_bg = "#56a894"
            user_text = "#061210"
            assistant_bg = "#17211e"
            assistant_text = "#ecf4f0"
            assistant_border = "#2d3b36"
        else:
            text = "#17201d"
            muted = "#65736f"
            title = "#17201d"
            user_bg = "#1f6f62"
            user_text = "#ffffff"
            assistant_bg = "#ffffff"
            assistant_text = "#17201d"
            assistant_border = "#d8e1dd"
        return f"""
        body {{
            margin: 0;
            padding: 22px;
            font-family: "Avenir Next", "PingFang SC", "SF Pro Text", "Helvetica Neue";
            font-size: 14px;
            line-height: 1.58;
            color: {text};
            background: transparent;
        }}
        .message-row {{ display: block; margin: 0 0 14px 0; clear: both; }}
        .message-row.user {{ text-align: right; }}
        .message-row.assistant {{ text-align: left; }}
        .message-bubble {{
            display: inline-block;
            max-width: 74%;
            border-radius: 14px;
            padding: 12px 14px;
            text-align: left;
        }}
        .message-bubble.user {{
            background: {user_bg};
            color: {user_text};
        }}
        .message-bubble.assistant {{
            background: {assistant_bg};
            color: {assistant_text};
            border: 1px solid {assistant_border};
        }}
        .message-name {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0;
            opacity: 0.68;
            margin-bottom: 4px;
        }}
        .message-body {{ white-space: normal; }}
        .empty-state {{
            margin-top: 160px;
            text-align: center;
            color: {muted};
        }}
        .empty-state h2 {{
            color: {title};
            font-size: 24px;
            margin: 0 0 8px 0;
        }}
        .empty-state p {{ margin: 0; }}
        """

    # ---------------- busy / animation ----------------

    def _job_started(self, kind: str, label: str):
        self._active_jobs[kind] = label
        self._sync_busy_ui()

    def _job_finished(self, kind: str):
        self._active_jobs.pop(kind, None)
        self._sync_busy_ui()

    def _sync_busy_ui(self):
        """One source of truth for busy state — a finishing background job
        must not re-enable destructive actions while a chat is streaming."""

        busy = bool(self._active_jobs)
        # The composer stays enabled during background memory jobs; it is
        # only locked while a chat turn is in flight.
        self.send_btn.setEnabled(not self._chat_busy)
        for btn in (
            self.build_btn,
            self.analyze_btn,
            self.forget_chat_btn,
            self.manage_uploads_btn,
            self.wipe_all_btn,
        ):
            btn.setEnabled(not busy)
        label = next(reversed(self._active_jobs.values())) if busy else tr(self.lang, "ready")
        self.status_pill.setText(label)
        self.status_pill.setProperty("busy", busy)
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)
        self._fade_in_widget(self.status_pill, duration=140, start=0.45)

    def play_intro_animation(self):
        self._fade_in_widget(self.sidebar, duration=260, start=0.0)
        self._fade_in_widget(self.main_panel, duration=340, start=0.0)

    def _fade_in_widget(self, widget: QWidget, duration: int = 220, start: float = 0.0):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(start)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(duration)
        anim.setStartValue(start)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        self._animations.append(anim)

        def finish():
            widget.setGraphicsEffect(None)
            if anim in self._animations:
                self._animations.remove(anim)

        anim.finished.connect(finish)
        anim.start()

    def _set_combo_by_data(self, combo: QComboBox, value: str) -> bool:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return True
        return False

    # ---------------- state persistence ----------------

    def export_state(self) -> dict:
        return {
            "lang": self.lang,
            "theme": self.theme,
            "persona": self.persona,
            "model_key": self.model_combo.currentData() or "",
            "input_text": self.input_edit.toPlainText(),
            "chat_messages_html": self._chat_messages_html,
            "session_messages": self.session_messages,
            "pending_files": [p for p in self.pending_files if p and os.path.exists(p)],
            "memory_panel_visible": bool(self.mem_group.isVisible()),
        }

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return

        lang = str(state.get("lang", "") or "")
        theme = str(state.get("theme", "") or "")
        persona = str(state.get("persona", "") or "")
        model_key = str(state.get("model_key", "") or "")

        if lang:
            self._set_combo_by_data(self.lang_combo, lang)
        if theme:
            self._set_combo_by_data(self.theme_combo, theme)
        if persona:
            self._set_combo_by_data(self.persona_combo, persona)
        if model_key:
            self._set_combo_by_data(self.model_combo, model_key)

        raw_fragments = state.get("chat_messages_html", [])
        if isinstance(raw_fragments, list) and raw_fragments:
            self._chat_messages_html = [str(x) for x in raw_fragments if isinstance(x, str)]
        self._render_all()
        if not self._chat_messages_html:
            # Pre-rework app_state.json stored the transcript only as a full
            # HTML document; show it read-only rather than dropping it.
            legacy_html = state.get("chat_html", "")
            if isinstance(legacy_html, str) and legacy_html.strip():
                self.chat_view.setHtml(legacy_html)

        input_text = state.get("input_text", "")
        if isinstance(input_text, str):
            self.input_edit.setPlainText(input_text)

        session_messages: list[dict] = []
        raw_messages = state.get("session_messages", [])
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "") or "")
                content = str(item.get("content", "") or "")
                if role in ("user", "assistant") and content:
                    session_messages.append({"role": role, "content": content})
        self.session_messages = session_messages[-MAX_SESSION_MESSAGES:]

        raw_pending = state.get("pending_files", [])
        pending_files: list[str] = []
        if isinstance(raw_pending, list):
            for p in raw_pending:
                if not isinstance(p, str):
                    continue
                if p and os.path.exists(p) and p.lower().endswith((".txt", ".docx")):
                    pending_files.append(p)
        self.pending_files = pending_files
        self._refresh_pending_label()

        memory_visible = bool(state.get("memory_panel_visible", False))
        self.memory_panel_visible = memory_visible
        self.mem_group.setVisible(memory_visible)
        self.toggle_mem_btn.setText(
            tr(self.lang, "hide_memory") if memory_visible else tr(self.lang, "view_memory")
        )
        if memory_visible:
            self._refresh_memory_panel()
        else:
            self._refresh_memory_status()
