# views/chat.py

from __future__ import annotations

import html
import os
import time
import traceback

import requests
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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
from models_config import TARGET_MODELS, resolve_installed_model
from ollama_client import OLLAMA_URL, chat as ollama_chat


PERSONA_NAMES = {
    "zh": {"listener": "温柔倾听者", "buddy": "聊天搭子", "writer": "创意作家"},
    "en": {"listener": "Gentle Listener", "buddy": "Chat Buddy", "writer": "Creative Writer"},
}


PERSONA_PROMPTS = {
    "listener": "您是一位温柔体贴的本地 AI 朋友“小枢”。先用一句话复述要点，再用 1–2 句表达理解，然后给出不超过 2 条可操作建议。语气温柔，输出 4–8 句。",
    "buddy": "您是一位轻松幽默、实用主义的聊天搭子“小枢”。先确认目标，再给出简洁可执行的步骤清单。每条不超过 20 字。",
    "writer": "您是一位善于结构化表达的创意作家“小枢”。用小标题 + 列表组织内容，重点在逻辑清晰与表达精炼。",
}


MAX_SESSION_MESSAGES = 12  # keep last 6 turns (user+assistant)

RECOMMENDED_MODEL_LABELS = {
    "deepseek_qwen3_8b": "DeepSeek R1 8B · Easy",
    "qwen3_30b_a3b": "Qwen3 30B-A3B · Fast",
    "deepseek_r1_70b": "DeepSeek R1 70B · Stable",
    "gpt_oss_120b": "gpt-oss 120B · Best",
}


def _installed_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


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
            paths = [u.toLocalFile() for u in md.urls()]
            if any(p and p.lower().endswith((".txt", ".docx")) for p in paths):
                event.acceptProposedAction()

    def dropEvent(self, event):
        md = event.mimeData()
        if not md.hasUrls():
            return
        paths = [u.toLocalFile() for u in md.urls()]
        paths = [p for p in paths if p and p.lower().endswith((".txt", ".docx"))]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


class ChatPage(QWidget):
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

        self.mm = MemoryManager()

        self._build_ui()

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
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_change)
        sidebar_layout.addWidget(self.model_label)
        sidebar_layout.addWidget(self.model_combo)

        self.persona_label = QLabel(tr(self.lang, "select_persona"))
        self.persona_label.setObjectName("FieldLabel")
        self.persona_combo = QComboBox()
        self.persona_combo.setObjectName("ControlCombo")
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
        self.build_btn.clicked.connect(self._on_build_memory)
        self.analyze_btn = QPushButton(tr(self.lang, "analyze_memory"))
        self.analyze_btn.setObjectName("SecondaryButton")
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
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.chat_view, 1)

        composer = QFrame()
        composer.setObjectName("Composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(14, 14, 14, 14)
        composer_layout.setSpacing(10)
        self.input_edit = ComposerEdit()
        self.input_edit.setPlaceholderText(tr(self.lang, "input_placeholder"))
        self.input_edit.send_requested.connect(self._on_send)
        self.send_btn = QPushButton(tr(self.lang, "send"))
        self.send_btn.setObjectName("PrimaryButton")
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
        self._render_empty_chat()
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

    def _populate_model_combo(self):
        self.model_combo.clear()

        for m in TARGET_MODELS:
            display_label = RECOMMENDED_MODEL_LABELS.get(m["key"], m["label"])
            self.model_combo.addItem(display_label, m["key"])
            self.model_combo.setItemData(self.model_combo.count() - 1, m["label"], Qt.ToolTipRole)

        installed_models = _installed_ollama_models()
        extras = [
            name
            for name in installed_models
            if name and not _is_covered_by_recommended_model(name)
        ]
        for name in extras:
            self.model_combo.addItem(f"Installed: {name}", name)
            self.model_combo.setItemData(self.model_combo.count() - 1, name, Qt.ToolTipRole)

        self._on_model_change(self.model_combo.currentIndex())

    def _on_lang_change(self, idx):
        self.lang = self.lang_combo.currentData()
        if callable(self.on_change_lang):
            self.on_change_lang(self.lang)

        self._refresh_static_texts()
        self.persona_label.setText(tr(self.lang, "select_persona"))
        self._fill_personas()

        self.input_edit.setPlaceholderText(tr(self.lang, "input_placeholder"))
        self.mem_view_title.setText(tr(self.lang, "memory_title"))
        self.drop_area.setText(tr(self.lang, "drop_here"))
        self._refresh_static_texts()
        self._refresh_pending_label()
        self._refresh_memory_status()

        if not self._chat_messages_html:
            self._render_empty_chat()
        if self.mem_group.isVisible():
            self._refresh_memory_panel()

    def _on_theme_change(self, idx):
        self.theme = self.theme_combo.currentData()
        if callable(self.on_change_theme):
            self.on_change_theme(self.theme)
        if self._chat_messages_html:
            self._render_chat_html()
        else:
            self._render_empty_chat()

    def _refresh_static_texts(self):
        zh = self.lang == "zh"
        self.tagline.setText("本地长记忆工作台" if zh else "Local memory workspace")
        self.model_label.setText("模型" if zh else "Model")
        self.lang_label.setText("语言" if zh else "Language")
        self.theme_label.setText("主题" if zh else "Theme")
        self.memory_title.setText("记忆状态" if zh else "Memory")
        self.chunk_caption.setText("片段" if zh else "Chunks")
        self.term_caption.setText("术语" if zh else "Terms")
        self.upload_caption.setText("文件" if zh else "Uploads")
        self.last_caption.setText("最近" if zh else "Last")
        self.build_btn.setText(tr(self.lang, "build_memory") if zh else "Build Memory")
        self.analyze_btn.setText(tr(self.lang, "analyze_memory") if zh else "Analyze")
        self.toggle_mem_btn.setText(
            (tr(self.lang, "hide_memory") if zh else "Hide Details")
            if self.mem_group.isVisible()
            else (tr(self.lang, "view_memory") if zh else "View Details")
        )
        self.clear_pending_btn.setText("清空" if zh else "Clear")
        self.send_btn.setText(tr(self.lang, "send"))
        self.chat_title.setText("聊天" if zh else "Chat")
        self.chat_subtitle.setText(
            "本地记忆、模型与对话在这里汇合。"
            if zh
            else "Local memory, model routing, and chat in one workspace."
        )
        if not bool(self.status_pill.property("busy")):
            self.status_pill.setText("就绪" if zh else "Ready")

    def _on_persona_change(self, idx):
        self.persona = self.persona_combo.currentData()

    def _on_model_change(self, idx):
        tip = self.model_combo.itemData(idx, Qt.ToolTipRole)
        self.model_combo.setToolTip(str(tip or self.model_combo.currentText()))

    def _toggle_memory(self):
        vis = not self.mem_group.isVisible()
        self.memory_panel_visible = vis
        self.mem_group.setVisible(vis)
        self._refresh_static_texts()
        if vis:
            self._refresh_memory_panel()
            self._fade_in_widget(self.mem_group, duration=180, start=0.0)

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
            self.pending_label.setText("No pending files" if self.lang == "en" else "暂无待导入文件")
            return

        names = [os.path.basename(p) or p for p in self.pending_files[:4]]
        suffix = ""
        if len(self.pending_files) > 4:
            suffix = f"\n+ {len(self.pending_files) - 4} more"
        label = "Pending files" if self.lang == "en" else "待导入文件"
        self.pending_label.setText(f"{label}:\n" + "\n".join(f"- {name}" for name in names) + suffix)

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
            last_text = time.strftime("%Y-%m-%d %H:%M", time.localtime(last_ts)) if last_ts > 0 else tr(self.lang, "memory_last_analyze_none")
            memory_dir = str(status.get("memory_dir", "") or "")

            self.chunk_value.setText(str(status.get("chunks", 0)))
            self.term_value.setText(str(status.get("terms", 0)))
            self.upload_value.setText(str(status.get("uploads", 0)))
            self.last_value.setText(last_text)
            self.mem_status_label.setText("记忆目录已就绪" if self.lang == "zh" else "Memory root ready")
            self.mem_status_label.setToolTip(memory_dir)
        except Exception:
            self.chunk_value.setText("-")
            self.term_value.setText("-")
            self.upload_value.setText("-")
            self.last_value.setText("-")
            self.mem_status_label.setText(tr(self.lang, "memory_status_error"))

    def _on_build_memory(self):
        try:
            self._set_busy(True, "构建中" if self.lang == "zh" else "Building")
            if self.pending_files:
                stored = self.mm.import_files(self.pending_files)
                if not stored:
                    QMessageBox.warning(self, "Memory", "未导入任何文件（仅支持 .txt/.docx）。")
                    return

            stats = self.mm.rebuild_memory()
            self.pending_files = []
            self._refresh_pending_label()

            msg = (
                f"Build/Rebuild 完成：\n"
                f"- Uploads: {stats.get('uploads', 0)}\n"
                f"- Chunks: {stats.get('chunks', 0)}\n"
                f"- Terms: {stats.get('terms', 0)}\n"
            )
            QMessageBox.information(self, tr(self.lang, "analysis_complete"), msg)
            self._refresh_memory_status()

            if self.mem_group.isVisible():
                self._refresh_memory_panel()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Memory Error", str(e))
        finally:
            self._set_busy(False)

    def _on_analyze_memory(self):
        try:
            self._set_busy(True, "分析中" if self.lang == "zh" else "Analyzing")
            self.mm.analyze_memory()
            QMessageBox.information(self, tr(self.lang, "analysis_complete"), "OK")
            self._refresh_memory_status()
            if self.mem_group.isVisible():
                self._refresh_memory_panel()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Memory Error", str(e))
        finally:
            self._set_busy(False)

    def _current_model_name(self) -> str:
        choice_key = self.model_combo.currentData()
        if not choice_key:
            return ""

        installed = _installed_ollama_models()
        if _is_recommended_model_key(str(choice_key)):
            name, candidates = resolve_installed_model(installed, choice_key)
            if name:
                return name

            rec = candidates[0] if candidates else "<model>"
            QMessageBox.warning(
                self,
                "模型未安装",
                f"未检测到所选模型，请先在终端执行：\n\n  ollama pull {rec}\n",
            )
            return ""

        exact_name = str(choice_key)
        if installed and exact_name not in installed:
            QMessageBox.warning(
                self,
                "模型未安装",
                f"未检测到该本地模型，请先在终端执行：\n\n  ollama pull {exact_name}\n",
            )
            return ""

        return exact_name

    def _on_send(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        model = self._current_model_name()
        if not model:
            return

        self._append_message("user", "您" if self.lang == "zh" else "You", text)
        self.input_edit.clear()
        self._set_busy(True, "思考中" if self.lang == "zh" else "Thinking")

        assistant_style = PERSONA_PROMPTS.get(self.persona, "")
        try:
            plan = self.mm.build_turn_plan(
                user_text=text,
                assistant_style=assistant_style,
                lang=self.lang,
                session_messages=self.session_messages[-MAX_SESSION_MESSAGES:],
            )
            messages = list(plan.get("messages", []))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Memory", f"构建记忆提示失败，将继续无记忆对话：\n{e}")
            messages = [{"role": "system", "content": assistant_style or ""}, {"role": "user", "content": text}]
            plan = {"two_pass": False}

        try:
            if str(plan.get("direct_answer", "")).strip():
                reply = str(plan.get("direct_answer", "")).strip()
            elif bool(plan.get("two_pass")) and str(plan.get("mode", "")) == "multi_hop":
                synthesis_text = ollama_chat(
                    list(plan.get("synthesis_messages", [])),
                    model=model,
                    options={"temperature": 0, "num_predict": 384},
                )
                synthesis = self.mm.parse_multi_hop_synthesis(synthesis_text, str(plan.get("question_subtype") or ""))
                if synthesis is not None:
                    synthesis = self.mm.repair_multi_hop_synthesis(
                        user_text=text,
                        synthesis=synthesis,
                        retrieved_bundle=dict(plan.get("bundle", {})),
                    )
                    answer_messages = self.mm.build_multi_hop_answer_messages(
                        user_text=text,
                        assistant_style=assistant_style,
                        synthesis=synthesis,
                        lang=self.lang,
                        retrieved_bundle=dict(plan.get("bundle", {})),
                    )
                else:
                    answer_messages = list(plan.get("fallback_messages", messages))
                reply = ollama_chat(answer_messages, model=model, options={"temperature": 0, "num_predict": 224})
                if synthesis is not None and self.mm.needs_multi_hop_answer_fallback(reply, synthesis, text):
                    reply = self.mm.render_multi_hop_answer(text, synthesis)
            else:
                reply = ollama_chat(messages, model=model)
                if str(plan.get("mode", "")) == "fact" and self.mm.needs_fact_answer_fallback(
                    reply,
                    dict(plan.get("bundle", {})),
                    text,
                ):
                    reply = self.mm.render_fact_answer(text, retrieved_bundle=dict(plan.get("bundle", {})))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Ollama 错误", str(e))
            self._set_busy(False)
            return

        self._append_message("assistant", "小枢", reply)

        self.session_messages.append({"role": "user", "content": text})
        self.session_messages.append({"role": "assistant", "content": reply})
        self.session_messages = self.session_messages[-MAX_SESSION_MESSAGES:]

        try:
            self.mm.append_turn(user_text=text, assistant_text=reply)
        except Exception as e:
            QMessageBox.warning(self, "Memory", f"写入记忆失败（对话仍可继续）：\n{e}")
        self._refresh_memory_status()

        if self.mem_group.isVisible():
            self._refresh_memory_panel()

        self._set_busy(False)

    def _append_message(self, role: str, name: str, text: str):
        if not self._chat_messages_html:
            self.chat_view.clear()
        role_class = "user" if role == "user" else "assistant"
        safe_name = _escape_text(name)
        safe_text = _escape_text(text)
        fragment = (
            f'<div class="message-row {role_class}">'
            f'<div class="message-bubble {role_class}">'
            f'<div class="message-name">{safe_name}</div>'
            f'<div class="message-body">{safe_text}</div>'
            f"</div></div>"
        )
        self._chat_messages_html.append(fragment)
        self._render_chat_html()

    def _render_empty_chat(self):
        title = "准备好了" if self.lang == "zh" else "Ready when you are"
        body = "选择模型后开始对话，或先导入记忆文件。" if self.lang == "zh" else "Choose a model, start chatting, or import memory files first."
        self.chat_view.setHtml(
            self._html_shell(
                f'<div class="empty-state"><h2>{_escape_text(title)}</h2><p>{_escape_text(body)}</p></div>'
            )
        )

    def _render_chat_html(self):
        self.chat_view.setHtml(self._html_shell("".join(self._chat_messages_html)))
        self.chat_view.moveCursor(QTextCursor.End)

    def _html_shell(self, content: str) -> str:
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
        <html>
        <head>
        <style>
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
        </style>
        </head>
        <body>{content}</body>
        </html>
        """

    def _set_busy(self, busy: bool, label: str = ""):
        self.send_btn.setEnabled(not busy)
        self.build_btn.setEnabled(not busy)
        self.analyze_btn.setEnabled(not busy)
        self.status_pill.setText(label if busy else ("就绪" if self.lang == "zh" else "Ready"))
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

    def export_state(self) -> dict:
        return {
            "lang": self.lang,
            "theme": self.theme,
            "persona": self.persona,
            "model_key": self.model_combo.currentData() or "",
            "input_text": self.input_edit.toPlainText(),
            "chat_html": self.chat_view.toHtml(),
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

        raw_messages = state.get("session_messages", [])
        has_raw_session = False
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if isinstance(item, dict) and str(item.get("role", "") or "") in ("user", "assistant"):
                    if str(item.get("content", "") or "").strip():
                        has_raw_session = True
                        break

        raw_fragments = state.get("chat_messages_html", [])
        if isinstance(raw_fragments, list) and raw_fragments:
            self._chat_messages_html = [str(x) for x in raw_fragments if isinstance(x, str)]
            self._render_chat_html()
        else:
            chat_html = state.get("chat_html", "")
            if has_raw_session and isinstance(chat_html, str) and chat_html.strip():
                self.chat_view.setHtml(chat_html)
            else:
                self._render_empty_chat()

        input_text = state.get("input_text", "")
        if isinstance(input_text, str):
            self.input_edit.setPlainText(input_text)

        session_messages: list[dict] = []
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
        self.toggle_mem_btn.setText(tr(self.lang, "hide_memory") if memory_visible else tr(self.lang, "view_memory"))
        if memory_visible:
            self._refresh_memory_panel()
        else:
            self._refresh_memory_status()

        if not self._chat_messages_html and not self.chat_view.toPlainText().strip():
            self._render_empty_chat()
