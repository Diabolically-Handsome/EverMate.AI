# views/bubbles.py
"""Widget-based chat transcript.

Qt's rich-text engine cannot render real chat bubbles (no border-radius,
no max-width, no inline-block), which produced full-width color bars and
merged lines. This view builds each message as an actual widget: rounded
QFrame bubbles, 74%-width cap, left/right alignment, selectable text, and
clean streaming updates.
"""

from __future__ import annotations

import html
import re
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from views.effects import SmoothScroller, fade_in

BUBBLE_WIDTH_RATIO = 0.74

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_CODE = re.compile(r"`([^`\n]+)`")


def render_markdown_lite(text: str) -> str:
    """Escape, then apply the tiny markdown subset models actually use.

    Bold and inline code only — everything else stays literal. Escaping
    happens first, so message content can never inject markup.
    """

    safe = html.escape(text or "")
    safe = _MD_BOLD.sub(r"<b>\1</b>", safe)
    safe = _MD_CODE.sub(r"<code>\1</code>", safe)
    return safe.replace("\n", "<br>")


class MessageBubble(QFrame):
    def __init__(self, role: str, name: str, text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.role = role
        self.setObjectName("BubbleUser" if role == "user" else "BubbleAssistant")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(4)

        self.name_label = QLabel(name)
        self.name_label.setObjectName("BubbleName")
        self.body_label = QLabel()
        self.body_label.setObjectName("BubbleBody")
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(Qt.RichText)
        self.body_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.body_label.setOpenExternalLinks(False)

        layout.addWidget(self.name_label)
        layout.addWidget(self.body_label)
        self.set_text(text)

    def set_text(self, text: str) -> None:
        self.body_label.setText(render_markdown_lite(text))


class ChatLogView(QScrollArea):
    """Scrollable column of message bubbles with an empty-state placeholder."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatArea")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container.setObjectName("ChatLogContainer")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(22, 22, 22, 22)
        self._layout.setSpacing(14)
        self._layout.addStretch(1)
        self.setWidget(self._container)

        self._rows: List[QWidget] = []
        self._bubbles: List[MessageBubble] = []
        self.scroller = SmoothScroller(self.verticalScrollBar())
        self._animations: list = []

        self._empty = QWidget(self._container)
        empty_layout = QVBoxLayout(self._empty)
        empty_layout.setContentsMargins(0, 140, 0, 0)
        empty_layout.setSpacing(8)
        self.empty_title = QLabel("")
        self.empty_title.setObjectName("EmptyTitle")
        self.empty_title.setAlignment(Qt.AlignCenter)
        self.empty_body = QLabel("")
        self.empty_body.setObjectName("EmptyBody")
        self.empty_body.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(self.empty_title)
        empty_layout.addWidget(self.empty_body)
        self._layout.insertWidget(0, self._empty)

    # ---------------- content ----------------

    def set_empty_texts(self, title: str, body: str) -> None:
        self.empty_title.setText(title)
        self.empty_body.setText(body)

    def add_message(self, role: str, name: str, text: str, animate: bool = True) -> MessageBubble:
        self._empty.setVisible(False)
        bubble = MessageBubble(role, name, text)
        self._apply_max_width(bubble)

        row = QWidget(self._container)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch(1)

        # insert above the trailing stretch
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._rows.append(row)
        self._bubbles.append(bubble)
        if animate:
            fade_in(bubble, duration=200, keep=self._animations)
        self.scroll_to_bottom()
        return bubble

    def remove_last(self) -> None:
        if not self._rows:
            return
        row = self._rows.pop()
        self._bubbles.pop()
        self._layout.removeWidget(row)
        row.deleteLater()
        if not self._rows:
            self._empty.setVisible(True)

    def clear_messages(self) -> None:
        while self._rows:
            self.remove_last()

    def last_bubble(self) -> Optional[MessageBubble]:
        return self._bubbles[-1] if self._bubbles else None

    def scroll_to_bottom(self) -> None:
        # Defer one tick so the new row has a real height before scrolling.
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: self.scroller.to_bottom())

    # ---------------- sizing ----------------

    def _apply_max_width(self, bubble: MessageBubble) -> None:
        usable = max(220, int(self.viewport().width() * BUBBLE_WIDTH_RATIO))
        bubble.setMaximumWidth(usable)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for bubble in self._bubbles:
            self._apply_max_width(bubble)
