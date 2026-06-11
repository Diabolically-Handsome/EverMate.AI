
# views/wizard.py
import json
import os
import sys
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsOpacityEffect, QMessageBox, QStackedLayout, QStackedWidget
)
from engine.storage import write_text
from i18n_qt import tr, APP_TITLE
from runtime_paths import resource_path, user_app_support_root
from .chat import ChatPage, InstanceLockedError
from .effects import ParticleCanvas, fade_in, slide_fade_in

THEME_ACCENTS = {"light": "#1f6f62", "dark": "#56a894"}

def _load_stylesheet(theme: str) -> str:
    fname = resource_path("assets", f"style_{theme}.qss")
    if fname.exists():
        return fname.read_text(encoding="utf-8")
    # fallback
    return resource_path("assets", "style_light.qss").read_text(encoding="utf-8")

EXTRA_QSS = """
QComboBox::drop-down { width: 26px; border: none; }
QComboBox::down-arrow {
    subcontrol-origin: padding;
    subcontrol-position: right center;
    margin-right: 7px;
}
"""


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _save_json(path: str, data: dict) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        icon_path = resource_path("assets", "icons", "app_icon.svg")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            self.setWindowIcon(QIcon(str(resource_path("assets", "icons", "app_icon_256.png"))))
        self.setMinimumSize(960, 640)
        self.lang = "en"
        self.theme = "dark"
        self._animations: list[QPropertyAnimation] = []

        self.stack = QStackedWidget()
        self.page_welcome = self._build_welcome()
        try:
            self.page_chat = ChatPage(on_change_lang=self._on_change_lang, on_change_theme=self._on_change_theme)
        except InstanceLockedError:
            QMessageBox.critical(
                self,
                tr(self.lang, "instance_locked_title"),
                tr(self.lang, "instance_locked_body"),
            )
            sys.exit(1)
        self.stack.addWidget(self.page_welcome)
        self.stack.addWidget(self.page_chat)

        # Persist state shortly after every meaningful change, not only on a
        # clean close — a crash used to lose the whole session.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1500)
        self._save_timer.timeout.connect(self._save_state)
        self.page_chat.state_changed.connect(lambda: self._save_timer.start())

        root = QWidget()
        root.setObjectName("AppRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18,18,18,18)
        layout.setSpacing(0)
        layout.addWidget(self.stack)
        self.setCentralWidget(root)

        self.apply_theme(self.theme)
        self.resize(1180, 760)
        self._restore_state()

        def intro():
            self._fade_in_widget(self.stack.currentWidget(), duration=300)
            if self.stack.currentWidget() is self.page_welcome:
                self.play_welcome_entrance()

        QTimer.singleShot(0, intro)

    def _build_welcome(self) -> QWidget:
        w = QWidget()
        w.setObjectName("start-page")

        # Two stacked layers: the "memory constellation" canvas behind, the
        # text content in front.
        stack = QStackedLayout(w)
        stack.setStackingMode(QStackedLayout.StackAll)

        self.welcome_canvas = ParticleCanvas(w)
        self._apply_canvas_palette()

        content = QWidget(w)
        content.setObjectName("WelcomeContent")
        content.setAttribute(Qt.WA_StyledBackground, False)
        v = QVBoxLayout(content)
        v.setContentsMargins(44,44,44,44)
        v.setSpacing(12)
        title = QLabel(APP_TITLE)
        title.setObjectName("BigTitle")
        subtitle = QLabel(tr(self.lang, "subtitle"))
        subtitle.setObjectName("Subtitle")
        note = QLabel("Local memory workspace" if self.lang == "en" else "本地长记忆工作台")
        note.setObjectName("MutedLabel")
        btn = QPushButton(tr(self.lang, "get_started"))
        btn.setObjectName("PrimaryButton")
        btn.setFixedSize(168, 46)
        def go():
            self.stack.setCurrentIndex(1)
            self.page_chat.play_intro_animation()
        btn.clicked.connect(go)
        v.addStretch(1)
        v.addWidget(title)
        v.addWidget(subtitle)
        v.addWidget(note)
        v.addSpacing(16)
        v.addWidget(btn, alignment=Qt.AlignLeft)
        v.addStretch(2)

        stack.addWidget(content)
        stack.addWidget(self.welcome_canvas)
        stack.setCurrentWidget(content)

        self._welcome_entrance_widgets = [title, subtitle, note, btn]
        return w

    def play_welcome_entrance(self) -> None:
        """Staggered entrance for the welcome page text."""

        for i, widget in enumerate(getattr(self, "_welcome_entrance_widgets", [])):
            slide_fade_in(widget, dy=22, duration=420, delay=90 * i, keep=self._animations)

    # --- theme & language ---
    def apply_theme(self, theme: str):
        self.theme = theme
        qss = _load_stylesheet(theme) + "\n" + EXTRA_QSS
        self.setStyleSheet(qss)
        self._apply_canvas_palette()

    def _apply_canvas_palette(self):
        canvas = getattr(self, "welcome_canvas", None)
        if canvas is None:
            return
        accent = THEME_ACCENTS.get(self.theme, "#1f6f62")
        if self.theme == "light":
            # Light backgrounds wash the constellation out; paint it bolder.
            canvas.set_palette(accent, line_alpha=120, node_alpha=190)
        else:
            canvas.set_palette(accent)

    def _fade_in_widget(self, widget: QWidget, duration: int = 240):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        self._animations.append(anim)

        def finish():
            widget.setGraphicsEffect(None)
            if anim in self._animations:
                self._animations.remove(anim)

        anim.finished.connect(finish)
        anim.start()

    def _on_change_lang(self, lang: str):
        self.lang = lang
        # only welcome page text uses i18n here; chat page handles itself
        # rebuild welcome page quickly to update texts
        was_current = self.stack.currentWidget() is self.page_welcome
        self.stack.removeWidget(self.page_welcome)
        self.page_welcome.deleteLater()
        self.page_welcome = self._build_welcome()
        self.stack.insertWidget(0, self.page_welcome)
        if was_current:
            self.stack.setCurrentIndex(0)
            self._fade_in_widget(self.page_welcome, duration=220)
            QTimer.singleShot(0, self.play_welcome_entrance)

    def _on_change_theme(self, theme: str):
        self.apply_theme(theme)

    # --- app state ---
    def _state_path(self) -> str:
        try:
            memory_root = str(self.page_chat.mm.memory_dir)
        except Exception:
            memory_root = ""
        if not memory_root:
            memory_root = str(user_app_support_root() / "memory")
        return os.path.join(memory_root, "app_state.json")

    def _collect_state(self) -> dict:
        return {
            "version": 1,
            "active_page": int(self.stack.currentIndex()),
            "window": {
                "width": int(self.width()),
                "height": int(self.height()),
                "x": int(self.x()),
                "y": int(self.y()),
                "is_maximized": bool(self.isMaximized()),
            },
            "chat": self.page_chat.export_state(),
        }

    def _restore_state(self) -> None:
        data = _load_json(self._state_path())
        if not data:
            return

        chat_state = data.get("chat", {})
        if isinstance(chat_state, dict):
            self.page_chat.restore_state(chat_state)

        idx = data.get("active_page", 0)
        if isinstance(idx, int) and 0 <= idx < self.stack.count():
            self.stack.setCurrentIndex(idx)

        window = data.get("window", {})
        if isinstance(window, dict):
            width = int(window.get("width", 0) or 0)
            height = int(window.get("height", 0) or 0)
            if width >= 640 and height >= 420:
                self.resize(width, height)

            x = window.get("x", None)
            y = window.get("y", None)
            if isinstance(x, int) and isinstance(y, int) and self._position_on_screen(x, y):
                self.move(x, y)

            if bool(window.get("is_maximized", False)):
                self.showMaximized()

    @staticmethod
    def _position_on_screen(x: int, y: int) -> bool:
        """Reject saved positions on monitors that are no longer attached."""

        from PySide6.QtGui import QGuiApplication

        for screen in QGuiApplication.screens():
            geo = screen.availableGeometry()
            if geo.contains(x + 40, y + 40):
                return True
        return False

    def _save_state(self) -> None:
        try:
            _save_json(self._state_path(), self._collect_state())
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_state()
        try:
            self.page_chat.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
