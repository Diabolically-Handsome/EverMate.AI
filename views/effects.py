# views/effects.py
"""Reusable animation helpers and decorative widgets for the EverMate GUI.

Design rules:
- Every animation is lightweight (opacity / pos / int interpolation, or one
  ~30fps timer on the welcome page only) — the UI thread must stay free.
- A widget holds at most one QGraphicsEffect at a time; helpers that attach
  effects always clean up after themselves.
- All colors come in from the caller so light/dark themes stay consistent.
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QTimer,
    QVariantAnimation,
    Qt,
)
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QWidget


def fade_in(
    widget: QWidget,
    duration: int = 220,
    start: float = 0.0,
    delay: int = 0,
    keep: Optional[list] = None,
) -> None:
    """Fade a widget in, removing the effect when done.

    `keep` is a caller-owned list that retains the animation object so it is
    not garbage-collected mid-flight.
    """

    def begin():
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(start)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(start)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        if keep is not None:
            keep.append(anim)

        def finish():
            widget.setGraphicsEffect(None)
            if keep is not None and anim in keep:
                keep.remove(anim)

        anim.finished.connect(finish)
        anim.start()

    if delay > 0:
        QTimer.singleShot(delay, widget, begin)
    else:
        begin()


def slide_fade_in(
    widget: QWidget,
    dy: int = 18,
    dx: int = 0,
    duration: int = 340,
    delay: int = 0,
    keep: Optional[list] = None,
) -> None:
    """Entrance: slide from an offset back to the layout position + fade.

    Captures the layout-assigned position at start time, so call it after the
    layout has settled (e.g. via QTimer.singleShot(0, ...)).
    """

    def begin():
        end_pos = widget.pos()
        widget.move(end_pos + QPoint(dx, dy))
        pos_anim = QPropertyAnimation(widget, b"pos", widget)
        pos_anim.setDuration(duration)
        pos_anim.setStartValue(end_pos + QPoint(dx, dy))
        pos_anim.setEndValue(end_pos)
        pos_anim.setEasingCurve(QEasingCurve.OutCubic)
        if keep is not None:
            keep.append(pos_anim)

        def finish():
            if keep is not None and pos_anim in keep:
                keep.remove(pos_anim)

        pos_anim.finished.connect(finish)
        pos_anim.start()
        fade_in(widget, duration=duration, keep=keep)

    if delay > 0:
        QTimer.singleShot(delay, widget, begin)
    else:
        begin()


class PulseController:
    """Looping breathing effect (opacity 1.0 ⇄ 0.45) for a busy indicator."""

    def __init__(self, widget: QWidget):
        self.widget = widget
        self._anim: Optional[QPropertyAnimation] = None

    @property
    def active(self) -> bool:
        return self._anim is not None

    def start(self) -> None:
        if self._anim is not None:
            return
        effect = QGraphicsOpacityEffect(self.widget)
        effect.setOpacity(1.0)
        self.widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self.widget)
        anim.setDuration(1100)
        anim.setStartValue(1.0)
        anim.setKeyValueAt(0.5, 0.45)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutSine)
        anim.setLoopCount(-1)
        anim.start()
        self._anim = anim

    def stop(self) -> None:
        if self._anim is None:
            return
        self._anim.stop()
        self._anim = None
        self.widget.setGraphicsEffect(None)


def glow_pulse(
    widget: QWidget,
    color: QColor,
    max_blur: int = 26,
    duration: int = 650,
    keep: Optional[list] = None,
) -> None:
    """One soft glow swell-and-fade around a widget (e.g. "memory updated")."""

    if widget.graphicsEffect() is not None:
        return  # never fight another live effect
    effect = QGraphicsDropShadowEffect(widget)
    effect.setColor(color)
    effect.setOffset(0, 0)
    effect.setBlurRadius(0)
    widget.setGraphicsEffect(effect)
    anim = QVariantAnimation(widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setKeyValueAt(0.4, float(max_blur))
    anim.setEndValue(0.0)
    anim.setEasingCurve(QEasingCurve.InOutQuad)
    anim.valueChanged.connect(lambda v: effect.setBlurRadius(float(v)))
    if keep is not None:
        keep.append(anim)

    def finish():
        widget.setGraphicsEffect(None)
        if keep is not None and anim in keep:
            keep.remove(anim)

    anim.finished.connect(finish)
    anim.start()


def count_up(
    set_text: Callable[[str], None],
    old: int,
    new: int,
    duration: int = 520,
    parent: Optional[QWidget] = None,
    keep: Optional[list] = None,
) -> None:
    """Animate a numeric label from old to new."""

    if old == new:
        set_text(str(new))
        return
    anim = QVariantAnimation(parent)
    anim.setDuration(duration)
    anim.setStartValue(int(old))
    anim.setEndValue(int(new))
    anim.setEasingCurve(QEasingCurve.OutCubic)
    anim.valueChanged.connect(lambda v: set_text(str(int(v))))
    if keep is not None:
        keep.append(anim)
        anim.finished.connect(lambda: anim in keep and keep.remove(anim))
    anim.start()


class SmoothScroller:
    """Animated scroll-to-bottom for a scrollbar (no more teleporting)."""

    def __init__(self, scrollbar):
        self.bar = scrollbar
        self._anim: Optional[QPropertyAnimation] = None

    def to_bottom(self, duration: int = 180) -> None:
        target = self.bar.maximum()
        if self.bar.value() >= target:
            return
        if self._anim is not None:
            self._anim.stop()
        anim = QPropertyAnimation(self.bar, b"value", self.bar)
        anim.setDuration(duration)
        anim.setStartValue(self.bar.value())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._anim = anim


def flash_style(
    widget: QWidget,
    css_template: str,
    from_color: QColor,
    to_color: QColor,
    duration: int = 420,
    keep: Optional[list] = None,
) -> None:
    """Briefly interpolate one color inside an inline stylesheet, then clear.

    `css_template` must contain `{color}` (e.g. "background: {color}; …").
    Used for the drop-area success flash where QSS alone can't animate.
    """

    anim = QVariantAnimation(widget)
    anim.setDuration(duration)
    anim.setStartValue(from_color)
    anim.setKeyValueAt(0.25, to_color)
    anim.setEndValue(from_color)

    def apply(color):
        c = QColor(color)
        widget.setStyleSheet(css_template.format(color=c.name(QColor.HexArgb)))

    anim.valueChanged.connect(apply)
    if keep is not None:
        keep.append(anim)

    def finish():
        widget.setStyleSheet("")
        if keep is not None and anim in keep:
            keep.remove(anim)

    anim.finished.connect(finish)
    anim.start()


class _Node:
    __slots__ = ("x", "y", "vx", "vy", "r")

    def __init__(self, rng: random.Random, w: int, h: int):
        self.x = rng.uniform(0, w)
        self.y = rng.uniform(0, h)
        speed = rng.uniform(6.0, 18.0)  # px/second — a calm drift
        angle = rng.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.r = rng.uniform(1.6, 3.4)


class ParticleCanvas(QWidget):
    """The "memory constellation": drifting nodes that link up when close.

    Pure-paint decoration for the welcome page. ~30fps while visible, fully
    stopped when hidden. Colors are set per theme via set_palette().
    """

    def __init__(self, parent: Optional[QWidget] = None, node_count: int = 26, seed: int = 11):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._rng = random.Random(seed)
        self._node_count = node_count
        self._nodes: List[_Node] = []
        self._link_dist = 150.0
        self._accent = QColor("#1f6f62")
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._step)

    def set_palette(self, accent: str) -> None:
        self._accent = QColor(accent)
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._nodes and self.width() > 0:
            self._seed_nodes()
        self._timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._timer.stop()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._nodes:
            self._seed_nodes()

    def _seed_nodes(self) -> None:
        w, h = max(1, self.width()), max(1, self.height())
        self._nodes = [_Node(self._rng, w, h) for _ in range(self._node_count)]

    def _step(self) -> None:
        dt = self._timer.interval() / 1000.0
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        margin = 24
        for n in self._nodes:
            n.x += n.vx * dt
            n.y += n.vy * dt
            if n.x < -margin:
                n.x = w + margin
            elif n.x > w + margin:
                n.x = -margin
            if n.y < -margin:
                n.y = h + margin
            elif n.y > h + margin:
                n.y = -margin
        self.update()

    def paintEvent(self, event):
        if not self._nodes:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        link = QColor(self._accent)
        for i, a in enumerate(self._nodes):
            for b in self._nodes[i + 1 :]:
                dx, dy = a.x - b.x, a.y - b.y
                dist = math.hypot(dx, dy)
                if dist < self._link_dist:
                    alpha = int(70 * (1.0 - dist / self._link_dist))
                    if alpha <= 2:
                        continue
                    link.setAlpha(alpha)
                    painter.setPen(QPen(link, 1))
                    painter.drawLine(int(a.x), int(a.y), int(b.x), int(b.y))

        node_color = QColor(self._accent)
        painter.setPen(Qt.NoPen)
        for n in self._nodes:
            node_color.setAlpha(110)
            painter.setBrush(node_color)
            painter.drawEllipse(QPoint(int(n.x), int(n.y)), int(n.r), int(n.r))
        painter.end()
