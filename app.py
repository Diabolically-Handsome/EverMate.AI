
# app.py
import os
import sys
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from i18n_qt import APP_TITLE
from runtime_paths import resource_path
from views.wizard import MainWindow

def main():
    app = QApplication(sys.argv)
    # app icon
    icon_path = resource_path("assets", "icons", "app_icon.svg")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    exit_after_ms = os.getenv("EVERMATE_EXIT_AFTER_MS", "").strip()
    if exit_after_ms.isdigit():
        QTimer.singleShot(int(exit_after_ms), win.close)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
