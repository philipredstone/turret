from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal


class ClickableLabel(QLabel):
    """Custom QLabel that emits signal when clicked"""
    clicked = pyqtSignal(object)
    
    def __init__(self, text=""):
        super().__init__(text)
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        self.clicked.emit(event)
        super().mousePressEvent(event)