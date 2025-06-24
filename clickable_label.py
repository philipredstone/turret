from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal


class ClickableLabel(QLabel):
    """
    Custom QLabel that emits a signal when clicked.
    
    This extends the standard QLabel to add click detection functionality,
    which is useful for creating interactive image displays where users
    can click on specific positions.
    """
    
    # Custom signal emitted when the label is clicked
    # The signal carries the mouse event object containing click position
    clicked = pyqtSignal(object)
    
    def __init__(self, text=""):
        """
        Initialize the clickable label.
        
        Args:
            text: Initial text to display (optional)
        """
        super().__init__(text)
        # Enable mouse tracking to detect all mouse movements
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        """
        Override mouse press event to emit clicked signal.
        
        This method is called automatically by Qt when the user clicks
        on the label. It emits our custom clicked signal with the event data.
        
        Args:
            event: QMouseEvent containing click position and button information
        """
        # Emit our custom signal with the event data
        self.clicked.emit(event)
        
        # Call parent implementation to maintain normal QLabel behavior
        super().mousePressEvent(event)