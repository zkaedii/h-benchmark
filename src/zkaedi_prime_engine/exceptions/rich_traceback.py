"""Rich Traceback Support for ZKAEDI Exceptions.

Provides beautiful, developer-friendly traceback formatting using the rich library.
"""

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import rich for enhanced tracebacks
try:
    from rich.traceback import install as install_rich
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.debug("rich library not available. Standard tracebacks will be used.")


def install_rich_traceback(show_locals: bool = True, width: Optional[int] = None):
    """Install rich traceback handler for enhanced error display.
    
    Args:
        show_locals: Whether to show local variables in traceback
        width: Optional width for traceback display
    """
    if not RICH_AVAILABLE:
        logger.warning("rich library not available. Install with: pip install rich")
        return False
    
    try:
        install_rich(
            show_locals=show_locals,
            width=width,
            extra_lines=3,
            theme="monokai"  # Beautiful color scheme
        )
        logger.info("✅ Rich traceback installed successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to install rich traceback: {e}")
        return False


def format_exception_rich(exception: Exception) -> str:
    """Format exception with rich formatting.
    
    Args:
        exception: Exception to format
        
    Returns:
        Formatted exception string
    """
    if not RICH_AVAILABLE:
        return str(exception)
    
    try:
        console = Console()
        with console.capture() as capture:
            console.print_exception(show_locals=True)
        return capture.get()
    except Exception as e:
        logger.warning(f"Rich formatting failed: {e}")
        return str(exception)

