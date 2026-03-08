"""Static security regression tests for frontend templates."""

import re
from pathlib import Path


def test_batch_queue_template_avoids_filename_html_interpolation():
    """Queued file names should not be interpolated into HTML templates."""
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert 'title="${fileName}"' not in html
    assert "${fileName}</p>" not in html
    assert not re.search(r"<p[^>]+title=\"\\$\\{fileName\\}\"", html)
    assert "fileNameElement.textContent = fileName;" in html
