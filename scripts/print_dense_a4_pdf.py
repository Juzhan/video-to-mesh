"""Render the calibrated dense A4 layout as a single vector PDF."""

from pathlib import Path
import xml.etree.ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    pdfmetrics.registerFont(
        TTFont("Chinese", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    )
    out = ROOT / "calibration/a4-dense"
    out.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(
        str(out / "reference-board.pdf"),
        pagesize=(297 * mm, 210 * mm),
        pageCompression=1,
    )
    c.setTitle("单页 A4 密集参照板 - 48 个定位码 - 100% 实际大小")
    c.setViewerPreference("PrintScaling", "None")
    c.setFillGray(0)
    for e in ET.parse(ROOT / "calibration/a4-dense/board.svg").getroot():
        if e.tag.endswith("rect") and e.get("fill") == "black":
            x, y, w, h = [float(e.get(k)) for k in ("x", "y", "width", "height")]
            c.rect(x * mm, (210 - y - h) * mm, w * mm, h * mm, fill=1, stroke=0)
    c.setFont("Chinese", 6.5)
    c.drawCentredString(
        148.5 * mm,
        205 * mm,
        "A4 横向 297 × 210 mm | 48 个独立定位码 | 实际大小 100% | 不要适合页面",
    )
    c.drawCentredString(
        148.5 * mm,
        3.5 * mm,
        "遮住的定位码自动跳过；露出的码需分散在两个方向 | 配置：a4-dense/board.json",
    )
    c.setLineWidth(0.2 * mm)
    c.line(98.5 * mm, 175 * mm, 198.5 * mm, 175 * mm)
    for x in (98.5, 198.5):
        c.line(x * mm, 173.7 * mm, x * mm, 176.3 * mm)
    c.drawCentredString(148.5 * mm, 170.5 * mm, "打印后量此横线：100 mm")
    c.line(36 * mm, 55 * mm, 36 * mm, 155 * mm)
    for y in (55, 155):
        c.line(34.7 * mm, y * mm, 37.3 * mm, y * mm)
    c.saveState()
    c.translate(40 * mm, 105 * mm)
    c.rotate(90)
    c.drawCentredString(0, 0, "打印后量此竖线：100 mm")
    c.restoreState()
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
