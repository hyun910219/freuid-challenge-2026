"""report.md -> report.pdf (weasyprint). Toolchain dry-run verified 2026-07-10.

Run: python3 make_pdf.py  (system python3: markdown + weasyprint 66.0)
"""
import pathlib

import markdown
from weasyprint import HTML

HERE = pathlib.Path(__file__).resolve().parent
CSS = """body{font-family:sans-serif;font-size:10pt;margin:2cm}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #999;padding:4px 8px;font-size:9pt}
code{background:#f2f2f2;padding:1px 3px;font-size:9pt}h1{font-size:16pt}h2{font-size:13pt}h3{font-size:11pt}"""


def main():
    md = (HERE / "report.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    out = HERE / "report.pdf"
    HTML(string=html).write_pdf(str(out))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
