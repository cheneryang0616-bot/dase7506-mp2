"""Minimal Markdown -> HTML -> A4 PDF for the MP2 report (no external deps).

Handles headings, pipe tables, bullet lists, bold/italic/inline code, links, hr.
Uses Chrome headless; Chrome does not exit on its own, so the output file is
polled and the browser is killed (avoids tool timeouts).
"""
import html
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
ROOT = Path(__file__).resolve().parent.parent


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<![\w*])\*([^*]+)\*(?![\w*])', r'<i>\1</i>', t)
    return t


def md_to_html(md):
    out, i = [], 0
    lines = md.split('\n')
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # table
        if line.lstrip().startswith('|') and i + 1 < n and re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i + 1]):
            header = [c.strip() for c in line.strip().strip('|').split('|')]
            out.append('<table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in header) + '</tr></thead><tbody>')
            i += 2
            while i < n and lines[i].lstrip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                out.append('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in cells) + '</tr>')
                i += 1
            out.append('</tbody></table>')
            continue
        # heading
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            lvl = len(m.group(1))
            out.append(f'<h{lvl}>{inline(m.group(2))}</h{lvl}>')
            i += 1
            continue
        # hr
        if re.match(r'^\s*---+\s*$', line):
            out.append('<hr/>')
            i += 1
            continue
        # bullet list (continuation lines are indented by 2 spaces)
        if re.match(r'^\s*[*-]\s+', line):
            items = []
            while i < n and (re.match(r'^\s*[*-]\s+', lines[i]) or (items and lines[i].strip() and not re.match(r'^\s*(#|\||$)|\s*---+\s*$', lines[i]) and lines[i].startswith('  '))):
                if re.match(r'^\s*[*-]\s+', lines[i]):
                    items.append(re.sub(r'^\s*[*-]\s+', '', lines[i]))
                else:
                    items[-1] += ' ' + lines[i].strip()
                i += 1
            out.append('<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in items) + '</ul>')
            continue
        # numbered list
        if re.match(r'^\s*\d+\.\s+', line):
            items = []
            while i < n and re.match(r'^\s*\d+\.\s+', lines[i]):
                items.append(re.sub(r'^\s*\d+\.\s+', '', lines[i]))
                i += 1
            out.append('<ol>' + ''.join(f'<li>{inline(x)}</li>' for x in items) + '</ol>')
            continue
        # paragraph
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(r'^\s*(#|\||[*-]\s|\d+\.\s)', lines[i]) and not re.match(r'^\s*---+\s*$', lines[i]):
            buf.append(lines[i])
            i += 1
        out.append('<p>' + inline(' '.join(x.strip() for x in buf)) + '</p>')
    return '\n'.join(out)


CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color:#111; }
h1 { font-size: 16pt; margin: 0 0 6pt; }
h2 { font-size: 12pt; margin: 12pt 0 4pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 9pt 0 3pt; }
p { margin: 4pt 0; text-align: left; }
p:has(+ table) { break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.8pt; }
th, td { border: 1px solid #999; padding: 2.5pt 4pt; text-align: left; }
th { background: #eee; }
thead { display: table-header-group; }
ul { margin: 4pt 0 4pt 16pt; padding: 0; }
li { margin: 2pt 0; }
code { font-family: Menlo, monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 2pt; overflow-wrap: anywhere; }
hr { border: none; border-top: 1px solid #ccc; margin: 8pt 0; }
h2, h3 { break-after: avoid; }
table, tr { break-inside: avoid; }
"""


def build_pdf(md_path, pdf_path, html_path=None):
    md = Path(md_path).read_text()
    body = md_to_html(md)
    doc = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    html_path = Path(html_path or (Path(md_path).with_suffix('.html')))
    html_path.write_text(doc)
    out = Path(pdf_path)
    if out.exists():
        out.unlink()
    profile = '/tmp/chrome_mp2_report'
    shutil.rmtree(profile, ignore_errors=True)
    cmd = [CHROME, '--headless=new', '--no-sandbox', '--disable-gpu',
           f'--user-data-dir={profile}', '--no-pdf-header-footer',
           '--virtual-time-budget=4000',
           f'--print-to-pdf={out.resolve()}', html_path.resolve().as_uri()]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        time.sleep(0.5)
        if out.exists() and out.stat().st_size > 1000:
            time.sleep(0.8)
            break
    proc.kill()
    shutil.rmtree(profile, ignore_errors=True)
    if not out.exists():
        raise RuntimeError('PDF was not produced')
    return out


if __name__ == '__main__':
    md = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'report/report.md')
    pdf = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / 'report/report.pdf')
    p = build_pdf(md, pdf)
    data = p.read_bytes()
    print('wrote', p, p.stat().st_size, 'bytes; pages =', data.count(b'/Type /Page') - data.count(b'/Type /Pages'))
