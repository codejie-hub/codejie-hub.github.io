#!/usr/bin/env python3
"""Prepare a local Chrome harness, or verify its dumped DOM.

  python tools/check-roadmap.py prepare /tmp/codex-roadmap-site /tmp/roadmap-check
  google-chrome --headless=new --no-sandbox --disable-gpu \
    --allow-file-access-from-files --virtual-time-budget=5000 \
    --dump-dom file:///tmp/roadmap-check/check.html > /tmp/roadmap-check/result.html
  python tools/check-roadmap.py verify /tmp/roadmap-check/result.html

The harness uses the built page and its real local CSS/JS. Remote theme assets
are excluded so that third-party availability cannot affect these checks.
"""

import argparse
import html
from html.parser import HTMLParser
from pathlib import Path
import re


CHECKS = r"""
<script>
window.addEventListener('load', async () => {
  const output = document.createElement('output');
  output.id = 'roadmap-check-result';
  document.body.append(output);
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const tracks = [...document.querySelectorAll('[data-roadmap-track]')];
  const visible = () => tracks.filter(track => !track.hidden);
  const buttons = [...document.querySelectorAll('[data-roadmap-route]')];
  const select = (id) => buttons.find(button => button.dataset.roadmapRoute === id).click();
  const navigate = (direction) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Browser history did not update')), 1500);
    window.addEventListener('popstate', () => { clearTimeout(timer); resolve(); }, { once: true });
    history.go(direction);
  });
  try {
    check(tracks.length === 7, 'Seven research routes must be available');
    check(document.querySelectorAll('[data-roadmap-paper]').length === 24, 'All 24 papers must be available');
    check(!document.querySelector('[data-roadmap-controls]').hidden, 'Filters must be usable after initialization');
    if (location.hash === '#vla') {
      check(visible().length === 1 && visible()[0].dataset.roadmapTrack === 'vla', 'An incoming VLA URL must select the VLA route');
    } else {
      check(visible().length === 7, 'The initial overview must show all routes');
    }
    select('all');
    check(visible().length === 7, 'Reset must restore all routes');
    select('vla');
    check(visible().length === 1 && visible()[0].dataset.roadmapTrack === 'vla', 'Choosing VLA must hide unrelated routes');
    check(visible()[0].querySelectorAll('[data-roadmap-paper]').length === 5, 'VLA must retain its five paper links');
    check(location.hash === '#vla', 'The selected route must be shareable in the URL');
    check(buttons.filter(button => button.getAttribute('aria-pressed') === 'true').length === 1, 'Exactly one filter must be announced as selected');
    check(document.querySelector('[data-roadmap-status]').textContent.includes('5'), 'The live summary must describe the filtered papers');
    await navigate(-1);
    check(visible().length === 7, 'Browser Back must restore the previous overview');
    await navigate(1);
    check(visible().length === 1 && visible()[0].dataset.roadmapTrack === 'vla', 'Browser Forward must restore the selected route');
    history.replaceState(null, '', '#not-a-route');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    check(visible().length === 7, 'Unknown fragments must preserve a readable overview');
    select('all');
    check(matchMedia('(max-width: 480px)').matches === (innerWidth <= 480), 'Responsive checks must use the actual CSS viewport');
    check(document.documentElement.scrollWidth <= innerWidth + 1, 'The page must not overflow horizontally');
    for (const node of document.querySelectorAll('[data-roadmap-paper]')) {
      const bounds = node.getBoundingClientRect();
      check(bounds.left >= -1 && bounds.right <= innerWidth + 1, 'Paper nodes must fit within the viewport');
    }
    const keyboardTarget = buttons.find(button => button.dataset.roadmapRoute === 'vla');
    keyboardTarget.focus({ preventScroll: true });
    check(document.activeElement === keyboardTarget, 'Route selection must be keyboard focusable');
    keyboardTarget.blur();
    output.dataset.status = 'pass';
    output.textContent = 'PASS: filtering, deep links, reset, history, unknown routes, focus and responsive layout';
  } catch (error) {
    output.dataset.status = 'fail';
    output.textContent = 'FAIL: ' + error.message;
  }
  output.style.cssText = 'position:fixed;bottom:0;left:0;z-index:9999;font:12px monospace;padding:4px;background:#fff;color:#111';
});
</script>
"""


def prepare(site, destination, mode):
    site = site.resolve()
    page = site / 'roadmap/index.html'
    assert page.is_file(), 'FAIL: /roadmap/ is not available in the built website'
    document = page.read_text()
    # Run the actual feature without depending on external CDN assets.
    document = re.sub(r'<script\b[^>]*\bsrc=["\'](?:https?:)?//[^>]*>.*?</script>', '', document, flags=re.S)
    document = re.sub(r'<link\b[^>]*\bhref=["\'](?:https?:)?//[^>]*>', '', document)
    document = re.sub(r'((?:src|href)=["\'])(/assets/[^"\']+)', lambda m: m[1] + (site / m[2].lstrip('/')).as_uri(), document)
    document = re.sub(r'<html\b[^>]*>', f'<html lang="zh-CN" data-bs-theme="{mode}">', document, count=1)
    document = document.replace('</body>', CHECKS + '</body>') if '</body>' in document else document + CHECKS
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'check.html'
    target.write_text(document)
    print(target.as_uri())


class Result(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.status = None
        self.message = ''

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get('id') == 'roadmap-check-result':
            self.active = True
            self.status = values.get('data-status')

    def handle_endtag(self, tag):
        if tag == 'output':
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.message += data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('site', type=Path)
    preparation.add_argument('destination', type=Path)
    preparation.add_argument('--mode', choices=['light', 'dark'], default='light')
    verification = commands.add_parser('verify')
    verification.add_argument('document', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.site, args.destination, args.mode)
    else:
        result = Result()
        result.feed(args.document.read_text())
        assert result.status == 'pass', html.unescape(result.message) or 'Browser checks did not finish'
        print(result.message)


if __name__ == '__main__':
    main()
