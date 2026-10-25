#!/usr/bin/env python3
"""Validate the content contract, all local media, and the reviewed migration.

Run before a Jekyll build. Optional: --site .qa/site checks the rendered HTML too.
"""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
import argparse
import re
import sys
import xml.etree.ElementTree as ET

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
errors = []


def check(ok, message):
    if not ok:
        errors.append(message)


class SiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.figures = self.mermaid = self.demos = self.cards = 0
        self.types = []
        self.ids = []
        self.anchor_depth = 0
        self.nested_links = 0
        self.deferred = 0
        self.post = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if 'data-content-type' in d:
            self.post = dict(kind=d['data-content-type'], hidden=d.get('data-content-hidden') == 'true')
        if 'id' in d:
            self.ids.append(d['id'])
        if tag == 'a':
            if self.anchor_depth:
                self.nested_links += 1
            self.anchor_depth += 1
        if 'data-article-figure' in d:
            self.figures += 1
        if 'data-mermaid-figure' in d:
            self.mermaid += 1
        if 'data-demo-frame' in d:
            self.demos += 1
        if 'data-note-card' in d:
            self.cards += 1
            self.types.append(d.get('data-type'))
        if tag == 'template' and 'data-deferred-image' in d:
            self.deferred += 1
        if tag == 'img' and 'data-managed-image' in d:
            check(bool(d.get('alt')), 'Managed images need descriptive alt text')
            check(bool(d.get('width')) and bool(d.get('height')), 'Managed images need intrinsic dimensions')
            names = [k for k, v in attrs]
            check(len(names) == len(set(names)), 'Managed image has duplicate HTML attributes')

    def handle_endtag(self, tag):
        if tag == 'a':
            self.anchor_depth = max(0, self.anchor_depth - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path)
    parser.add_argument('--baseline', action='store_true', help='Additionally assert the 18-post reviewed release snapshot')
    args = parser.parse_args()
    domains = yaml.safe_load((ROOT / '_data/domains.yml').read_text(encoding='utf-8'))
    kinds = yaml.safe_load((ROOT / '_data/content_types.yml').read_text(encoding='utf-8'))
    manifest = yaml.safe_load((ROOT / '_data/illustrations.yml').read_text(encoding='utf-8'))
    check(bool(domains), 'At least one topic is required')
    if args.baseline:
        check(len(domains) == 7, 'Exactly seven reviewed topics are required')
    check(len({d['slug'] for d in domains}) == len(domains), 'Topic slugs must be unique')
    names = {d['name'] for d in domains}
    topic_counts, type_counts = Counter(), Counter()
    image_count = mermaid_count = demo_count = 0
    posts = list(sorted((ROOT / '_posts').glob('*.md')))
    if args.baseline:
        check(len(posts) == 18, 'The reviewed article set must contain 18 posts')
    for p in posts:
        text = p.read_text(encoding='utf-8-sig')
        front = yaml.safe_load(text.split('---', 2)[1])
        body = text.split('---', 2)[2]
        topics = front.get('categories', [])
        check(len(topics) == 1 and topics[0] in names, f'{p.name}: exactly one recognized topic required')
        topic_counts.update(topics)
        check(front.get('content_type') in kinds, f'{p.name}: recognized content_type required')
        type_counts.update([front.get('content_type')])
        check('paper-notes' not in front.get('tags', []) and 'paper-survey' not in front.get('tags', []), f'{p.name}: duplicate type tags remain')
        cover = front.get('image', {})
        path = cover.get('path', '') if isinstance(cover, dict) else cover
        check('placeholders/' not in path, f'{p.name}: generic placeholder cover remains')
        check((ROOT / path.lstrip('/')).is_file(), f'{p.name}: missing cover {path}')
        check(bool(cover.get('alt')), f'{p.name}: cover alt missing')
        for m in re.finditer(r'{% include figure\.html\s+(.*?)%}', body, re.S):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m[1]))
            image_count += 1
            for key in ['src', 'alt', 'caption', 'width', 'height']:
                check(bool(attrs.get(key)), f'{p.name}: figure needs {key}')
            for key in ['src', 'original']:
                if key in attrs:
                    asset = ROOT / attrs[key].lstrip('/')
                    check(asset.is_file(), f'{p.name}: missing {asset}')
            asset = ROOT / attrs.get('src', '').lstrip('/')
            if asset.is_file() and asset.suffix != '.svg':
                im = Image.open(asset)
                check((str(im.width), str(im.height)) == (attrs.get('width'), attrs.get('height')), f'{p.name}: incorrect figure dimensions {asset.name}')
        mermaid_count += body.count('data-mermaid-figure')
        demo_count += len(re.findall(r'{% include demo-frame\.html', body))
        for m in re.finditer(r'{% include demo-frame\.html\s+(.*?)%}', body, re.S):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m[1]))
            src = attrs.get('src', '')
            check((ROOT / src.lstrip('/')).is_file(), f'{p.name}: demo source missing')
            poster = ROOT / 'assets/images/demos' / (Path(src).stem + '.jpg')
            check(poster.is_file(), f'{p.name}: actual demo screenshot missing: {poster.name}')
        backup = ROOT / '.qa/before' / p.name
        if backup.exists():
            before_body = backup.read_text(encoding='utf-8-sig').split('---', 2)[2]
            after_body = re.sub(r'{% include figure\.html\s+.*?%}', '', body, flags=re.S)
            check(re.sub(r'\s+', ' ', before_body).strip() == re.sub(r'\s+', ' ', after_body).strip(), f'{p.name}: existing user prose changed beyond planned figure insertion')
    if args.baseline:
        check([topic_counts[d['name']] for d in domains] == [5, 1, 2, 2, 3, 4, 1], f'Unexpected topic counts: {topic_counts}')
        check(type_counts == Counter(tutorial=8, paper=6, survey=3, qa=1), f'Unexpected type counts: {type_counts}')
        check(image_count == 48, f'Expected 48 image placements (15 existing, 32 new, 1 redraw), got {image_count}')
        check(mermaid_count == 16, f'Expected all 16 logical Mermaid figures, got {mermaid_count}')
        check(demo_count == 7, f'Expected all seven demos, got {demo_count}')
        check(len(manifest) == 32, 'Exactly 32 new teaching figure placements required')
    for item in manifest:
        asset = ROOT / 'assets/images/figures' / (item['asset'] + '.svg')
        check(asset.is_file(), f'Missing teaching figure: {item["asset"]}')
        post = next(p for p in posts if p.name[11:-3] == item['post'])
        text = post.read_text(encoding='utf-8')
        marker = f'id="fig-{item["asset"]}"'
        check(text.count(marker) == 1, f'{post.name}: figure must appear exactly once: {item["asset"]}')
        expected_heading = item['section'].removeprefix('@before:')
        pos = text.find(marker)
        heading = re.search(r'^#{2,3}\s+' + re.escape(expected_heading) + r'\s+.+$', text, re.M)
        check(heading is not None, f'{post.name}: placement heading missing')
        if heading:
            check(pos < heading.start() if item['section'].startswith('@before:') else pos > heading.start(), f'{post.name}: figure on wrong side of specified section')
    svg_files = list((ROOT / 'assets/images/figures').glob('*.svg')) + list((ROOT / 'assets/images/covers').glob('*.svg'))
    for p in svg_files:
        try:
            svg = ET.parse(p).getroot()
            check(svg.get('viewBox') is not None, f'{p.name}: SVG needs a viewBox')
            check(svg.find('{http://www.w3.org/2000/svg}title') is not None, f'{p.name}: SVG needs a title')
        except ET.ParseError as e:
            check(False, f'{p.name}: invalid SVG: {e}')
    for p in (ROOT / 'assets/images/posts/visual-servoing/previews').glob('*.webp'):
        check(p.stat().st_size < 400*1024, f'{p.name}: preview exceeds 400 KiB')
    if args.site:
        site = args.site.resolve()
        totals = Counter()
        article_types = []
        for path in (site / 'posts').glob('*/index.html'):
            p = SiteParser(); p.feed(path.read_text(encoding='utf-8'))
            totals.update(figures=p.figures, mermaid=p.mermaid, demos=p.demos, deferred=p.deferred)
            check(not p.nested_links, f'{path}: nested image links')
            check(len(p.ids) == len(set(p.ids)), f'{path}: duplicate HTML IDs')
            check(p.post is not None, f'{path}: missing article classification metadata')
            if p.post and not p.post['hidden']:
                article_types.append(p.post['kind'])
        for name, count in [('blog', len(article_types)), ('publications', sum(t in ['paper', 'survey'] for t in article_types))]:
            p = SiteParser(); p.feed((site / name / 'index.html').read_text(encoding='utf-8'))
            check(p.cards == count, f'Rendered {name}: expected {count} cards, got {p.cards}')
            if name == 'publications':
                check(all(t in ['paper', 'survey'] for t in p.types), 'Publication list contains tutorials or quiz')
        if args.baseline:
            check(totals == Counter(figures=48, mermaid=16, demos=7, deferred=2), f'Unexpected rendered media totals: {totals}')
    if errors:
        print('\n'.join('FAIL: ' + error for error in errors))
        raise SystemExit(1)
    print(f'PASS: {len(posts)} posts; topic counts ' + '/'.join(str(topic_counts[d['name']]) for d in domains) + '; types ' + str(dict(type_counts)))
    print(f'PASS: {len(manifest)} teaching diagrams, {image_count} image placements, {mermaid_count} Mermaid figures, {demo_count} demos; sources, captions, dimensions and existing user prose checked.')


if __name__ == '__main__':
    main()
