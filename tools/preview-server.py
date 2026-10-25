#!/usr/bin/env python3
"""Serve local previews with explicit SVG/JS types, independent of Windows registry."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
import argparse

class PreviewHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.svg': 'image/svg+xml', '.js': 'text/javascript', '.mjs': 'text/javascript',
        '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
    }

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=4174)
    parser.add_argument('--directory',type=Path,default=Path('.qa/site'))
    args=parser.parse_args()
    directory=args.directory.resolve()
    if not directory.is_dir():parser.error('Preview directory does not exist')
    handler=partial(PreviewHandler,directory=str(directory))
    print(f'Serving {directory} at http://127.0.0.1:{args.port}',flush=True)
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
        server.serve_forever()
