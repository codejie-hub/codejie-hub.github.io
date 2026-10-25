/* Shared article navigation and media. No dependencies; content remains readable without JS. */
(() => {
  'use strict';
  document.querySelectorAll('[data-note-browser]').forEach((root) => {
    const cards = [...root.querySelectorAll('[data-note-card]')];
    const domains = [...root.querySelectorAll('[data-note-domain]')];
    const types = [...root.querySelectorAll('[data-note-type]')];
    const heading = root.querySelector('[data-note-heading]');
    const defaultHeading = heading.textContent;
    let domain = 'all';
    let type = 'all';
    const read = () => {
      const hash = location.hash.slice(1);
      const match = hash.match(/^(?:blog|paper)-domain-([^&]+)/);
      domain = match ? match[1] : 'all';
      if (domain === 'world-models') {
        location.replace(`${root.dataset.roadmapUrl}#planning`);
        return false;
      }
      if (!domains.some((b) => b.dataset.noteDomain === domain)) domain = 'all';
      type = new URLSearchParams(hash).get('type') || 'all';
      if (!types.some((b) => b.dataset.noteType === type)) type = 'all';
      return true;
    };
    const render = () => {
      let count = 0;
      cards.forEach((card) => {
        card.hidden = !((domain === 'all' || card.dataset.domain === domain) && (type === 'all' || card.dataset.type === type));
        if (!card.hidden) card.querySelector('[data-note-index]').textContent = String(++count).padStart(2, '0');
      });
      domains.forEach((b) => {
        const value = b.dataset.noteDomain;
        const active = value === domain;
        b.classList.toggle('is-active', active);
        b.setAttribute('aria-pressed', String(active));
        b.querySelector('[data-filter-count]').textContent = cards.filter((c) => (value === 'all' || c.dataset.domain === value) && (type === 'all' || c.dataset.type === type)).length;
      });
      types.forEach((b) => {
        const value = b.dataset.noteType;
        const active = value === type;
        b.classList.toggle('is-active', active);
        b.setAttribute('aria-pressed', String(active));
        b.querySelector('[data-filter-count]').textContent = cards.filter((c) => (value === 'all' || c.dataset.type === value) && (domain === 'all' || c.dataset.domain === domain)).length;
      });
      const label = (buttons, attr, value) => buttons.find((b) => b.dataset[attr] === value).firstChild.textContent.trim();
      heading.textContent = domain === 'all' && type === 'all' ? defaultHeading : [domain === 'all' ? '全部主题' : label(domains, 'noteDomain', domain), type === 'all' ? '全部类型' : label(types, 'noteType', type)].join(' · ');
      root.querySelector('[data-note-count]').textContent = `${count} 篇`;
      root.querySelector('[data-note-empty]').hidden = count !== 0;
    };
    const update = () => {
      const parts = [];
      if (domain !== 'all') parts.push(`${root.dataset.prefix}-domain-${domain}`);
      if (type !== 'all') parts.push(`type=${encodeURIComponent(type)}`);
      const hash = parts.length ? `#${parts.join('&')}` : '';
      if (hash !== location.hash) history.pushState(null, '', `${location.pathname}${location.search}${hash}`);
      render();
    };
    domains.forEach((b) => b.addEventListener('click', () => { domain = b.dataset.noteDomain; update(); }));
    types.forEach((b) => b.addEventListener('click', () => { type = b.dataset.noteType; update(); }));
    const restore = () => { if (read()) render(); };
    addEventListener('popstate', restore);
    addEventListener('hashchange', restore);
    root.querySelector('[data-note-controls]').hidden = false;
    restore();
  });

  document.querySelectorAll('article[data-toc] .content').forEach((content) => {
    let figure = 0;
    let demo = 0;
    content.querySelectorAll('[data-article-figure], [data-mermaid-figure], [data-demo-frame]').forEach((node) => {
      if (node.matches('[data-demo-frame]')) {
        node.querySelector('.demo-number').textContent = `演示 ${++demo} · `;
      } else {
        const number = node.dataset.appendix || String(++figure);
        const target = node.querySelector('.figure-number');
        if (target) target.textContent = `图 ${number} · `;
      }
    });
  });
  document.querySelectorAll('details').forEach((details) => {
    const load = () => {
      if (!details.open) return;
      details.querySelectorAll('template[data-deferred-image]').forEach((template) => {
        template.replaceWith(template.content.cloneNode(true));
      });
    };
    details.addEventListener('toggle', load);
    load();
  });

  // Keep long Mermaid labels at their natural size; only the inner diagram scrolls.
  document.querySelectorAll('[data-mermaid-figure]').forEach((figure) => {
    let queued = false;
    const fit = () => {
      queued = false;
      const svg = figure.querySelector('svg');
      const width = svg?.viewBox?.baseVal?.width;
      if (width) svg.style.setProperty('--mermaid-readable-width', `${Math.max(480, Math.ceil(width))}px`);
    };
    new MutationObserver(() => {
      if (!queued) { queued = true; requestAnimationFrame(fit); }
    }).observe(figure, { childList: true, subtree: true });
    figure.closest('details')?.addEventListener('toggle', fit);
    fit();
  });

  const frames = [...document.querySelectorAll('[data-demo-frame]')];
  const wide = matchMedia('(min-width: 768px)');
  const mount = (frame) => {
    if (!wide.matches || frame.dataset.mounted) return;
    frame.dataset.mounted = 'true';
    const iframe = document.createElement('iframe');
    iframe.title = frame.dataset.title;
    iframe.src = frame.dataset.src;
    iframe.allowFullscreen = true;
    iframe.addEventListener('load', () => { frame.dataset.loaded = 'true'; });
    frame.querySelector('.demo-frame__mount').append(iframe);
    setTimeout(() => { if (!frame.dataset.loaded) frame.dataset.failed = 'true'; }, 15000);
  };
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) mount(entry.target);
    }), { rootMargin: '120px' });
    frames.forEach((frame) => observer.observe(frame));
  } else frames.forEach(mount);
  wide.addEventListener('change', () => frames.forEach((frame) => {
    const r = frame.getBoundingClientRect();
    if (r.top < innerHeight + 120 && r.bottom > -120) mount(frame);
  }));
})();
