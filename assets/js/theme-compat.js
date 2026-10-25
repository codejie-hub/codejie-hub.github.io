/* Keep Chirpy 7.3's data-mode API in sync with the site's Bootstrap 5 palette. */
(() => {
  const html = document.documentElement;
  const system = matchMedia('(prefers-color-scheme: dark)');
  const locked = html.getAttribute('data-bs-theme');
  let preference;
  try { preference = localStorage.getItem('cj-theme') || sessionStorage.getItem('mode') || 'system'; }
  catch { preference = 'system'; }
  if (!['light', 'dark', 'system'].includes(preference)) preference = 'system';
  const apply = (next, persist = false) => {
    preference = next;
    const resolved = locked || (next === 'system' ? (system.matches ? 'dark' : 'light') : next);
    html.setAttribute('data-bs-theme', resolved);
    if (locked || next !== 'system') html.setAttribute('data-mode', resolved);
    else html.removeAttribute('data-mode');
    html.toggleAttribute('data-theme-persisted', next !== 'system');
    try {
      if (next === 'system') sessionStorage.removeItem('mode');
      else sessionStorage.setItem('mode', resolved);
      if (persist) localStorage.setItem('cj-theme', next);
    } catch { /* Theme remains usable when storage is unavailable. */ }
    window.postMessage({ id: 'theme-mode' }, location.origin);
  };
  apply(preference);
  system.addEventListener('change', () => apply(preference));
  document.addEventListener('DOMContentLoaded', () => {
    const trigger = document.querySelector('#mode-toggle');
    const menu = document.querySelector('#theme-menu');
    if (!trigger || !menu) return;
    const close = () => { menu.classList.remove('show'); trigger.setAttribute('aria-expanded', 'false'); };
    trigger.addEventListener('click', (event) => {
      event.preventDefault(); event.stopImmediatePropagation();
      const open = menu.classList.toggle('show');
      trigger.setAttribute('aria-expanded', String(open));
    }, true);
    menu.querySelectorAll('button[data-theme-mode]').forEach((button) => {
      button.addEventListener('click', () => {
        apply(button.dataset.themeMode, true); close(); trigger.focus({ preventScroll: true });
      });
    });
    document.addEventListener('click', (event) => { if (!trigger.contains(event.target) && !menu.contains(event.target)) close(); });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && menu.classList.contains('show')) { close(); trigger.focus(); } });
  });
})();
