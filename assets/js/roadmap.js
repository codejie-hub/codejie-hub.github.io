(() => {
  const roadmap = document.querySelector('[data-roadmap]');
  if (!roadmap) return;

  const tracks = Array.from(roadmap.querySelectorAll('[data-roadmap-track]'));
  const buttons = Array.from(roadmap.querySelectorAll('[data-roadmap-route]'));
  const status = roadmap.querySelector('[data-roadmap-status]');
  const controls = roadmap.querySelector('[data-roadmap-controls]');

  const showRoute = (requestedRoute) => {
    const selected = tracks.find((track) => track.dataset.roadmapTrack === requestedRoute);
    const activeRoute = selected ? requestedRoute : 'all';
    let paperCount = 0;
    let routeCount = 0;

    tracks.forEach((track) => {
      track.hidden = activeRoute !== 'all' && track !== selected;
      if (!track.hidden) {
        routeCount += 1;
        paperCount += track.querySelectorAll('[data-roadmap-paper]').length;
      }
    });
    buttons.forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.roadmapRoute === activeRoute));
    });
    status.textContent = selected
      ? `${selected.dataset.routeLabel} · ${paperCount} 篇论文`
      : `${routeCount} 条路线 · ${paperCount} 篇论文`;
  };

  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      const route = button.dataset.roadmapRoute;
      const fragment = route === 'all' ? '' : `#${route}`;
      if (location.hash !== fragment) {
        history.pushState(null, '', `${location.pathname}${location.search}${fragment}`);
      }
      showRoute(route);
    });
  });

  const restoreRoute = () => showRoute(location.hash.slice(1));
  window.addEventListener('popstate', restoreRoute);
  window.addEventListener('hashchange', restoreRoute);
  restoreRoute();
  controls.hidden = false;
})();
