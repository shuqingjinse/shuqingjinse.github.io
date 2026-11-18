(() => {
  const stage = document.querySelector('.game-preview-stage');
  if (!stage) return;
  const mobile = window.matchMedia('(max-width: 760px)');
  let chosen = false;
  function select(mode) {
    stage.dataset.preview = mode;
    stage.querySelectorAll('[data-preview]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.preview === mode));
    });
    stage.querySelectorAll('.game-preview').forEach(panel => {
      panel.hidden = panel.id !== 'game-preview-' + mode;
    });
  }
  stage.querySelectorAll('[data-preview]').forEach(button => {
    button.addEventListener('click', () => { chosen = true; select(button.dataset.preview); });
  });
  select(mobile.matches ? 'mobile' : 'desktop');
  mobile.addEventListener('change', () => { if (!chosen) select(mobile.matches ? 'mobile' : 'desktop'); });
})();
