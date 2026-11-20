(() => {
  'use strict';
  if (window.__siteVisitStatsStarted) return;
  window.__siteVisitStatsStarted = true;
  const keys = ['page_pv', 'site_pv', 'site_uv'];
  let totals = null;
  let status = '统计加载中';
  function paint() {
    for (const key of keys) {
      for (const node of document.querySelectorAll('[data-visit-stat="' + key + '"]')) {
        node.textContent = totals ? totals[key].toLocaleString('zh-CN') : '—';
      }
    }
    for (const node of document.querySelectorAll('[data-visit-status]')) node.textContent = status;
  }
  // Standalone games create their start screens at runtime; keep counts on those screens.
  function mountGameStats() {
    for (const host of document.querySelectorAll('.intro-card, #overlay .intro-bottom')) {
      if (host.querySelector('.visit-stats')) continue;
      const box = document.createElement('aside');
      box.className = 'visit-stats visit-stats-game';
      box.setAttribute('aria-label', '访问统计');
      box.innerHTML = '<span>本页访问 <strong data-visit-stat="page_pv">—</strong> 次</span><span>全站访客 <strong data-visit-stat="site_uv">—</strong> 人</span><span class="visit-stats-status" data-visit-status role="status"></span>';
      host.append(box);
    }
  }
  function start() {
    const isGame = document.currentScript?.hasAttribute('data-game-stats') ||
      document.querySelector('script[data-game-stats]');
    if (isGame) {
      mountGameStats();
      const observer = new MutationObserver(records => {
        // Ignore our own text updates to avoid a mutation feedback loop.
        if (records.some(record => [...record.addedNodes].some(node => node.nodeType === 1 && !node.closest?.('.visit-stats')))) {
          mountGameStats();
          paint();
        }
      });
      observer.observe(document.body, {childList: true, subtree: true});
    }
    // Local previews must never be mistaken for real production visits.
    if (!['http:', 'https:'].includes(location.protocol) ||
        /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)) {
      status = '本地预览不计数';
      paint();
      return;
    }
    const callback = 'SiteVisit_' + Math.random().toString(36).slice(2);
    const request = document.createElement('script');
    request.async = true;
    request.referrerPolicy = 'no-referrer-when-downgrade';
    let finished = false;
    let timer;
    function finish(data) {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      const valid = data && keys.every(key => Number.isSafeInteger(data[key]) && data[key] >= 0);
      if (valid) totals = data;
      status = valid ? '' : '统计暂不可用';
      paint();
      request.remove();
      // A late JSONP response after a timeout should be harmless.
      window[callback] = () => {};
    }
    window[callback] = finish;
    request.onerror = () => finish(null);
    request.src = 'https://busuanzi.ibruce.info/busuanzi?jsonpCallback=' + callback;
    timer = setTimeout(() => finish(null), 10000);
    paint();
    document.head.append(request);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once:true});
  else start();
})();
