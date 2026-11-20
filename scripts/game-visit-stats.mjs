export function withGameVisitStats(html, prefix) {
  if (html.includes('data-game-stats')) return html;
  return html.replace('</head>', `<link rel="stylesheet" href="${prefix}assets/css/visit-stats.css">\n</head>`)
    .replace('</body>', `<script defer data-game-stats src="${prefix}assets/js/visit-stats.js"></script>\n</body>`);
}
