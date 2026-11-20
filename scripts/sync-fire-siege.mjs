import {withGameVisitStats} from './game-visit-stats.mjs';
import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

// Run the game's npm run build first, then pass the generated standalone HTML.
const input=process.argv[2],mobile=process.argv.includes('--mobile');
if(!input)throw new Error('Usage: node scripts/sync-fire-siege.mjs /path/to/game.html [--mobile]');
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const target=resolve(root,mobile?'games/fire-siege/mobile/index.html':'games/fire-siege/index.html');
const source=await readFile(resolve(input),'utf8');
if(!source.includes('烽火破城')||!source.includes('</body>'))throw new Error('Expected the built 烽火破城 standalone game.');
if(mobile&&!source.includes('data-mode="mobile"'))throw new Error('Expected the dedicated mobile game build.');
if(source.includes('id="site-game-integration"'))throw new Error('Use the original game build, not an already integrated page.');
const integration=`
<!-- Site navigation is separate from the standalone game's bundled logic. -->
<style id="site-game-integration">
.site-game-back{font:11px "PingFang SC","Microsoft YaHei",sans-serif;color:#adc0b2;text-decoration:none;white-space:nowrap;padding:9px 0}
.site-game-switch{display:block;margin:15px auto 0;color:#d3d7b2;font-size:12px;text-decoration:underline;text-underline-offset:4px;letter-spacing:.5px;min-height:30px}body>div>.site-game-switch{display:none}
.site-game-back:hover{color:#f2d493}
.site-game-back:focus-visible{outline:2px solid #eacb90;outline-offset:4px;border-radius:3px}
.topbar nav .site-game-fullscreen{font-size:22px;padding:4px;line-height:1}
@media(max-width:800px){.site-game-back{font-size:10px}.topbar nav{gap:10px}}
@media(max-width:570px){.site-game-back{font-size:9px}.topbar nav{gap:8px}.topbar nav .site-game-fullscreen{font-size:20px;padding:2px}}
html[data-mode=mobile] .topbar nav .site-game-back{display:grid;place-items:center;width:30px;min-height:44px;padding:0;font-size:20px;line-height:1}
html[data-mode=mobile] .topbar nav{gap:0}
@media(max-width:350px){html[data-mode=mobile] .topbar nav .site-game-back{width:25px}}
</style>
<script>
(()=>{
  const nav=document.querySelector('.topbar nav');
  if(!nav)return;
  const back=document.createElement('a');
  back.className='site-game-back';back.href='${mobile?'../../':'../'}';back.textContent='${mobile?'←':'← 小游戏'}';back.setAttribute('aria-label','返回小游戏栏目');
  nav.prepend(back);
  if(document.fullscreenEnabled&&!${mobile}){
    const full=document.createElement('button');
    full.className='site-game-fullscreen';full.type='button';full.textContent='⛶';full.title='全屏游戏';full.setAttribute('aria-label','全屏游戏');
    full.onclick=async()=>{
      try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}
      catch{full.title='浏览器未允许全屏，可以继续直接游玩';}
      document.getElementById('battlefield')?.focus({preventScroll:true});
    };
    document.addEventListener('fullscreenchange',()=>{
      const active=Boolean(document.fullscreenElement);full.title=active?'退出全屏':'全屏游戏';full.setAttribute('aria-label',full.title);
    });
    nav.append(full);
  }
  const switchLink=document.createElement('a');switchLink.className='site-game-switch';switchLink.href='${mobile?'../':'mobile/'}';switchLink.textContent='${mobile?'切换电脑版 →':'手机上玩？进入触屏版 →'}';
  document.querySelector('.intro-card')?.append(switchLink);
  const help=document.querySelector('#help-modal');if(help){const link=switchLink.cloneNode(true);help.append(link);}
  // Let Enter activate the website link and full-screen button before game hotkeys.
  for(const type of ['keydown','keyup'])window.addEventListener(type,e=>{
    if(e.target instanceof Element&&e.target.closest('.site-game-back,.site-game-fullscreen,.site-game-switch')&&['Enter','Space'].includes(e.code))e.stopImmediatePropagation();
  },true);
})();
</script>
`;
await mkdir(dirname(target),{recursive:true});
await writeFile(target,withGameVisitStats(source.replace('</body>',integration+'</body>'),mobile?'../../../':'../../'));
if(mobile)for(const asset of ['manifest.webmanifest','siege-icon-192.png','siege-icon-512.png'])await copyFile(resolve(dirname(resolve(input)),asset),resolve(dirname(target),asset));
console.log('Updated '+target+' with site navigation and device entry.');
