import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

// Run the game's npm run build first, then pass the generated standalone HTML.
const input=process.argv[2];
if(!input)throw new Error('Usage: node scripts/sync-fire-siege.mjs /path/to/烽火破城.html');
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const target=resolve(root,'games/fire-siege/index.html');
const source=await readFile(resolve(input),'utf8');
if(!source.includes('烽火破城')||!source.includes('</body>'))throw new Error('Expected the built 烽火破城 standalone game.');
if(source.includes('id="site-game-integration"'))throw new Error('Use the original game build, not an already integrated page.');
const integration=`
<!-- Site navigation is separate from the standalone game's bundled logic. -->
<style id="site-game-integration">
.site-game-back{font:11px "PingFang SC","Microsoft YaHei",sans-serif;color:#adc0b2;text-decoration:none;white-space:nowrap;padding:9px 0}
.site-game-back:hover{color:#f2d493}
.site-game-back:focus-visible{outline:2px solid #eacb90;outline-offset:4px;border-radius:3px}
.topbar nav .site-game-fullscreen{font-size:22px;padding:4px;line-height:1}
@media(max-width:800px){.site-game-back{font-size:10px}.topbar nav{gap:10px}}
@media(max-width:570px){.site-game-back{font-size:9px}.topbar nav{gap:8px}.topbar nav .site-game-fullscreen{font-size:20px;padding:2px}}
</style>
<script>
(()=>{
  const nav=document.querySelector('.topbar nav');
  if(!nav)return;
  const back=document.createElement('a');
  back.className='site-game-back';back.href='../';back.textContent='← 小游戏';back.setAttribute('aria-label','返回小游戏栏目');
  nav.prepend(back);
  if(document.fullscreenEnabled){
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
  // Let Enter activate the website link and full-screen button before game hotkeys.
  for(const type of ['keydown','keyup'])window.addEventListener(type,e=>{
    if(e.target instanceof Element&&e.target.closest('.site-game-back,.site-game-fullscreen')&&['Enter','Space'].includes(e.code))e.stopImmediatePropagation();
  },true);
})();
</script>
`;
await mkdir(dirname(target),{recursive:true});
await writeFile(target,source.replace('</body>',integration+'</body>'));
console.log('Updated games/fire-siege/index.html with site navigation and optional full-screen control.');
