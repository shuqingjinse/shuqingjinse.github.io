import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {withGameVisitStats} from './game-visit-stats.mjs';
const script=readFileSync(new URL('../assets/js/visit-stats.js',import.meta.url),'utf8');
function fixture(hostname='bieji-ai.github.io') {
 const boxes=Object.fromEntries(['page_pv','site_pv','site_uv'].map(k=>[k,[{textContent:'—'},{textContent:'—'}]]));
 const status={textContent:''},requests=[],timers=[];
 const document={readyState:'complete',currentScript:null,querySelector:()=>null,querySelectorAll:selector=>{
  if(selector==='[data-visit-status]')return [status];
  const key=selector.match(/data-visit-stat="([^"]+)"/);return key?boxes[key[1]]:[];
 },createElement:()=>({remove(){this.removed=true}}),head:{append:s=>requests.push(s)}};
 const window={};const context=vm.createContext({document,window,location:{protocol:'https:',hostname},setTimeout:cb=>(timers.push(cb),timers.length),clearTimeout:()=>{},Math,Number});
 const run=()=>vm.runInContext(script,context);run();
 const respond=data=>window[new URL(requests[0].src).searchParams.get('jsonpCallback')](data);
 return {boxes,status,requests,timers,respond,run};
}
test('one remote request updates repeated displays, including real zero; repeated setup does not double count',()=>{
 const f=fixture();f.run();assert.equal(f.requests.length,1);
 f.respond({page_pv:1234,site_pv:0,site_uv:45});
 assert.deepEqual(f.boxes.page_pv.map(x=>x.textContent),['1,234','1,234']);
 assert.equal(f.boxes.site_pv[0].textContent,'0');assert.equal(f.status.textContent,'');assert.ok(f.requests[0].removed);
});
test('failed, malformed and late responses leave unavailable counts rather than invented values',()=>{
 for(const fail of [f=>f.requests[0].onerror(),f=>f.respond({page_pv:'<img>',site_pv:2,site_uv:3}),f=>f.timers[0]()]){
  const f=fixture();fail(f);assert.equal(f.status.textContent,'统计暂不可用');assert.equal(f.boxes.page_pv[0].textContent,'—');
  f.respond({page_pv:10,site_pv:20,site_uv:30});assert.equal(f.boxes.page_pv[0].textContent,'—');
 }
});
test('local previews never request production counts',()=>{
 for(const host of ['localhost','127.0.0.1','[::1]']){const f=fixture(host);assert.equal(f.requests.length,0);assert.equal(f.status.textContent,'本地预览不计数');}
});
test('game sync preserves integration once at each nesting depth',()=>{
 const source='<!doctype html><html><head></head><body>game</body></html>';
 for(const prefix of ['../../','../../../']){
  const once=withGameVisitStats(source,prefix);assert.equal(withGameVisitStats(once,prefix),once);
  assert.ok(once.includes('href="'+prefix+'assets/css/visit-stats.css"'));
  assert.ok(once.includes('src="'+prefix+'assets/js/visit-stats.js"'));
  assert.equal((once.match(/data-game-stats/g)||[]).length,1);
 }
});
