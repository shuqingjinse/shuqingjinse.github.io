import {KINDS,prepareIndex,searchIndex,excerpt,tokens} from './search-core.js';
const trigger=document.querySelector('[data-site-search]');
if(trigger){
 let dialog,input,results,status,filters,records,loading,previousFocus,debounce,filter='all',matches=[];
 const element=(tag,className,text)=>{const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;};
 function highlighted(text,query){
  const fragment=document.createDocumentFragment(),terms=tokens(query).sort((a,b)=>b.length-a.length);
  if(!terms.length){fragment.append(document.createTextNode(text));return fragment;}
  const regex=new RegExp(terms.map(t=>t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'gi');
  let last=0;for(const match of text.matchAll(regex)){fragment.append(document.createTextNode(text.slice(last,match.index)));fragment.append(element('mark','',match[0]));last=match.index+match[0].length;}
  fragment.append(document.createTextNode(text.slice(last)));return fragment;
 }
 function create(){
  dialog=element('dialog','site-search-dialog');dialog.id='site-search-dialog';dialog.setAttribute('aria-labelledby','site-search-title');
  dialog.innerHTML=`<div class="site-search-heading"><h2 id="site-search-title">搜索知识库</h2><button type="button" class="site-search-close" aria-label="关闭搜索">关闭 <kbd>Esc</kbd></button></div><form class="site-search-form" role="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></svg><label class="site-search-sr" for="site-search-input">搜索课程、文章、面试题和游戏</label><input id="site-search-input" type="search" placeholder="搜索知识、课程或问题…" autocomplete="off" spellcheck="false" maxlength="160"><button type="button" class="site-search-clear" aria-label="清空搜索" hidden>清空</button></form><div class="site-search-filters" role="group" aria-label="筛选内容类型"></div><p class="site-search-status" role="status" aria-live="polite"></p><div class="site-search-results" id="site-search-results"></div><div class="site-search-footer"><span>↑ ↓ 选择 · Enter 打开</span><span>标题优先，也搜正文</span></div>`;
  document.body.append(dialog);input=dialog.querySelector('input');results=dialog.querySelector('.site-search-results');status=dialog.querySelector('.site-search-status');filters=dialog.querySelector('.site-search-filters');
  for(const [key,label] of [['all','全部'],...Object.entries(KINDS)]){
   const button=element('button','',label);button.type='button';button.dataset.kind=key;button.setAttribute('aria-pressed',String(key==='all'));button.addEventListener('click',()=>{filter=key;render();});filters.append(button);
  }
  dialog.querySelector('.site-search-close').addEventListener('click',()=>dialog.close());
  dialog.querySelector('.site-search-clear').addEventListener('click',()=>{input.value='';render();input.focus();});
  dialog.querySelector('form').addEventListener('submit',e=>e.preventDefault());
  input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.isComposing&&e.keyCode!==229){e.preventDefault();clearTimeout(debounce);render();results.querySelector('a')?.click();}});
  input.addEventListener('input',()=>{clearTimeout(debounce);debounce=setTimeout(render,90);});
  input.addEventListener('compositionend',()=>render());
  dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  dialog.addEventListener('cancel',e=>{e.preventDefault();dialog.close();});
  dialog.addEventListener('close',()=>{clearTimeout(debounce);document.documentElement.classList.remove('site-search-open');trigger.setAttribute('aria-expanded','false');previousFocus?.focus({preventScroll:true});});
  dialog.addEventListener('keydown',e=>{
   if(e.isComposing)return;
   if(e.key==='Escape'){e.preventDefault();e.stopPropagation();dialog.close();return;}
   if(!['ArrowDown','ArrowUp'].includes(e.key))return;
   const links=[...results.querySelectorAll('a')];if(!links.length)return;
   e.preventDefault();const current=links.indexOf(document.activeElement),next=e.key==='ArrowDown'?Math.min(current+1,links.length-1):current<=0?-1:current-1;
   (next<0?input:links[next]).focus({preventScroll:true});if(next>=0)links[next].scrollIntoView({block:'nearest'});
  });
 }
 async function load(){
  if(records)return;if(loading)return loading;
  status.textContent='正在加载搜索内容…';
  loading=fetch(trigger.dataset.indexUrl,{credentials:'same-origin'}).then(response=>{if(!response.ok)throw new Error('index');return response.json();}).then(data=>{records=prepareIndex(data);}).catch(()=>{
   records=undefined;results.replaceChildren();const box=element('div','site-search-empty','搜索暂时没加载成功。');const retry=element('button','site-search-retry','重新加载');retry.type='button';retry.addEventListener('click',async()=>{await load();if(records)render();});box.append(retry);results.append(box);status.textContent='请重试，或从顶部栏目继续浏览。';
  }).finally(()=>{loading=undefined;});
  return loading;
 }
 function render(){
  if(!records)return;
  const query=input.value.trim();dialog.querySelector('.site-search-clear').hidden=!query;results.replaceChildren();
  for(const button of filters.querySelectorAll('button'))button.setAttribute('aria-pressed',String(button.dataset.kind===filter));
  if(!query){
   status.textContent=`可搜索 ${records.length} 条内容 · 课程、文章、面试题与游戏`;
   const box=element('div','site-search-empty'),label=element('p','','从一个问题开始，或者试试这些关键词');box.append(label);
   const suggestions=element('div','site-search-suggestions');for(const term of ['PPO','KV Cache','CS336','智能体']){const b=element('button','',term);b.type='button';b.addEventListener('click',()=>{input.value=term;render();input.focus();});suggestions.append(b);}box.append(suggestions);results.append(box);return;
  }
  matches=searchIndex(records,query,filter);status.textContent=matches.length?`找到 ${matches.length} 条结果${matches.length>30?'，展示前 30 条':''}`:'没有找到匹配内容';
  if(!matches.length){results.append(element('div','site-search-empty','换一个短些的关键词，或切回“全部”再试试。'));return;}
  const list=element('ol','site-search-list');
  for(const record of matches.slice(0,30)){
   const item=element('li'),link=element('a','site-search-result');link.href=record.url;
   const tag=element('span','site-search-tag site-search-tag-'+record.kind,KINDS[record.kind]);
   const title=element('h3');title.append(highlighted(record.title,query));
   const summary=element('p');summary.append(highlighted(excerpt(record,query),query));
   link.append(tag,title,summary);item.append(link);list.append(item);
  }
  results.append(list);results.scrollTop=0;
 }
 async function open(){
  if(!dialog)create();if(dialog.open){input.focus();return;}
  previousFocus=document.activeElement;dialog.showModal();document.documentElement.classList.add('site-search-open');trigger.setAttribute('aria-expanded','true');input.focus();await load();if(records)render();
 }
 trigger.addEventListener('click',open);
 document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&!e.altKey&&e.key.toLowerCase()==='k'){e.preventDefault();open();}},true);
}
