// Small, dependency-free Chinese/English search shared by the dialog and checks.
export const KINDS={learn:'课程',think:'文章',practice:'面试',play:'游戏'};
export const normalize=value=>String(value??'').normalize('NFKC').toLocaleLowerCase().replace(/[\s_\-–—·]+/g,' ').trim();
const compact=value=>normalize(value).replace(/\s/g,'');
export function tokens(query){return normalize(query).split(/\s+/).filter(Boolean).slice(0,8);}
export function prepareIndex(records){
 return records.filter(r=>r&&KINDS[r.kind]&&typeof r.title==='string'&&typeof r.url==='string'&&r.url.startsWith('/')&&!r.url.startsWith('//')&&!r.url.includes('\\')).map(r=>({...r,titleSearch:compact(r.title),summarySearch:compact((r.description||'')+' '+(r.tags||'')),bodySearch:compact(r.text)}));
}
export function searchIndex(records,query,kind='all'){
 const terms=tokens(query).map(compact);if(!terms.length)return [];
 const phrase=compact(query);
 return records.filter(r=>kind==='all'||r.kind===kind).map(r=>{
  let score=0;
  for(const term of terms){
   if(r.titleSearch.includes(term))score+=35;
   else if(r.summarySearch.includes(term))score+=14;
   else if(r.bodySearch.includes(term))score+=3;
   else return null;
  }
  if(r.titleSearch.includes(phrase))score+=40;
  if(r.titleSearch===phrase)score+=50;
  return {record:r,score};
 }).filter(Boolean).sort((a,b)=>b.score-a.score||a.record.title.length-b.record.title.length).map(r=>r.record);
}
export function excerpt(record,query,length=130){
 const summary=record.description||'',text=String(record.text||summary).replace(/\s+/g,' ').trim();
 const term=tokens(query).find(t=>text.toLocaleLowerCase().includes(t));
 if(!term)return (summary||text).slice(0,length)+((summary||text).length>length?'…':'');
 const i=text.toLocaleLowerCase().indexOf(term),start=Math.max(0,i-36),slice=text.slice(start,start+length);
 return (start?'…':'')+slice+(start+length<text.length?'…':'');
}
