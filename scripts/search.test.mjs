import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareIndex, searchIndex, excerpt} from '../assets/js/search-core.js';

const records=prepareIndex([
 {title:'KV Cache',url:'/kv-cache/',kind:'think',description:'解释注意力缓存',text:'缓存可以复用历史的 key 和 value。'},
 {title:'PPO 与 DPO 的区别',url:'/llm-interview/#q-2',kind:'practice',description:'后训练面试',text:'PPO 需要奖励模型和价值模型。'},
 {title:'CS336 · 从零构建语言模型',url:'/course-notes/cs336/',kind:'learn',text:'本讲讨论 KV Cache、资源核算与训练。'},
 {title:'烽火破城手机版',url:'/games/fire-siege/mobile/',kind:'play',tags:'手机 攻城',text:'将军可以召唤佛朗机炮。'},
]);

test('Chinese titles, body text, mixed case and full-width queries are searchable',()=>{
 assert.equal(searchIndex(records,'烽火破城')[0].kind,'play');
 assert.equal(searchIndex(records,'资源核算')[0].kind,'learn');
 assert.equal(searchIndex(records,'ｋｖ ＣＡＣＨＥ')[0].url,'/kv-cache/');
 assert.equal(searchIndex(records,'CS336 语言模型')[0].kind,'learn');
 assert.equal(searchIndex(records,'手机')[0].kind,'play');
});

test('Title matches rank above mentions in a body; all query terms must match',()=>{
 assert.equal(searchIndex(records,'KV Cache')[0].url,'/kv-cache/');
 assert.equal(searchIndex(records,'PPO DPO')[0].url,'/llm-interview/#q-2');
 assert.deepEqual(searchIndex(records,'PPO 烽火破城'),[]);
 assert.deepEqual(searchIndex(records,'   '),[]);
 assert.deepEqual(searchIndex(records,'不存在的关键词'),[]);
});

test('Category filters preserve matching results and reject other categories',()=>{
 assert.equal(searchIndex(records,'KV Cache','learn').length,1);
 assert.equal(searchIndex(records,'KV Cache','think').length,1);
 assert.deepEqual(searchIndex(records,'KV Cache','play'),[]);
});

test('Only internal URLs are allowed, and optional fields do not become search terms',()=>{
 assert.equal(prepareIndex([
  {title:'bad',kind:'play',url:'javascript:alert(1)'},
  {title:'bad',kind:'play',url:'//example.com'},
  {title:'bad',kind:'play',url:'/\\example.com'},
  {title:'bad',kind:'other',url:'/ok/'},
  null,
 ]).length,0);
 assert.deepEqual(searchIndex(records,'undefined'),[]);
 assert.deepEqual(searchIndex(records,'[.*+?]'),[]);
});

test('Body excerpts include the matching passage and remain concise',()=>{
 const result=excerpt({text:'开头'.repeat(120)+'资源核算'+'结尾'.repeat(120)},'资源核算');
 assert.ok(result.includes('资源核算'));
 assert.ok(result.startsWith('…'));
 assert.ok(result.endsWith('…'));
 assert.ok(result.length<=132);
});
