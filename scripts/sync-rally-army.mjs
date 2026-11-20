import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const input=process.argv[2];if(!input)throw new Error('Usage: node scripts/sync-rally-army.mjs /path/to/散兵成军手机版.html');
const html=await readFile(resolve(input),'utf8');
if(!html.includes('散兵成军')||!html.includes('id="app"')||html.includes('src="/src/main.ts"'))throw new Error('Expected the built standalone 散兵成军 page.');
const target=resolve(dirname(fileURLToPath(import.meta.url)),'../games/rally-army/index.html');
await mkdir(dirname(target),{recursive:true});await writeFile(target,html);console.log('Updated '+target);
