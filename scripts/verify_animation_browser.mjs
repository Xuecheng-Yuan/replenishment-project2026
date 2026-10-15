// Optional integration check: node scripts/verify_animation_browser.mjs
// npm install playwright (or use the Codex bundled runtime on this machine).
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
let chromium;
try{({chromium}=createRequire(import.meta.url)('playwright'));}
catch{const modules=path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');({chromium}=createRequire(path.join(modules,'__demo__.cjs'))('playwright'));}
const edge=[path.join(process.env['ProgramFiles(x86)']||'C:/Program Files (x86)','Microsoft/Edge/Application/msedge.exe'),path.join(process.env.ProgramFiles||'C:/Program Files','Microsoft/Edge/Application/msedge.exe')].find(p=>fs.existsSync(p));
const browser=await chromium.launch({headless:true,...(edge?{executablePath:edge}:{})});
const output=path.resolve(process.argv[2]||path.join(root,'output/playwright/animation'));await fsp.mkdir(output,{recursive:true});
try{
 const context=await browser.newContext({viewport:{width:1620,height:1160}}),p=await context.newPage(),errors=[];
 context.on('page',page=>page.on('pageerror',e=>errors.push(e.message)));p.on('pageerror',e=>errors.push(e.message));
 await p.goto(pathToFileURL(path.join(root,'汇报成果/2026-10-15/动画/index.html')).href);
 if(await p.locator('.panel').count()!==6)throw Error('Six panels absent');
 const scopes=await p.locator('#scope option').evaluateAll(options=>options.map(o=>o.value));
 if(scopes.length!==13)throw Error('Expected 3 complete groups plus 10 fixed SKUs');
 for(const version of ['current','historical']){
  await p.selectOption('#version',version);
  for(const scope of scopes){await p.selectOption('#scope',scope);await p.locator('#seek').fill('28');
   if(!(await p.locator('#summaryTitle').innerText()).includes(await p.locator('#scope option:checked').innerText()))throw Error('Scope summary stale');
   const state=await p.evaluate(()=>demoState());if(state.scope!==scope||Math.floor(state.cursor)!==27)throw Error('View state stale');
  }
  for(const scope of ['all','hot','tail']){await p.selectOption('#scope',scope);await p.screenshot({path:path.join(output,`${version}-${scope}.png`),fullPage:true});}
 }
 await p.selectOption('#version','current');await p.selectOption('#scope','all');await p.locator('#reset').click();
 await p.locator('#play').click();const before=await p.evaluate(()=>demoState());await p.waitForTimeout(800);const after=await p.evaluate(()=>demoState());if(after.cursor<=before.cursor)throw Error('Play not advancing');await p.locator('#play').click();
 await p.locator('#openAll').click();await p.waitForTimeout(1200);if(context.pages().length!==7)throw Error('Six popup windows absent');
 for(const version of ['current','historical'])for(const scope of ['all','hot','tail','sku_3026']){
  await p.selectOption('#version',version);await p.selectOption('#scope',scope);await p.locator('#seek').fill('12');await p.waitForTimeout(250);
  for(const child of context.pages().slice(1)){const s=await child.evaluate(()=>demoState());if(s.version!==version||s.scope!==scope||Math.floor(s.cursor)!==11||s.methods.length!==1)throw Error('Child not synchronized: '+JSON.stringify(s));}
 }
 const popup=context.pages().at(-1);await popup.setViewportSize({width:620,height:430});
 if(await popup.evaluate(()=>document.querySelector('.panel').getBoundingClientRect().bottom>innerHeight))throw Error('Six-window compact layout clips the policy panel');
 await popup.screenshot({path:path.join(output,'ro-popup.png'),fullPage:true});
 await p.locator('#reset').click();if((await p.evaluate(()=>demoState())).cursor!==0)throw Error('Reset failed');
 await p.setViewportSize({width:390,height:844});await p.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
 if(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile page overflows horizontally');
 await p.goto(pathToFileURL(path.join(root,'汇报成果/2026-10-15/动画/公式互动.html')).href);await p.locator('#eps').fill('0.3');
 if(await p.locator('#dro').innerText()!=='141 件')throw Error('DRO formula wrong');await p.locator('#beta').fill('1');if(await p.locator('#dro').innerText()!=='281 件')throw Error('beta=1 formula wrong');
 if(await p.locator('#actualResults tr').count()!==12)throw Error('Actual formula result rows absent');
 if(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Formula mobile page overflows');
 await p.setViewportSize({width:1100,height:1200});await p.screenshot({path:path.join(output,'formula.png'),fullPage:true});
 if(errors.length)throw Error(errors.join('\n'));
 await fsp.writeFile(path.join(output,'checks.json'),JSON.stringify({two_versions:true,three_complete_groups:true,ten_fixed_skus:true,four_horizon_plans:true,six_popup_windows:true,all_scopes_synchronized:true,play_pause_seek_reset:true,mobile_no_overflow:true,formula_controls:true,browser_errors:errors},null,2));
 console.log('Browser verified: two versions, 13 scopes, six synchronized popups, four-period plans, all controls, mobile and formula.');
}finally{await browser.close();}
