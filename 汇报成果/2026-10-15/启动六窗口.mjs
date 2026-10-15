import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const modules=path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
let chromium;
try{({chromium}=createRequire(import.meta.url)('playwright'));}
catch{try{({chromium}=createRequire(path.join(modules,'__demo__.cjs'))('playwright'));}
catch{throw Error('未找到 Playwright。可直接双击 动画/index.html，在页面打开六个同步窗口；或安装 playwright 后运行此脚本。');}}
const candidates=[path.join(process.env['ProgramFiles(x86)']||'C:/Program Files (x86)','Microsoft/Edge/Application/msedge.exe'),path.join(process.env.ProgramFiles||'C:/Program Files','Microsoft/Edge/Application/msedge.exe')];
const executablePath=candidates.find(p=>fs.existsSync(p));
if(!executablePath)throw Error('未找到 Edge，请双击 动画/index.html。');
const browser=await chromium.launch({headless:false,executablePath,args:['--start-maximized']});
const context=await browser.newContext({viewport:null});
const master=await context.newPage();
await master.goto(pathToFileURL(path.join(root,'动画/index.html')).href);
await master.locator('#openAll').click();
await master.waitForTimeout(900);
if(context.pages().length!==7)throw Error('未成功打开全部六个窗口。');
const screen=await master.evaluate(()=>({width:window.screen.availWidth,height:window.screen.availHeight}));
const windowIds=[];
const layouts=[];
for(const [i,page] of context.pages().slice(1).entries()){
 const session=await context.newCDPSession(page);
 const {windowId}=await session.send('Browser.getWindowForTarget');windowIds.push(windowId);
 const width=Math.max(360,Math.floor(screen.width/3));const height=Math.max(380,Math.floor(screen.height/2));
 await session.send('Browser.setWindowBounds',{windowId,bounds:{windowState:'normal'}});
 await session.send('Browser.setWindowBounds',{windowId,bounds:{left:(i%3)*width,top:Math.floor(i/3)*height,width,height}});
 await page.waitForTimeout(200);
 layouts.push(await page.evaluate(()=>({method:new URL(location.href).searchParams.get('method'),width:innerWidth,height:innerHeight,panelBottom:document.querySelector('.panel').getBoundingClientRect().bottom,scrollWidth:document.documentElement.scrollWidth})));
 await session.detach();
}
if(new Set(windowIds).size!==6)throw Error('六个策略没有分成六个独立原生窗口。');
await fsp.writeFile(path.join(root,'动画/最近启动状态.json'),JSON.stringify({started:new Date().toISOString(),strategyWindows:6,distinctWindowIds:windowIds,screen,layouts,controlPage:'动画/index.html'},null,2));
await context.pages().at(-1).bringToFront();
await new Promise(resolve=>browser.on('disconnected',resolve));
