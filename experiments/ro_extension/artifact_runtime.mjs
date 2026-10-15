// Discover the installed authoring runtime without copying node_modules.
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

export async function loadArtifactTool(){
 const bundled=path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
 const resolvers=[createRequire(path.join(bundled,'__replenishment_runtime__.cjs')),createRequire(import.meta.url)];
 for(const resolver of resolvers){
  let entry;
  try{entry=resolver.resolve('@oai/artifact-tool');}catch{continue;}
  return import(pathToFileURL(entry).href);
 }
 throw new Error('Excel重建需要Codex内置的@oai/artifact-tool运行库。模型、CSV和动画可仅用requirements_v3.txt中的Python依赖重建；已交付Excel可直接打开。');
}
