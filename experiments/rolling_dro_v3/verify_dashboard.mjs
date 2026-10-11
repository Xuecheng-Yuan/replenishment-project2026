// Check the delivered HTML's embedded data, offline links and strategy selector.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const folder=path.join(root,'visualization');
const page=fs.readFileSync(path.join(folder,'index.html'),'utf8');
assert(!/__\w+__/.test(page),'Unreplaced template token');
for (const match of page.matchAll(/href="([^"]+)"/g)) {
  const link=match[1];
  if(!link.startsWith('#') && !/^[a-z]+:/i.test(link)) {
    assert(fs.existsSync(path.resolve(folder,link)),`Missing link ${link}`);
  }
}
const elements={
  method:{value:'DRO_cluster',addEventListener(event,fn){this.callback=fn;}},
  periodRows:{innerHTML:''}
};
const context=vm.createContext({document:{getElementById(id){return elements[id];}}});
const script=page.match(/<script>([\s\S]+)<\/script>/)[1];
vm.runInContext(script,context);
const records=vm.runInContext('data',context);
assert.equal(records.length,140);
for(const method of ['DRO_cluster','DRO_global','DRO_category','sS','SAA']) {
  elements.method.value=method;
  elements.method.callback();
  const rendered=elements.periodRows.innerHTML;
  assert.equal((rendered.match(/<tr>/g)||[]).length,28);
  assert(rendered.startsWith('<tr><td>1</td>'));
  assert(rendered.includes('<tr><td>28</td>'));
  const source=records.find(r=>r.method===method && r.period===28);
  assert(rendered.includes(source.avg_inventory_per_sku.toFixed(2)));
  assert(rendered.includes(source.inventory_demand_ratio.toFixed(6)));
  for(const r of records.filter(r=>r.method===method)) {
    assert(Math.abs(r.inventory_demand_ratio-r.ending_inventory_units/r.demand_units)<1e-12);
  }
}
console.log('Dashboard: 140 records, 5 selectors, 28 rows per selector, offline links pass');
