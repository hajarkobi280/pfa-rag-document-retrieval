const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const code=fs.readFileSync(require('node:path').join(__dirname,'../static/app.js'),'utf8');
function client(response){
  const context={fetch:async()=>response,TextDecoder,Uint8Array};
  vm.createContext(context);
  vm.runInContext(code.slice(code.indexOf('async function streamChat(')),context);
  return context.streamChat;
}
test('HTML 404 explains that Flask needs restarting',async()=>{
  const run=client(new Response('<!doctype html>',{status:404}));
  await assert.rejects(run({},()=>{}),/ancienne version.*Redémarrez Flask/);
});
test('HTML 500 does not expose a JSON parser error',async()=>{
  const run=client(new Response('<!doctype html>',{status:500}));
  await assert.rejects(run({},()=>{}),/erreur \(500\)/);
});
test('successful HTML is rejected before reading as NDJSON',async()=>{
  const run=client(new Response('<!doctype html>',{headers:{'Content-Type':'text/html'}}));
  await assert.rejects(run({},()=>{}),/flux attendu/);
});
test('valid stream preserves progress and final response',async()=>{
  const run=client(new Response('\n{"type":"stage","message":"Recherche..."}\n{"type":"final","answer":"Citation"}\n',{headers:{'Content-Type':'application/x-ndjson'}}));
  const messages=[];
  const result=await run({},message=>messages.push(message));
  assert.deepEqual(messages,['Recherche...']);
  assert.equal(result.answer,'Citation');
});
