'use strict';
const $ = id => document.getElementById(id);
const icons = {
  temple:'<path d="m3 9 9-6 9 6H3Zm2 2v7m5-7v7m4-7v7m5-7v7M3 21h18M4 18h16"/>',
  robot:'<rect x="4" y="7" width="16" height="13" rx="4"/><path d="M12 3v4M8 12h.01M16 12h.01M9 16h6M1 11v5m22-5v5"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',close:'<path d="m6 6 12 12M6 18 18 6"/>',
  menu:'<path d="M4 6h16M4 12h16M4 18h16"/>',more:'<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
  send:'<path d="m3 3 18 9-18 9 4-9-4-9Zm4 9h14"/>',file:'<path d="M14 3H5v18h14V8l-5-5Zm0 0v5h5M8 12h8M8 16h6"/>'
};
function icon(name){const span=document.createElement('span');span.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true">'+icons[name]+'</svg>';return span;}
function el(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;}
function button(text,cls,action){const b=el('button',cls,text);b.type='button';if(action)b.addEventListener('click',action);return b;}
function iconButton(name,label,action){const b=button('', 'icon-button',action);b.setAttribute('aria-label',label);b.title=label;b.append(icon(name));return b;}
document.querySelectorAll('[data-icon]').forEach(node=>node.append(icon(node.dataset.icon)));
let currentSessionId=null, sessions=[], attachments=[], generating=false, uploading=false, navigating=false, followBottom=true, navigationVersion=0;
const busy=()=>generating||uploading||navigating;
async function api(url,options={}){
  let response;
  try{response=await fetch(url,options);}catch(e){throw new Error('Connexion au serveur local impossible. Vérifiez que l’application est démarrée.');}
  let data;try{data=await response.json();}catch(e){data={};}
  if(!response.ok){const friendly={409:'Une opération est déjà en cours. Attendez sa fin, puis réessayez.',413:'Le fichier dépasse la limite de 20 Mo.',400:'Demande ou PDF invalide.'};throw new Error(friendly[response.status] ? friendly[response.status]+(response.status===400&&data.error?' '+data.error:'') : data.error||'Erreur du serveur ('+response.status+').');}
  return data;
}
const post=(url,data)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function controls(){
  $('questionInput').disabled=busy();$('sendBtn').disabled=busy();$('attachBtn').disabled=generating||navigating;$('pdfFileInput').disabled=generating||navigating;
  $('newChatBtn').disabled=busy();document.querySelectorAll('.conversation button,.suggestion-card,.attachment button,.retry-btn').forEach(b=>b.disabled=busy());
}
function scrollDown(force=false){if(force||followBottom)requestAnimationFrame(()=>$('chatArea').scrollTop=$('chatArea').scrollHeight);}
$('chatArea').addEventListener('scroll',()=>{const a=$('chatArea');followBottom=a.scrollHeight-a.scrollTop-a.clientHeight<90;});
function welcome(){
  $('chatArea').replaceChildren();const box=el('section','welcome-screen');box.id='welcomeScreen';const avatar=el('div','avatar');avatar.append(icon('robot'));box.append(avatar,el('h2','','Bonjour, comment puis-je vous aider ?'),el('p','','Posez une question administrative ou joignez un PDF. Retrouvez une réponse accompagnée de ses sources.'));
  const grid=el('div','suggestions-grid');['Comment demander les APL ?',"Comment refaire ma carte d'identité ?",'c est quoi la cnss','Importez un PDF avec le bouton +'].forEach(q=>grid.append(button(q,'suggestion-card',()=>{if(busy())return;$('questionInput').value=q;sendQuestion();})));box.append(grid);$('chatArea').append(box);
}
function appendMessage(role,text,result){
  $('welcomeScreen')?.remove();if(role==='event'){const event=el('div','event-message',text);$('chatArea').append(event);scrollDown();return event;}
  const row=el('div','message-row '+(role==='user'?'user':'assistant'));const bubble=el('div','bubble');
  if(role!=='user'){const avatar=el('div','avatar assistant');avatar.append(icon('robot'));row.append(avatar);}
  if(result?.source_notice)bubble.append(el('div','answer-source-indicator '+(result.source_used==='internet'?'external':''),result.source_notice));
  else if(result?.source_used&&result.source_used!=='none')bubble.append(el('div','answer-source-indicator', 'Source : '+result.source_used));
  const used=new Set((result?.claims||[]).flatMap(c=>(c.evidence||[]).map(e=>e.source_id)));
  const sources=(result?.sources||[]).filter(s=>used.has(s.source_id));
  const clean=sources.length?text.split(/\n\s*Sources\s*:/)[0]:text;
  bubble.append(el('div','bubble-text',clean));
  if(sources.length){const details=el('details','sources-container');details.append(el('summary','','Sources & Preuves vérifiées ('+sources.length+')'));
    sources.forEach(s=>{const card=el('div','source-card');card.append(el('strong','','['+s.source_id+'] '+(s.title||'Document')));
      const isPdf=(s.chunk_id||'').startsWith('PDF_'),isWeb=(s.chunk_id||'').startsWith('WEB_');card.append(el('div','source-badge '+(isWeb?'web':''),isPdf?'PDF importé':isWeb?(s.official?'Site officiel':'Internet — non officiel'):'Service-Public'));
      const quotes=new Set((result.claims||[]).flatMap(c=>(c.evidence||[]).filter(e=>e.source_id===s.source_id).map(e=>e.quote)));
      quotes.forEach(q=>card.append(el('blockquote','source-quote',q)));
      try{const url=new URL(s.url);if(['http:','https:'].includes(url.protocol)){const a=el('a','source-link',s.url);a.href=url.href;a.target='_blank';a.rel='noopener noreferrer';card.append(a);}}catch(e){}
      details.append(card);
    });bubble.append(details);
  }
  if(result?.timings){const details=el('details','sources-container');details.append(el('summary','','Détails'));
    details.append(el('p','','Source : '+(result.source_used||'none')+' · Modèle : '+(result.model||'—')+' · Ollama : '+(result.ollama_version||'indisponible')));
    const labels={classification:'Classification',pdf_search:'Recherche PDF',base_search:'Recherche base',reranking:'Reclassement',llm_generation:'Génération LLM',review:'Vérification',internet_search:'Recherche Internet',page_download:'Téléchargement des pages',total:'Temps total'};
    Object.entries(result.timings).forEach(([key,value])=>details.append(el('div','',(labels[key]||key)+' : '+Number(value).toFixed(3)+' s')));bubble.append(details);
  }
  if(role==='assistant')bubble.append(button('Copier','copy-btn',async event=>{const copy=event.currentTarget;try{await navigator.clipboard.writeText(text);copy.textContent='Copié';}catch(e){errorBubble('La copie a échoué. Sélectionnez le texte pour le copier.');}}));
  row.append(bubble);$('chatArea').append(row);scrollDown();return row;
}
function errorBubble(message,retry){const row=appendMessage('assistant',message);row.classList.add('error');row.setAttribute('role','alert');if(retry)row.querySelector('.bubble').append(button('Réessayer','retry-btn',()=>{if(busy())return;row.remove();retry();}));return row;}
function typing(){const row=appendMessage('assistant','');const content=row.querySelector('.bubble-text');content.className='typing';content.append(el('span','','Recherche en cours…'));const dots=el('span','dots');dots.setAttribute('aria-hidden','true');for(let i=0;i<3;i++)dots.append(el('i'));content.append(dots);row.querySelector('.copy-btn').remove();return row;}
const narrowScreen=window.matchMedia('(max-width:799px)');
function drawerAccess(){const open=narrowScreen.matches&&$('sidebar').classList.contains('open');$('sidebar').inert=narrowScreen.matches&&!open;document.querySelector('main').inert=open;$('sidebarBackdrop').hidden=!open;}
narrowScreen.addEventListener('change',drawerAccess);drawerAccess();
function drawer(open){$('sidebar').classList.toggle('open',open);$('sidebarBackdrop').hidden=!open;$('openSidebar').setAttribute('aria-expanded',String(open));drawerAccess();if(open)$('closeSidebar').focus();else if(narrowScreen.matches)$('openSidebar').focus();}
$('openSidebar').onclick=()=>drawer(true);$('closeSidebar').onclick=() => drawer(false);$('sidebarBackdrop').onclick=()=>drawer(false);
document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.querySelectorAll('.conversation-menu').forEach(n=>n.remove());if($('sidebar').classList.contains('open'))drawer(false);}
  if(e.key==='Tab'&&$('sidebar').classList.contains('open')){const nodes=[...$('sidebar').querySelectorAll('button:not(:disabled),input:not(:disabled)')];const first=nodes[0],last=nodes.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}
});
function dateGroup(value){const now=new Date(),date=new Date(value);const today=Date.UTC(now.getFullYear(),now.getMonth(),now.getDate()),day=Date.UTC(date.getFullYear(),date.getMonth(),date.getDate());const age=Math.floor((today-day)/86400000);return age<=0?"Aujourd'hui":age===1?'Hier':age<7?'7 derniers jours':'Plus ancien';}
function renderSessions(){const list=$('conversationList');list.replaceChildren();const groups=new Map();const term=$('conversationSearch').value.toLocaleLowerCase('fr');
  sessions.filter(s=>s.title.toLocaleLowerCase('fr').includes(term)).forEach(s=>{const group=dateGroup(s.updated_at);if(!groups.has(group)){const section=el('section');section.append(el('h2','date-group',group));groups.set(group,section);list.append(section);}const row=el('div','conversation'+(s.session_id===currentSessionId?' active':''));const title=button(s.title,'conversation-title',()=>openConversation(s.session_id));title.title=s.title;if(s.session_id===currentSessionId)title.setAttribute('aria-current','page');const more=iconButton('more','Actions pour '+s.title,()=>showMenu(row,s,more));more.classList.add('conversation-more');row.append(title,more);groups.get(group).append(row);});
  if(!list.childElementCount)list.append(el('p','date-group',term?'Aucune conversation trouvée.':'Vos conversations apparaîtront ici.'));controls();
}
function showMenu(row,session,trigger){document.querySelectorAll('.conversation-menu').forEach(n=>n.remove());const menu=el('div','conversation-menu');const rename=button('Renommer','',async()=>{menu.remove();const title=prompt('Titre de la conversation (80 caractères maximum)',session.title);if(title===null)return;try{await post('/api/sessions/'+session.session_id+'/rename',{title});await refreshSessions();}catch(e){errorBubble(e.message);}});
  const remove=button('Supprimer','',async()=>{menu.remove();if(!confirm('Supprimer cette conversation et ses références à '+session.doc_count+' PDF ? Les PDF partagés avec une autre conversation seront conservés.'))return;navigating=true;controls();try{await post('/api/sessions/'+session.session_id+'/delete',{});sessions=await api('/api/sessions');if(currentSessionId===session.session_id){currentSessionId=null;navigating=false;if(sessions.length)await openConversation(sessions[0].session_id);else await newConversation();}renderSessions();}catch(e){errorBubble(e.message);}finally{navigating=false;controls();}});menu.append(rename,remove);row.append(menu);rename.focus();const close=e=>{if(!menu.contains(e.target)&&e.target!==trigger&&!trigger.contains(e.target)){menu.remove();document.removeEventListener('click',close);}};setTimeout(()=>document.addEventListener('click',close),0);
}
async function refreshSessions(){sessions=await api('/api/sessions');renderSessions();}
function updateHash(){history.replaceState(null,'','#c='+currentSessionId);}
async function ensureSession(){if(!currentSessionId){const data=await post('/api/session',{});currentSessionId=data.session_id;updateHash();}return currentSessionId;}
async function newConversation(){if(busy())return;navigationVersion++;currentSessionId=null;attachments=[];history.replaceState(null,'',location.pathname+location.search);$('questionInput').value='';resizeInput();renderAttachments();welcome();renderSessions();drawer(false);$('questionInput').focus();}
async function openConversation(id){if(busy())return;const version=++navigationVersion;navigating=true;controls();try{const data=await api('/api/history/'+encodeURIComponent(id));if(version!==navigationVersion)return;currentSessionId=id;updateHash();attachments=(data.documents||[]).map(d=>({...d,name:d.filename,state:'prêt'}));welcome();followBottom=true;(data.history||[]).forEach(m=>appendMessage(m.role,m.content||'',m.result));renderAttachments();renderSessions();scrollDown(true);drawer(false);}catch(e){errorBubble(e.message,()=>openConversation(id));}finally{navigating=false;controls();}}
$('newChatBtn').onclick=newConversation;$('conversationSearch').oninput=renderSessions;
window.addEventListener('hashchange',()=>{const id=new URLSearchParams(location.hash.slice(1)).get('c');if(busy()){if(currentSessionId)updateHash();return;}if(id&&/^[a-f0-9]{32}$/.test(id))openConversation(id);else newConversation();});
function resizeInput(){const input=$('questionInput');input.style.height='auto';input.style.height=Math.min(input.scrollHeight,144)+'px';}
$('questionInput').addEventListener('input',resizeInput);
$('questionInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();sendQuestion();}});
$('chatForm').addEventListener('submit',e=>{e.preventDefault();sendQuestion();});
async function sendQuestion(question=null,retry=false){if(busy())return;const q=question??$('questionInput').value.trim();if(!q)return;generating=true;controls();if(!retry)appendMessage('user',q);$('questionInput').value='';resizeInput();const indicator=typing();try{const sid=await ensureSession();const data=await streamChat({session_id:sid,question:q},message=>{indicator.querySelector('.typing span').textContent=message;});indicator.remove();appendMessage('assistant',data.answer,data.result);await refreshSessions().catch(e=>errorBubble('Réponse reçue, mais historique non actualisé : '+e.message,refreshSessions));}catch(e){indicator.remove();errorBubble(e.message,()=>sendQuestion(q,true));}finally{generating=false;controls();$('questionInput').focus();}}
function sizeLabel(size){if(!size)return 'Taille inconnue';return size>=1048576?(size/1048576).toFixed(1)+' Mo':Math.ceil(size/1024)+' Ko';}
function renderAttachments(){const area=$('attachments');area.replaceChildren();attachments.forEach(item=>{const chip=el('div','attachment'+(item.state==='erreur'?' error':''));chip.append(icon('file'));const info=el('div','attachment-info');info.append(el('div','attachment-name',item.name),el('div','attachment-state',[sizeLabel(item.size),item.state==='indexation'?'Indexation en cours…':item.state,item.error].filter(Boolean).join(' · ')));chip.append(info);
  if(item.state==='indexation'){const dots=el('span','dots');dots.setAttribute('aria-hidden','true');for(let i=0;i<3;i++)dots.append(el('i'));chip.append(dots);}
  if(item.state==='erreur'&&item.file)chip.append(button('Réessayer','retry-btn',()=>{item.state='en attente';item.error='';processUploads();}));
  chip.append(iconButton('close','Retirer '+item.name,()=>removeAttachment(item)));area.append(chip);});controls();}
async function removeAttachment(item){if(busy())return;try{if(item.doc_id){await post('/api/documents/delete',{session_id:currentSessionId,doc_id:item.doc_id});}attachments=attachments.filter(a=>a!==item);renderAttachments();await refreshSessions();}catch(e){errorBubble(e.message,()=>removeAttachment(item));}}
function addFiles(files){if(generating||navigating){errorBubble('Attendez la fin de la réponse avant de joindre un PDF.');return;}
  for(const file of files){let error='';if(!file.name.toLowerCase().endsWith('.pdf'))error='Seuls les fichiers PDF sont acceptés.';else if(file.size>20*1024*1024)error='Le fichier dépasse la limite de 20 Mo.';attachments.push({name:file.name,size:file.size,file,state:error?'erreur':'en attente',error});}renderAttachments();processUploads();}
async function processUploads(){if(uploading||generating||navigating)return;uploading=true;controls();try{await ensureSession();let item;while((item=attachments.find(a=>a.state==='en attente'))){item.state='indexation';renderAttachments();const form=new FormData();form.append('file',item.file);form.append('session_id',currentSessionId);try{const data=await api('/api/upload',{method:'POST',body:form});item.doc_id=data.doc_id;item.chunks=data.chunks;item.state='prêt';delete item.file;attachments=attachments.filter(a=>a===item||a.doc_id!==item.doc_id);appendMessage('event',data.event.content);}catch(e){item.state='erreur';item.error=e.message;}renderAttachments();}await refreshSessions();}catch(e){errorBubble(e.message,processUploads);}finally{uploading=false;renderAttachments();}}
$('attachBtn').onclick=()=>$('pdfFileInput').click();$('pdfFileInput').onchange=e=>{addFiles([...e.target.files]);e.target.value='';};
[$('chatArea'),$('chatForm')].forEach(zone=>{zone.addEventListener('dragover',e=>{e.preventDefault();if(!generating&&!navigating)zone.classList.add('dragover');});zone.addEventListener('dragleave',e=>{if(!zone.contains(e.relatedTarget))zone.classList.remove('dragover');});zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('dragover');addFiles([...e.dataTransfer.files]);});});
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
async function checkStatus(){try{const d=await api('/api/status');for(const [id,ok,label]of [['ollamaStatus',d.ollama_ready,'Ollama'],['embeddingStatus',d.embedding_available,'E5'],['rerankerStatus',d.reranker_available&&d.retrieval_index_available,'Reranker']]){const n=$(id);n.textContent=(id==='ollamaStatus'?label+' · '+d.model+(d.ollama_version?' · v'+d.ollama_version:''):label)+(ok?' : disponible':' : indisponible');n.className='status-chip '+(ok?'ready':'offline');n.title=id==='ollamaStatus'?(d.model||'Ollama local'):'Présence des fichiers locaux ; chargement vérifié lors de la recherche.';}}catch(e){document.querySelectorAll('.status-chip').forEach(n=>{n.textContent=n.id.replace('Status','')+' : état indisponible';n.classList.add('offline');});}}
async function init(){welcome();try{await refreshSessions();const hashId=new URLSearchParams(location.hash.slice(1)).get('c');const id=/^[a-f0-9]{32}$/.test(hashId||'')?hashId:sessions[0]?.session_id;if(id)await openConversation(id);}catch(e){errorBubble(e.message,init);}checkStatus();}
init();

async function streamChat(data,onProgress){
  const response=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  if(!response.ok){
    if(response.status===404||response.status===405)throw new Error('Le serveur utilise une ancienne version de l’application. Redémarrez Flask, puis actualisez cette page.');
    let payload={};try{payload=await response.json();}catch(e){}
    throw new Error(payload.error||'Le serveur a renvoyé une erreur ('+response.status+'). Réessayez après son redémarrage.');
  }
  if(!(response.headers.get('content-type')||'').includes('application/x-ndjson'))throw new Error('Le serveur n’a pas renvoyé le flux attendu. Redémarrez Flask, puis actualisez cette page.');
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',final=null;
  const consume=line=>{if(!line.trim())return;let event;try{event=JSON.parse(line);}catch(e){throw new Error('La réponse du serveur est illisible. Réessayez ou redémarrez Flask.');}if(event.type==='stage')onProgress(event.message);if(event.type==='error'||event.error)throw new Error(event.error);if(event.type==='final')final=event;};
  try{while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});let cut;while((cut=buffer.indexOf('\n'))>=0){consume(buffer.slice(0,cut));buffer=buffer.slice(cut+1);}if(done)break;}consume(buffer);}
  finally{await reader.cancel();reader.releaseLock();}
  if(!final)throw new Error('La connexion a été interrompue avant la réponse.');return final;
}
