const $ = (id) => document.getElementById(id);
function escText(v){ return v == null ? '' : String(v); }
function toast(message, error=false){ const el=$('toast'); if(!el)return; el.textContent=message; el.className='toast show'+(error?' error':''); setTimeout(()=>el.className='toast',2600); }
async function api(url, options={}){
  const res=await fetch(url,{cache:'no-store',headers:{'Content-Type':'application/json',...(options.headers||{})},...options});
  const data=await res.json().catch(()=>({}));
  if(!res.ok) throw new Error(data.detail||`HTTP ${res.status}`);
  return data;
}
function fmtTime(v){ if(!v)return '—'; try{return new Date(v).toLocaleString();}catch{return v;} }
function actionLabel(v){return ({keep:'Keep',read_later:'Read Later',archive:'Archive',trash:'Clean / Trash',unsubscribe_trash:'Unsubscribe + Clean'})[v]||v||'—';}
function setText(id,value){const el=$(id);if(el)el.textContent=escText(value);}
function number(v){return Number.isFinite(Number(v))?Number(v):0;}
function poll(fn, delay){let stopped=false;async function tick(){if(stopped)return;try{await fn();}finally{if(!stopped)setTimeout(tick,delay);}}tick();return()=>{stopped=true;};}
async function scanNow(){await api('/api/worker/scan-now',{method:'POST'});toast('Full Gmail intake pass requested');}
async function pauseWorker(){await api('/api/worker/pause',{method:'POST'});toast('Gmail intake paused; classifier/action workers may continue draining their queues.');}
async function resumeWorker(){await api('/api/worker/resume',{method:'POST'});toast('Gmail intake resumed');}

function updateNav(stats){ if($('nav-pending'))$('nav-pending').textContent=stats.pending||0; if($('nav-rules'))$('nav-rules').textContent=stats.rules||0; }
function workerActive(worker){return ['fetching','scanning','rate-limited'].includes(worker.scanner)||['loading-model','fetching-bodies','classifying','rate-limited'].includes(worker.classifier)||Number(worker.executor_active||0)>0||['pacing','backoff'].includes(worker.gmail_api_state);}
function updateDashboard(d){
  const s=d.stats||{}, w=d.worker||{};
  updateNav(s);
  setText('stat-pending',s.pending||0);setText('stat-discovered',s.discovered||0);setText('stat-classifying',s.classifying||0);setText('stat-actionable',s.actionable||0);setText('stat-rules',s.rules||0);
  setText('scanner-status',w.scanner||'—');setText('classifier-status',w.classifier||'—');setText('executor-status',w.executor||'—');setText('executor-active',w.executor_active||0);setText('executor-workers',w.executor_workers||0);
  setText('gmail-api-status',w.gmail_api_state||'ready');setText('gmail-api-budget',w.gmail_api_units_per_minute||'—');
  const quotaDetail=w.gmail_api_detail||'Ready';const quotaWait=Number(w.gmail_api_wait_seconds||0);setText('gmail-api-detail',quotaWait>0?`${quotaDetail} · ${quotaWait.toFixed(1)}s` : quotaDetail);
  setText('classifier-current',w.classifier_current_subject||'—');setText('last-scan',fmtTime(w.last_scan_completed));
  setText('scan-phase',w.scanner||'—');setText('scan-current',w.scanner_current_subject||((w.scanner==='idle')?'Waiting for next pass…':'Working…'));
  setText('scan-page',w.scan_page||0);setText('scan-seen',w.scan_seen||0);setText('scan-new',w.scan_new||0);setText('scan-known',w.scan_known||0);setText('scan-discovered',w.scan_discovered||0);setText('scan-rule-matched',w.scan_rule_matched||0);
  setText('scan-estimate',w.scan_estimate?`~${w.scan_estimate} matching messages`:'total not known yet');
  const age=$('progress-age');if(age){if(w.last_progress_at){const secs=Math.max(0,Math.round((Date.now()-new Date(w.last_progress_at).getTime())/1000));age.textContent=secs<2?'updated just now':`updated ${secs}s ago`;}else age.textContent='waiting for worker';}
  const bar=$('activity-bar');if(bar)bar.classList.toggle('active',workerActive(w));
  const dot=$('live-dot');if(dot)dot.classList.toggle('idle',!workerActive(w));
  const label=$('worker-label');if(label)label.textContent=d.paused?'Intake paused':'Running';
  const lc=w.local_classifier;if(lc){
    setText('lc-device-maturity',`${lc.device||'cpu'} · ${lc.maturity||'cold-start'}`.toUpperCase());
    setText('lc-status',lc.status||'bootstrap');setText('lc-examples',lc.examples||0);
    setText('lc-latency',(lc.latency_ema_ms==null?'—':lc.latency_ema_ms)+' ms');
    const m=lc.metrics||{};
    setText('lc-accuracy',m.high_confidence_accuracy!=null?`${(m.high_confidence_accuracy*100).toFixed(1)}%`:(m.holdout?'0% coverage':'—'));
    setText('lc-accuracy-note',m.holdout?`holdout coverage ${m.high_confidence_coverage!=null?(m.high_confidence_coverage*100).toFixed(0)+'%':'—'} · accuracy ${m.accuracy!=null?(m.accuracy*100).toFixed(1)+'%':'—'}`:'holdout evaluation');
    setText('lc-auto-pending',`${lc.auto_actioned||0} / ${lc.pending||0}`);
    setText('lc-corrections',`corrections: ${lc.corrections||0}`);
    setText('lc-retrain',(lc.retrain_new_remaining!=null&&lc.retrain_new_remaining<=0)?'due now':`after ${lc.retrain_new_remaining??'—'} examples`);
    setText('lc-retrain-note',`or every ${lc.retrain_interval_hours||6} h; last: ${lc.trained_at||'—'}`);
  }
}
function startDashboardPolling(){poll(async()=>{try{updateDashboard(await api('/api/stats'));}catch(e){setText('scan-current',`Status temporarily unavailable: ${e.message}`);}},1200);}

let currentPending=null;
let pendingAnswerInFlight=false;
const pendingScopeSelections=new Map();
function metric(name,value){const d=document.createElement('div');d.className='metric';const s=document.createElement('span');s.textContent=name;const b=document.createElement('strong');b.textContent=value;d.append(s,b);return d;}
function signalBadge(text){const s=document.createElement('span');s.className='badge';s.textContent=text;return s;}
function scopeLabel(scope){return ({message:'This message only',list:'This mailing list',sender:'This sender',domain:'This sender domain'})[scope]||scope;}
function defaultScope(scopes){return scopes.includes('list')?'list':(scopes.includes('sender')?'sender':'message');}
function updateScopeHelp(scope){const el=$('scope-help');if(!el)return;el.textContent=({message:'Apply only to this one email. No learned rule is created.',list:'Remember the choice for this mailing list. Usually safest for newsletters and alerts.',sender:'Remember the choice for this exact sender address.',domain:'Remember the choice for this sender domain. Use carefully because it can affect different types of mail from the same organisation.'})[scope]||'';}
function renderPending(data){
  const items=data.items||[];setText('pending-count',data.count||0);
  if(items.length===0){$('pending-card').classList.add('hidden');$('pending-empty').classList.remove('hidden');$('upcoming-list').textContent='';const p=document.createElement('p');p.className='empty';p.textContent='No other pending messages.';$('upcoming-list').appendChild(p);currentPending=null;return;}
  $('pending-empty').classList.add('hidden');$('pending-card').classList.remove('hidden');
  const previousId=currentPending&&currentPending.id;
  const i=items[0];
  const sameItem=previousId===i.id;
  currentPending=i;
  setText('pending-category',i.category||'uncertain');setText('pending-position',`Oldest pending · #${i.id}`);setText('pending-subject',i.subject||'(no subject)');setText('pending-sender',i.sender||i.sender_address);setText('pending-snippet',i.snippet||'No preview available.');setText('pending-question',i.question||'What should OpenMailSweep do?');setText('pending-reason',`Why it stopped: ${i.decision_reason||'uncertain'}`);$('gmail-link').href=i.gmail_url;
  const sug=$('pending-suggestion');if(sug){if(i.proposed_action&&i.confidence!=null){sug.textContent=`Suggested action: ${actionLabel(i.proposed_action)} · local classifier confidence ${(Number(i.confidence)*100).toFixed(0)}% · ${i.decision_reason||''}`;}else{sug.textContent=i.decision_reason?`Local classifier reason: ${i.decision_reason}`:'';}}
  const sig=$('pending-signals');sig.textContent='';(i.safety_signals||[]).forEach(x=>sig.appendChild(signalBadge(x)));
  const mg=$('model-grid');mg.textContent='';mg.append(metric('Category',i.category||'—'),metric('Confidence',i.confidence==null?'—':Number(i.confidence).toFixed(3)),metric('Spam',i.spam_probability==null?'—':Number(i.spam_probability).toFixed(3)),metric('Bulk',i.bulk_probability==null?'—':Number(i.bulk_probability).toFixed(3)),metric('Useful',i.useful_probability==null?'—':Number(i.useful_probability).toFixed(3)));
  const sel=$('scope-select');
  const scopes=i.scope_options||['message'];
  const scopeKey=scopes.join('|');
  if(!sameItem||sel.dataset.scopeKey!==scopeKey){
    const remembered=pendingScopeSelections.get(String(i.id));
    const selected=remembered&&scopes.includes(remembered)?remembered:defaultScope(scopes);
    sel.textContent='';
    scopes.forEach(scope=>{const o=document.createElement('option');o.value=scope;o.textContent=scopeLabel(scope);o.selected=scope===selected;sel.appendChild(o);});
    sel.dataset.scopeKey=scopeKey;
    sel.value=selected;
    updateScopeHelp(selected);
  }
  const u=$('unsubscribe-choice');u.disabled=false;u.title=i.unsubscribe_auto_eligible?'OpenMailSweep will automatically choose the best safe unsubscribe method':'OpenMailSweep will try any safe standard unsubscribe method it can verify, then clean the message';
  const up=$('upcoming-list');up.textContent='';items.slice(1,9).forEach((x,idx)=>{const row=document.createElement('div');row.className='upcoming-item';const n=document.createElement('div');n.className='num';n.textContent=idx+2;const mid=document.createElement('div');const strong=document.createElement('strong');strong.textContent=x.subject||'(no subject)';const small=document.createElement('small');small.textContent=x.sender||x.sender_address||'';mid.append(strong,small);const badge=signalBadge(x.category||'uncertain');row.append(n,mid,badge);up.appendChild(row);});if(items.length===1){const p=document.createElement('p');p.className='empty';p.textContent='No other pending messages right now.';up.appendChild(p);}
}
async function refreshPending(){if(pendingAnswerInFlight)return;try{const d=await api('/api/pending?limit=12');renderPending(d);updateNav({pending:d.count});}catch(e){toast(e.message,true);}}
async function answerCurrent(action){if(!currentPending||pendingAnswerInFlight)return;const answeredId=currentPending.id;const scope=$('scope-select').value;if(action==='unsubscribe_trash'&&scope==='domain'){toast('Use mailing-list, sender, or message scope for unsubscribe.',true);return;}pendingAnswerInFlight=true;try{const d=await api(`/api/pending/${answeredId}/answer`,{method:'POST',body:JSON.stringify({action,scope_type:scope})});toast(`Saved. ${d.released} message${d.released===1?'':'s'} released to the action queue.`);pendingScopeSelections.delete(String(answeredId));currentPending=null;await refreshPending();}catch(e){toast(e.message,true);}finally{pendingAnswerInFlight=false;await refreshPending();}}
function startPendingScreen(){
  document.querySelectorAll('.choice[data-action]').forEach(b=>b.addEventListener('click',()=>answerCurrent(b.dataset.action)));
  const sel=$('scope-select');
  if(sel)sel.addEventListener('change',()=>{if(!currentPending)return;pendingScopeSelections.set(String(currentPending.id),sel.value);updateScopeHelp(sel.value);});
  poll(refreshPending,1800);
}

function queueSection(title,items,kind){const section=document.createElement('section');section.className='queue-section';const h=document.createElement('h2');h.textContent=`${title} (${items.length})`;section.appendChild(h);const list=document.createElement('div');list.className='queue-list';if(!items.length){const p=document.createElement('p');p.className='empty';p.textContent='None';list.appendChild(p);}items.forEach(i=>{const row=document.createElement('div');row.className='queue-item';const a=document.createElement('div');const st=document.createElement('strong');st.textContent=i.subject||'(no subject)';const sm=document.createElement('small');sm.textContent=i.sender||i.sender_address||'';a.append(st,sm);const b=document.createElement('div');const label=kind==='recent'?(i.final_action||i.status):(i.desired_action||i.status);b.appendChild(signalBadge(actionLabel(label)));const c=document.createElement('div');c.className='right';const top=document.createElement('div');top.textContent=i.decision_source||'';const small=document.createElement('small');small.textContent=kind==='recent'?([i.action_detail,fmtTime(i.completed_at||i.updated_at)].filter(Boolean).join(' · ')):(i.error||i.decision_reason||'');c.append(top,small);row.append(a,b,c);list.appendChild(row);});section.appendChild(list);return section;}
async function refreshQueue(){try{const d=await api('/api/queue?limit=60');const root=$('queue-sections');root.textContent='';root.append(queueSection('Awaiting local classification',d.discovered||[],'discovered'),queueSection('Classifying now',d.classifying||[],'classifying'),queueSection('Processing now',d.processing||[],'processing'),queueSection('Waiting to execute',d.actionable||[],'actionable'),queueSection('Needs attention',d.failed||[],'failed'),queueSection('Recently completed',d.recent||[],'recent'));}catch(e){toast(e.message,true);}}
function startQueueScreen(){poll(refreshQueue,1800);}

async function saveRule(id){const sel=document.querySelector(`.rule-action[data-rule="${id}"]`);if(!sel)return;try{await api(`/api/rules/${id}`,{method:'PUT',body:JSON.stringify({action:sel.value})});toast('Rule updated. Matching pending mail was released immediately.');setTimeout(()=>location.reload(),500);}catch(e){toast(e.message,true);}}
async function deleteRule(id){if(!confirm('Delete this learned rule? Unprocessed messages governed by it will return to Pending. Completed actions will not be reversed.'))return;try{const d=await api(`/api/rules/${id}`,{method:'DELETE'});toast(`Rule deleted. ${d.returned_to_pending} queued message(s) returned to Pending.`);setTimeout(()=>location.reload(),500);}catch(e){toast(e.message,true);}}
