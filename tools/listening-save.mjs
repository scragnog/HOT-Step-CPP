// listening-save.mjs — shared browser-side save logic for ear-test score
// sheets under _experiments/_LISTENING/.
//
// Both generators that produce these pages (make-scoresheet.mjs for the
// ear-test-scoresheet skill, and any prepare-listening.mts copied into a new
// experiment folder) splice PERSISTENCE_SCRIPT into their generated
// <script>, replacing a hand-copied persistence block. Keeping one copy here
// means a save-behavior fix lands in every future page instead of only the
// experiment it was written for.
//
// Requires these identifiers already in scope where it's inlined: STUDY, LS,
// scores, picks, fileHandle, say(text, isError), el(tag, attrs, ...kids),
// merge(data), snapshot(), render(). All current templates already define
// them under those exact names.
//
// Behavior: a page opened as file:// keeps the old showSaveFilePicker /
// download flow (it has no server to talk to). A page served over http(s) at
// /listening/<folder>/... POSTs scores to /listening/<folder>/scores instead
// — no dialog, no download, scores.json lands next to the page.
export const PERSISTENCE_SCRIPT = `
// ── Persistence: this browser (always) + scores.json, via POST when served over http, via file download when opened as file://. ──
try{merge(JSON.parse(localStorage.getItem(LS)||"{}"));}catch{}
const LISTENING_FOLDER=(()=>{const m=location.pathname.match(/^\\/listening\\/([^/]+)\\//);return m?m[1]:null;})();
const SCORES_POST_URL=(location.protocol.startsWith("http")&&LISTENING_FOLDER)?("/listening/"+LISTENING_FOLDER+"/scores"):null;
if(SCORES_POST_URL){
  fetch("/listening/"+LISTENING_FOLDER+"/scores.json").then(r=>r.ok?r.json():null).then(d=>{if(d){merge(d);render();}}).catch(()=>{});
}
const idb=()=>new Promise((ok,no)=>{const r=indexedDB.open("scoresheet",1);r.onupgradeneeded=()=>r.result.createObjectStore("handles");r.onsuccess=()=>ok(r.result);r.onerror=()=>no(r.error);});
async function idbDo(mode,fn){const db=await idb();return new Promise((ok,no)=>{const tx=db.transaction("handles",mode);const q=fn(tx.objectStore("handles"));tx.oncomplete=()=>ok(q?.result);tx.onerror=()=>no(tx.error);});}
let writing=false, again=false, timer=null;
function persist(){
  try{localStorage.setItem(LS,JSON.stringify(snapshot()));}catch{}
  clearTimeout(timer);timer=setTimeout(writeFile,400);
}
async function writeFile(){
  if(SCORES_POST_URL){
    if(writing){again=true;return;}
    writing=true;
    try{const r=await fetch(SCORES_POST_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(snapshot(),null,1)});if(!r.ok)throw new Error("HTTP "+r.status);say("Saved scores.json at "+new Date().toLocaleTimeString());}
    catch(e){say("Could not save: "+e.message+". Scores are still kept in this browser.",true);}
    finally{writing=false;if(again){again=false;writeFile();}}
    return;
  }
  if(!fileHandle)return;
  if(writing){again=true;return;}
  writing=true;
  try{const w=await fileHandle.createWritable();await w.write(JSON.stringify(snapshot(),null,1));await w.close();say("Saved to "+fileHandle.name+" at "+new Date().toLocaleTimeString());}
  catch(e){say("Could not write "+fileHandle.name+": "+e.message+". Scores are still kept in this browser.",true);}
  finally{writing=false;if(again){again=false;writeFile();}}
}
async function useHandle(h,fromClick){
  const perm=await h.queryPermission({mode:"readwrite"});
  if(perm!=="granted"&&!(fromClick&&await h.requestPermission({mode:"readwrite"})==="granted")){
    fileHandle=null;document.getElementById("connect").textContent="Reconnect "+h.name;say("Click Reconnect to keep saving to "+h.name+".");return;
  }
  fileHandle=h;
  try{const f=await h.getFile();const text=await f.text();if(text.trim())merge(JSON.parse(text));}catch{}
  document.getElementById("connect").textContent="Saving to "+h.name;
  await idbDo("readwrite",s=>s.put(h,STUDY.id));
  render();writeFile();
}
document.getElementById("connect").onclick=async()=>{
  try{
    const stored=await idbDo("readonly",s=>s.get(STUDY.id)).catch(()=>null);
    if(stored&&!fileHandle){await useHandle(stored,true);if(fileHandle)return;}
    const h=await window.showSaveFilePicker({suggestedName:"scores.json",types:[{description:"Scores",accept:{"application/json":[".json"]}}]});
    await useHandle(h,true);
  }catch(e){if(e.name!=="AbortError")say("This browser can't save to a file ("+e.message+"). Use Export scores instead.",true);}
};
document.getElementById("export").onclick=()=>{
  const a=el("a",{href:URL.createObjectURL(new Blob([JSON.stringify(snapshot(),null,1)],{type:"application/json"})),download:"scores.json"});
  document.body.append(a);a.click();a.remove();
};
if(SCORES_POST_URL){
  document.getElementById("connect").hidden=true;
  document.getElementById("export").hidden=true;
  say("Scores save automatically to scores.json in this folder.");
}else if(!window.showSaveFilePicker){document.getElementById("connect").hidden=true;say("This browser can't save to a file: use Export scores when you finish and put scores.json next to this page.");}
else say("Scores are kept in this browser. Click “Save scores to a file…” and save scores.json next to this page so Claude can read them.");
if(!SCORES_POST_URL)idbDo("readonly",s=>s.get(STUDY.id)).then(h=>h&&useHandle(h,false)).catch(()=>{});
`;
