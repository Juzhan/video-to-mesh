'use strict';
const $ = id => document.getElementById(id);
let code = localStorage.getItem('packingCode') || '', file = null, active = false, paused = false, xhr = null, forcedResume = null, wakeLock = null, previewURL = null;
let chunkSize = 4 * 1024 * 1024, maxBytes = 2 * 1024 ** 3, workerEnabled = true;
const labels = {uploading:'等待续传',verifying:'检查视频',uploaded:'已接收',queued:'排队中',processing:'正在生成',done:'模型已就绪',review:'模型已保留 · 待核验',failed:'处理失败'};
function message(text, error=false){$('upload-message').textContent=text;$('upload-message').className=error?'error':'success';}
function bytes(n){return (n/1024/1024).toFixed(1)+' MB';}
function key(f){return 'packingUpload:'+JSON.stringify([f.name,f.size,f.lastModified]);}
async function api(path,options={}){
 const response=await fetch(path,{...options,headers:{Authorization:'Bearer '+code,...(options.headers||{})}});
 const data=await response.json(); if(!response.ok){const e=new Error(data.error||'连接失败');e.status=response.status;throw e;} return data;
}
function post(path,data){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},...(data?{body:JSON.stringify(data)}:{})});}
function element(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
async function download(uid,f){try{const r=await fetch('/api/jobs/'+uid+'/file?path='+encodeURIComponent(f.path),{headers:{Authorization:'Bearer '+code}});if(!r.ok)throw Error('下载未完成，请重试');const url=URL.createObjectURL(await r.blob());const a=element('a');a.href=url;a.download=f.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}catch(e){message(e.message,true);}}
function renderJobs(jobs){
 const root=$('jobs');root.replaceChildren();if(!jobs.length){root.append(element('p','还没有上传视频','empty'));return;}
 for(const j of jobs){
  const card=element('article',undefined,'job'),top=element('div',undefined,'job-heading');top.append(element('h3',j.name),element('span',labels[j.state]||j.state,'badge '+j.state));card.append(top);
  let detail=(j.backend==='meshy'?'Meshy 混合':j.backend==='meshy-fast'?'Meshy 快速':'本地重建')+' · '+bytes(j.size);
  if(j.state==='uploading')detail+=' · 已接收 '+Math.round(j.received/j.size*100)+'%';
  if(j.state==='processing'&&j.progress)detail+=' · '+(j.progress.stage||'准备中')+' · '+Math.round(j.progress.elapsed_seconds||0)+' 秒';
  card.append(element('p',detail));if(j.error)card.append(element('p',j.error,'error'));
  if(j.state==='review')card.append(element('p','已导出模型；形状检查未达标，备用结果仍可下载。'));
  const actions=element('div',undefined,'downloads');
  if(j.state==='uploading'){const b=element('button','选择原视频续传','text-button');b.type='button';b.onclick=()=>{if(active)return;forcedResume=j;$('name').value=j.name;$('backend').value=j.backend;$('blender').checked=!!j.blender;$('library').click();};actions.append(b);}
  if(j.state==='failed'&&workerEnabled){const b=element('button','复用缓存重试','text-button');b.type='button';b.onclick=async()=>{try{await post('/api/jobs/'+j.id+'/retry');refresh();}catch(e){message(e.message,true);}};actions.append(b);}
  for(const f of j.files){const b=element('button',f.name,'text-button');b.type='button';b.onclick=()=>download(j.id,f);actions.append(b);}card.append(actions);root.append(card);
 }
}
async function refresh(){if(!code)return;try{const data=await api('/api/jobs');chunkSize=data.chunk_size;maxBytes=data.max_bytes;workerEnabled=data.worker_enabled;renderJobs(data.jobs);$('connection').textContent='电脑已连接';$('pair').hidden=true;$('workspace').hidden=false;$('pair-error').textContent='';if(!workerEnabled){$('cost-note').textContent='当前为接收模式：只保存视频，不调用生成 API。';$('send').textContent='上传到电脑';}return true;}catch(e){$('connection').textContent='连接中断';if(e.status===401){$('pair').hidden=false;$('workspace').hidden=true;$('pair-error').textContent=e.message;}return false;}}
$('pair-form').onsubmit=async event=>{event.preventDefault();code=$('code').value.trim();if(await refresh())localStorage.setItem('packingCode',code);};
$('refresh').onclick=refresh;
function selectFile(f){if(!f)return;if(active){message('请先完成或暂停当前上传',true);return;}file=f;if(!forcedResume)$('name').value=f.name.replace(/\.[^.]+$/,'');$('file-info').hidden=false;$('file-info').textContent=f.name+' · '+bytes(f.size);$('send').disabled=false;$('transfer').hidden=true;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(f);$('video-preview').src=previewURL;$('video-preview').hidden=false;message(forcedResume?'已选择续传文件，点击上传继续':'视频已选择，确认物体名称后上传');}
$('camera').onchange=e=>{forcedResume=null;selectFile(e.target.files[0]);};$('library').onchange=e=>selectFile(e.target.files[0]);
$('backend').onchange=()=>{$('cost-note').textContent=$('backend').value!=='local'?'上传完成后自动生成，Meshy 会使用你的 API 积分。默认不生成 smooth 版本。':'上传完成后自动进行本地重建，不消耗 Meshy 积分。';};
function putChunk(uid,offset,blob,onProgress){return new Promise((resolve,reject)=>{
 xhr=new XMLHttpRequest();xhr.open('PUT','/api/jobs/'+uid);xhr.setRequestHeader('Authorization','Bearer '+code);xhr.setRequestHeader('Upload-Offset',String(offset));xhr.timeout=90000;xhr.upload.onprogress=e=>onProgress(e.loaded);
 xhr.onload=()=>{try{const d=JSON.parse(xhr.responseText);if(xhr.status>=200&&xhr.status<300)resolve(d);else reject(new Error(d.error||'上传失败'));}catch{reject(new Error('电脑响应异常'));}};
 xhr.onerror=()=>reject(new Error('网络中断，重试可续传'));xhr.ontimeout=()=>reject(new Error('传输超时，重试可续传'));xhr.onabort=()=>reject(new Error('上传已暂停，再点上传即可续传'));xhr.send(blob);
 });}
$('pause').onclick=()=>{paused=true;if(xhr)xhr.abort();};
$('upload-form').onsubmit=async event=>{
 event.preventDefault();if(!file||active)return;if(file.size>maxBytes){message('视频过大，建议使用 1080p / 30 fps，单件控制在一分钟以内',true);return;}
 const selected=file;active=true;paused=false;$('send').disabled=true;$('pause').disabled=false;$('camera').disabled=true;$('library').disabled=true;$('transfer').hidden=false;message('');
 let job=null;
 try{
  try{if(navigator.wakeLock)wakeLock=await navigator.wakeLock.request('screen');}catch{}
  let saved=forcedResume?.id||localStorage.getItem(key(selected));
  if(saved){try{job=await api('/api/jobs/'+saved);}catch(e){if(e.status!==404)throw e;}}
  if(job&&job.size!==selected.size)throw Error('视频大小不一致，请选择原来的视频');
  if(!job){job=await post('/api/uploads',{name:$('name').value,filename:selected.name,size:selected.size,backend:$('backend').value,blender:$('blender').checked});localStorage.setItem(key(selected),job.id);}
  if(job.state!=='uploading'){message('这个视频已在电脑队列中，不会重复提交');await refresh();return;}
  let offset=job.received;const startOffset=offset,start=performance.now();
  const show=loaded=>{const total=Math.min(selected.size,offset+loaded),pct=100*total/selected.size,seconds=(performance.now()-start)/1000,speed=(total-startOffset)/Math.max(seconds,.1);$('upload-progress').value=pct;$('percent').textContent=pct.toFixed(0)+'%';$('transfer-title').textContent='正在上传 · '+job.name;$('speed').textContent=bytes(total)+' / '+bytes(selected.size)+(seconds>1?' · '+bytes(speed)+'/s · 约剩 '+Math.max(0,Math.ceil((selected.size-total)/Math.max(speed,1)))+' 秒':'');};
  // Verify a previously received prefix before resuming a user-selected file.
  if(offset>0)await putChunk(job.id,0,selected.slice(0,Math.min(offset,chunkSize)),()=>{});
  while(offset<selected.size){if(paused)throw Error('上传已暂停，再点上传即可续传');show(0);try{const r=await putChunk(job.id,offset,selected.slice(offset,offset+chunkSize),show);offset=r.received;}catch(e){if(paused)throw e;message('连接不稳定，正在查询断点…',true);await new Promise(r=>setTimeout(r,1500));const current=await api('/api/jobs/'+job.id);offset=current.received;throw new Error('已保存 '+Math.round(offset/selected.size*100)+'%，点击上传即可从断点继续');}}
  $('upload-progress').value=100;$('percent').textContent='100%';$('transfer-title').textContent='电脑正在确认视频';$('pause').disabled=true;
  job=await post('/api/jobs/'+job.id+'/complete');localStorage.removeItem(key(selected));forcedResume=null;
  $('transfer-title').textContent='电脑已接收';$('speed').textContent='现在可以继续录制下一件，或关闭这个网页。';message(workerEnabled?'已加入生成队列，电脑将自动处理。':'视频已保存到电脑。');file=null;$('camera').value='';$('library').value='';await refresh();
 }catch(e){message(e.message||'连接失败，视频可续传',true);$('transfer-title').textContent='上传尚未完成';}
 finally{active=false;xhr=null;$('send').disabled=!file;$('camera').disabled=false;$('library').disabled=false;try{await wakeLock?.release();}catch{}wakeLock=null;}
};
const fragment=new URLSearchParams(location.hash.slice(1));if(fragment.get('code')){code=fragment.get('code');localStorage.setItem('packingCode',code);history.replaceState(null,'',location.pathname);}
if(code){$('code').value=code;refresh();}setInterval(refresh,3000);
