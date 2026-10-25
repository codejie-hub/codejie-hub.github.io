/** Dependency-free Chrome DevTools capture and regression checks (Node >=22).
 * Assets: node tools/browser-audit.mjs assets http://127.0.0.1:4173
 * Site:   node tools/browser-audit.mjs site http://127.0.0.1:4174
 * Uses a new headless Chrome profile inside .qa; never touches a signed-in profile.
 */
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const root = resolve(import.meta.dirname, '..');
const mode = process.argv[2] || 'site';
const origin = process.argv[3] || 'http://127.0.0.1:4174';
const output = join(root, '.qa', 'browser');
await mkdir(output, { recursive: true });
const chrome = process.env.BLOG_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile = join(root, '.qa', `chrome-${mode}-${Date.now()}`);
await mkdir(profile, { recursive: true });
const processHandle = spawn(chrome, [
  '--headless=new', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--disable-extensions',
  '--enable-unsafe-swiftshader', '--use-angle=swiftshader',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'
], { windowsHide: true, stdio: 'ignore' });

class CDP {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.events = new Map();
    ws.addEventListener('message', ({ data }) => {
      const msg = JSON.parse(data);
      if (msg.id) {
        const pending = this.pending.get(msg.id);
        if (!pending) return;
        clearTimeout(pending.timer); this.pending.delete(msg.id);
        if (msg.error) pending.reject(new Error(JSON.stringify(msg.error)));
        else pending.resolve(msg.result);
      } else for (const callback of this.events.get(msg.method) || []) callback(msg.params);
    });
  }
  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.addEventListener('open', resolve, { once: true });
      socket.addEventListener('error', reject, { once: true });
    });
    return new CDP(socket);
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 45000);
      this.pending.set(id, { resolve, reject, timer });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  on(event, callback) {
    if (!this.events.has(event)) this.events.set(event, []);
    this.events.get(event).push(callback);
  }
  async evaluate(expression) {
    const result = await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
}

let browser, page;
const failures = [], checks = [], consoleErrors = [], requests = [];
const check = (condition, message) => {
  checks.push({ pass: Boolean(condition), message });
  if (!condition) { failures.push(message); console.log(`FAIL: ${message}`); }
};
try {
  let portData;
  for (let i = 0; i < 100; i++) {
    try { portData = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).trim().split('\n'); break; }
    catch { await delay(100); }
  }
  if (!portData) throw new Error('Chrome did not expose a DevTools endpoint');
  const endpoint = `http://127.0.0.1:${portData[0]}`;
  browser = await CDP.connect(`ws://127.0.0.1:${portData[0]}${portData[1]}`);
  const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
  const targets = await (await fetch(`${endpoint}/json/list`)).json();
  page = await CDP.connect(targets.find((t) => t.id === targetId).webSocketDebuggerUrl);
  await page.send('Page.enable');
  await page.send('Runtime.enable');
  await page.send('Network.enable');
  page.on('Runtime.exceptionThrown', (e) => consoleErrors.push({ url: currentURL, text: e.exceptionDetails.text, detail: e.exceptionDetails.exception?.description }));
  page.on('Network.requestWillBeSent', (e) => requests.push({ url: e.request.url, document: currentURL }));
  let currentURL = '';
  const navigate = async (url, width = 1200, height = 900, theme = 'light') => {
    currentURL = url;
    await page.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    await page.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: theme }] });
    await page.send('Page.navigate', { url });
    for (let i = 0; i < 120; i++) {
      try { if (await page.evaluate(`document.readyState === "complete" && location.href === ${JSON.stringify(url)}`)) break; } catch { /* navigation */ }
      await delay(100);
    }
    await page.evaluate(`Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 3000))]).then(() => {document.documentElement.setAttribute('data-bs-theme', '${theme}'); return true;})`);
    await delay(200);
  };
  const screenshot = async (path, jpeg = false) => {
    const { data } = await page.send('Page.captureScreenshot', { format: jpeg ? 'jpeg' : 'png', ...(jpeg ? { quality: 91 } : {}), fromSurface: true, captureBeyondViewport: false });
    await writeFile(path, Buffer.from(data, 'base64'));
  };

  if (mode === 'series') {
    const slugs = ['imitation-learning','grasp-geometry','contact-control','world-models','offline-rl'];
    for (const width of [375,768,1440]) for (const theme of ['light','dark']) for (const slug of slugs) {
      const path = `/posts/embodied-${slug}/`;
      await navigate(origin+path,width,1000,theme);
      await page.evaluate(`Promise.race([window.MathJax?.startup?.promise || Promise.resolve(),new Promise(r=>setTimeout(r,8000))]).then(()=>true)`);
      const result = await page.evaluate(`(() => {
        const content=document.querySelector('article .content');
        const formulas=[...content.querySelectorAll('mjx-container')];
        return {prose:content.innerText.length,sections:content.querySelectorAll('h2').length,math:formulas.length,mathErrors:content.querySelectorAll('mjx-merror').length,
          figures:content.querySelectorAll('[data-article-figure]').length,overflow:document.documentElement.scrollWidth>innerWidth+1,
          seriesLinks:content.querySelectorAll('.embodied-series a').length,
          code:content.querySelectorAll('code.language-python,.language-python code').length,
          raw:content.innerText.includes('{%'),errorText:[...content.querySelectorAll('mjx-merror')].map(n=>n.textContent),badImages:[...document.images].filter(n=>n.complete && !n.naturalWidth).map(n=>n.src)};
      })()`);
      check(result.prose>3000 && result.sections>=10,`${width} ${theme} ${slug}: full tutorial and sections rendered`);
      check(result.math>15 && result.mathErrors===0,`${width} ${theme} ${slug}: equations typeset without errors (${result.math}) ${JSON.stringify(result.errorText)}`);
      check(result.figures===2 && result.badImages.length===0,`${width} ${theme} ${slug}: cover and two figures available`);
      check(result.seriesLinks===4 && !result.raw,`${width} ${theme} ${slug}: series navigation resolves`);
      check(!result.overflow,`${width} ${theme} ${slug}: no page overflow after math layout`);
      for (let index=0;index<2;index++) {
        await page.evaluate(`document.querySelectorAll('[data-article-figure]')[${index}].scrollIntoView({block:'center',behavior:'instant'})`);
        await page.evaluate(`Promise.race([document.querySelectorAll('[data-article-figure] img')[${index}].decode().then(()=>true).catch(()=>false),new Promise(r=>setTimeout(()=>r(false),5000))])`);
        check(await page.evaluate(`(() => {const f=document.querySelectorAll('[data-article-figure]')[${index}];const r=f.getBoundingClientRect();return f.querySelector('img').naturalWidth>0 && r.left>=0 && r.right<=innerWidth+1 && f.querySelector('figcaption').innerText.length>20;})()`),`${width} ${theme} ${slug}: figure ${index+1} loads and fits`);
        if(width===375 && theme==='light') await screenshot(join(output,`embodied-${slug}-figure-${index+1}-mobile.png`));
      }
      // Scroll each display equation into view and check its local container and page.
      check(await page.evaluate(`(() => {for(const n of document.querySelectorAll('mjx-container[display="true"]')){n.scrollIntoView({block:'center',behavior:'instant'});if(document.documentElement.scrollWidth>innerWidth+1)return false;}return true;})()`),`${width} ${theme} ${slug}: display equations remain within scroll containers`);
      await page.evaluate(`document.querySelector('.embodied-series').open=true;scrollTo({top:0,behavior:'instant'})`);
      check(await page.evaluate(`document.querySelector('.embodied-series').getBoundingClientRect().right<=innerWidth+1`),`${width} ${theme} ${slug}: expanded learning route fits`);
      await screenshot(join(output,`embodied-${slug}-${width}-${theme}.png`));
      console.log(`Series review ${width} ${theme}: ${slug}; math=${result.math}; prose=${result.prose}`);
    }
    await navigate(origin+'/blog/',1440,1000);
    check(await page.evaluate(`document.querySelectorAll('[data-note-card]').length===23`),'Updated blog contains 23 articles');
    await navigate(origin+'/blog/#type=tutorial',375,1000);
    check(await page.evaluate(`document.querySelectorAll('[data-note-card]:not([hidden])').length===13`),'Tutorial filter contains 13 articles');
    await navigate(origin+'/blog/#blog-domain-vla&type=tutorial',375,1000);
    check(await page.evaluate(`document.querySelectorAll('[data-note-card]:not([hidden])').length===2`),'New imitation and world-model tutorials are in the VLA topic');
    await navigate(origin+'/publications/',1440,1000);
    check(await page.evaluate(`document.querySelectorAll('[data-note-card]').length===9`),'Publications remain the 9 papers and surveys');
    await navigate(origin+'/posts/embodied-imitation-learning/',375,1000);
    await page.evaluate(`document.querySelector('.article-figure__link').scrollIntoView({block:'center'});document.querySelector('.article-figure__link').click()`);
    await delay(600);
    check(await page.evaluate(`!!document.querySelector('.glightbox-container')`),'New SVG figure opens in lightbox');
    await page.send('Input.dispatchKeyEvent',{type:'rawKeyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    await page.send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    for(let i=0;i<30;i++){if(await page.evaluate(`!document.querySelector('.glightbox-container')`))break;await delay(50);}
    check(await page.evaluate(`!document.querySelector('.glightbox-container')`),'Escape closes new SVG figure');
    check(consoleErrors.length===0,`Series pages have no JavaScript exceptions: ${consoleErrors.length}`);
  } else if (mode === 'inspect') {
    for(const width of [375,1440]) for(const path of ['/posts/transformer/','/posts/flowmatching/','/posts/robot-kinematics/','/posts/pid-mpc-control/','/posts/diffusion-policy/','/posts/rl-base/','/posts/visual-servoing-survey/','/posts/hilserl/','/posts/pi0.5/']) {
      await navigate(origin+path,width,1000,width===1440?'dark':'light');
      await page.evaluate(`document.querySelectorAll('.content details').forEach(n=>n.open=true)`);
      await delay(300);
      const diagrams=await page.evaluate(`(() => [...document.querySelectorAll('[data-mermaid-figure]')].map((n,i)=>{n.dataset.inspectIndex=i;const s=n.querySelector('svg');const label=s?.querySelector('.nodeLabel');return {i,natural:s?.viewBox.baseVal.width,display:s?.getBoundingClientRect().width,font:label?parseFloat(getComputedStyle(label).fontSize):16,color:label?getComputedStyle(label.querySelector('p')||label).color:null};}))()`);
      for(const d of diagrams) {
        check(Boolean(d.natural) && d.display/d.natural*d.font>=13, `${width} ${path} diagram ${d.i}: effective label size >=13px`);
        check(d.color==='rgb(37, 50, 71)', `${width} ${path} diagram ${d.i}: readable label color on white diagram surface`);
        await page.evaluate(`document.querySelector('[data-inspect-index="${d.i}"]').scrollIntoView({block:'center',behavior:'instant'})`);
        check(await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), `${width} ${path} diagram ${d.i}: scrolling remains inside the figure`);
        if(d.i===0) await screenshot(join(output,`mermaid-${path.split('/').filter(Boolean).at(-1)}-${width}.png`));
      }
      console.log(`Mermaid review ${width}: ${path}; ${JSON.stringify(diagrams)}`);
    }
  } else if (mode === 'assets' || mode === 'series-assets') {
    const covers = join(root, 'assets/images/covers');
    for (const file of (await readdir(covers)).filter((f) => f.endsWith('.svg') && (mode!=='series-assets' || f.startsWith('embodied-')))) {
      await navigate(`${origin}/assets/images/covers/${file}`, 1200, 630);
      const coverState=await page.evaluate(`({url:location.href,root:document.documentElement.tagName,text:document.querySelectorAll('text').length,body:document.documentElement.outerHTML.slice(0,400)})`);
      check(coverState.text>5,`${file}: cover SVG is loaded (${JSON.stringify(coverState)})`);
      await screenshot(join(covers, file.replace('.svg', '.jpg')), true);
      console.log(`Captured cover: ${file}`);
    }
    const figures = join(root, 'assets/images/figures');
    const figureDir = join(output, 'figures'); await mkdir(figureDir, { recursive: true });
    for (const file of (await readdir(figures)).filter((f) => f.endsWith('.svg') && (mode!=='series-assets' || f.startsWith('embodied-')))) {
      const svg = await readFile(join(figures, file), 'utf8');
      const [, w, h] = svg.match(/viewBox="0 0 (\d+) (\d+)"/);
      await navigate(`${origin}/assets/images/figures/${file}`, Number(w), Number(h));
      check(await page.evaluate(`document.querySelectorAll('text').length>5`),`${file}: SVG content is present before capture`);
      const overflow = await page.evaluate(`(() => {const svg=document.documentElement; const b=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].map(n=>({text:n.textContent,r:n.getBoundingClientRect()})).filter(n=>n.r.left < -1 || n.r.right > b.right+1 || n.r.top < -1 || n.r.bottom > b.bottom+1).map(n=>n.text);})()`);
      check(overflow.length === 0, `${file}: text stays within viewBox ${JSON.stringify(overflow)}`);
      await screenshot(join(figureDir, file.replace('.svg', '.png')));
    }
    const demos = join(root, 'assets/demos');
    const posters = join(root, 'assets/images/demos'); await mkdir(posters, { recursive: true });
    for (const file of (await readdir(demos)).filter((f) => mode!=='series-assets' && f.endsWith('.html'))) {
      await navigate(`${origin}/assets/demos/${file}`, 1200, 750);
      await delay(2500);
      const state = await page.evaluate(`({canvas:document.querySelectorAll('canvas').length,text:document.body.innerText.slice(0,200),scroll:document.documentElement.scrollWidth})`);
      check(state.canvas > 0 || file === 'vit-vs-feature-binning.html', `${file}: actual demo scene initialized`);
      await screenshot(join(posters, file.replace('.html', '.jpg')), true);
      console.log(`Captured demo: ${file}; canvases=${state.canvas}`);
    }
  } else {
    const sourcePosts = await Promise.all((await readdir(join(root,'_posts'))).filter(f=>f.endsWith('.md')).map(async f=>{
      const front=(await readFile(join(root,'_posts',f),'utf8')).split('---')[1];
      return {hidden:/^hidden:\s*true/m.test(front),type:front.match(/^content_type:\s*(\w+)/m)?.[1],topic:front.match(/^categories:\s*\[([^\]]+)\]/m)?.[1]};
    }));
    const visiblePosts=sourcePosts.filter(p=>!p.hidden);
    const expectedAll=visiblePosts.length;
    const expectedVla=visiblePosts.filter(p=>p.topic==='机器人学习与 VLA').length;
    const expectedVlaPapers=visiblePosts.filter(p=>p.topic==='机器人学习与 VLA' && p.type==='paper').length;
    const expectedControl=visiblePosts.filter(p=>p.topic==='运动学与控制' && p.type==='tutorial').length;
    const expectedPublications=visiblePosts.filter(p=>['paper','survey'].includes(p.type)).length;
    const state = () => page.evaluate(`(() => {
      const cards=[...document.querySelectorAll('[data-note-card]')];
      return {count:cards.filter(n=>!n.hidden).length, all:cards.length, heading:document.querySelector('[data-note-heading]')?.textContent,
        types:cards.filter(n=>!n.hidden).map(n=>n.dataset.type),
        pressed:[...document.querySelectorAll('[data-note-domain][aria-pressed="true"]')].map(n=>n.dataset.noteDomain),
        hash:location.hash, overflow:document.documentElement.scrollWidth > innerWidth+1};})()`);
    const click = (selector) => page.evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
    await navigate(`${origin}/blog/`, 1440, 1000);
    check((await state()).count === expectedAll, `Blog contains all ${expectedAll} posts`);
    check(await page.evaluate(`getComputedStyle(document.querySelector('#theme-menu')).display === 'none'`), 'Theme menu is initially collapsed');
    await click('#mode-toggle');
    check(await page.evaluate(`document.querySelector('#mode-toggle').getAttribute('aria-expanded') === 'true' && getComputedStyle(document.querySelector('#theme-menu')).display !== 'none'`), 'Theme menu opens accessibly');
    await click('#theme-menu [data-theme-mode="dark"]');
    check(await page.evaluate(`document.documentElement.dataset.bsTheme==='dark' && document.documentElement.dataset.mode==='dark'`), 'Theme selection synchronizes Bootstrap and Chirpy palettes');
    await navigate(`${origin}/blog/`,1440,1000);
    check(await page.evaluate(`document.documentElement.dataset.mode==='dark'`), 'Explicit theme survives page navigation');
    await click('#mode-toggle'); await click('#theme-menu [data-theme-mode="system"]');
    await click('[data-note-domain="vla"]'); check((await state()).count === expectedVla, `VLA topic contains ${expectedVla} posts`);
    await click('[data-note-type="paper"]'); check((await state()).count === expectedVlaPapers, `Topic/type intersection contains ${expectedVlaPapers} VLA papers`);
    await page.evaluate('history.back()'); await delay(200); check((await state()).count === expectedVla, 'Back restores previous type');
    await page.evaluate('history.forward()'); await delay(200); check((await state()).count === expectedVlaPapers, 'Forward restores intersection');
    await navigate(`${origin}/blog/#blog-domain-control&type=tutorial`, 375, 900);
    check((await state()).count === expectedControl, `Incoming combined filter restores ${expectedControl} control tutorials`);
    await navigate(`${origin}/blog/#type=qa`, 375, 900);
    check((await state()).count === 1, 'Incoming type-only link selects the quiz');
    await navigate(`${origin}/blog/#blog-domain-not-real&type=unknown`, 375, 900);
    check((await state()).count === expectedAll, 'Unknown filters preserve a complete readable overview');
    await navigate(`${origin}/publications/`, 1440, 1000);
    let actual = await state(); check(actual.count === expectedPublications && actual.types.every((t) => ['paper','survey'].includes(t)), `Publication pool contains exactly ${expectedPublications} papers/surveys`);
    await screenshot(join(output, 'publications-desktop.png'));
    await navigate(`${origin}/blog/#blog-domain-world-models`, 1440, 1000); await delay(500);
    check(await page.evaluate('location.pathname.endsWith("/roadmap/") && location.hash === "#planning"'), 'Legacy world-models hash points to planning route');
    for (const legacy of ['vla', '控制', '多模态模型']) {
      await navigate(`${origin}/categories/${encodeURIComponent(legacy)}/`, 375, 900);
      check(await page.evaluate('document.querySelectorAll(".topic-index__group li a").length > 0'), `${legacy}: legacy archive retains article links`);
    }
    await navigate(`${origin}/blog/`, 1440, 1000);
    const urls = await page.evaluate(`[...document.querySelectorAll('[data-note-card] h2 a')].map(a=>a.pathname)`);
    for (const width of [375,1440]) for (const path of urls) {
      await navigate(origin + path, width, width === 375 ? 900 : 1000);
      const result = await page.evaluate(`(() => {
        const article=document.querySelector('article[data-toc]');
        const content=article?.querySelector('.content');
        const figures=[...document.querySelectorAll('[data-article-figure]')];
        const imageErrors=[...document.images].filter(n=>n.getAttribute('src') && n.complete && n.naturalWidth===0).map(n=>n.getAttribute('src'));
        const boxes=[...document.querySelectorAll('[data-article-figure],[data-mermaid-figure],[data-demo-frame]')].map(n=>n.getBoundingClientRect());
        return {article:!!article,prose:content?.innerText.length,overflow:document.documentElement.scrollWidth>innerWidth+1,
          boxesFit:boxes.every(b=>b.left>=-1 && b.right<=innerWidth+1),imageErrors,
          figures:figures.length,managed:figures.every(f=>!!f.querySelector('figcaption') && f.querySelectorAll('a a').length===0),
          iframe:document.querySelectorAll('[data-demo-frame] iframe').length,
          originals:performance.getEntriesByType('resource').filter(n=>/ibvs-handwritten-[12]\\.jpg/.test(n.name)).map(n=>n.name),
          mermaid:document.querySelectorAll('[data-mermaid-figure] svg').length,
          rawMermaid:document.querySelectorAll('[data-mermaid-figure] code.language-mermaid').length,
          title:article?.querySelector('h1')?.textContent};})()`);
      check(result.article && result.prose > 100, `${width} ${path}: article content rendered`);
      check(!result.overflow && result.boxesFit, `${width} ${path}: page/media fit viewport`);
      check(result.imageErrors.length === 0, `${width} ${path}: images load ${JSON.stringify(result.imageErrors)}`);
      check(result.managed, `${width} ${path}: figures have captions and no nested links`);
      if (width===375) check(result.iframe===0, `${path}: mobile does not mount demo iframe`);
      check(result.originals.length===0, `${path}: originals not preloaded by collapsed appendix`);
      const media = await page.evaluate(`(() => {
        document.querySelectorAll('.content details').forEach(d=>d.open=true);
        return [...document.querySelectorAll('.content [data-article-figure],.content [data-mermaid-figure]')].map((n,i)=>{n.dataset.auditIndex=i;return {i,mermaid:n.hasAttribute('data-mermaid-figure')};});})()`);
      for (const item of media) {
        await page.evaluate(`document.querySelector('[data-audit-index="${item.i}"]').scrollIntoView({block:'center',behavior:'instant'})`);
        await delay(80);
        const inspectMedia = () => page.evaluate(`(() => {const f=document.querySelector('[data-audit-index="${item.i}"]');const im=f.querySelector('img');const svg=f.querySelector('svg');const r=f.getBoundingClientRect();return {fit:r.left>=-1 && r.right<=innerWidth+1,loaded:im?im.complete&&im.naturalWidth>0:!!svg,src:im?.src,white:im?getComputedStyle(im).backgroundColor:'',height:r.height,mermaid:!!svg};})()`);
        let status = await inspectMedia();
        for(let attempt=0; !status.loaded && attempt<60; attempt++) { await delay(50); status=await inspectMedia(); }
        check(status.fit && status.height>30, `${width} ${path} media ${item.i}: visible media fits the reading column`);
        check(status.loaded, `${width} ${path} media ${item.i}: actual image or Mermaid SVG loaded (${status.src || 'Mermaid'})`);
      }
      await page.evaluate('scrollTo({top:0,behavior:"instant"})');
      await screenshot(join(output, `${path.split('/').filter(Boolean).at(-1)}-${width}-light.png`));
      console.log(`Reviewed ${width}: ${path}; figures=${result.figures}; Mermaid SVG=${result.mermaid}`);
    }
    // Dark theme / tablet coverage of the major media and navigation types.
    for (const path of ['/blog/','/posts/transformer/','/posts/visual-servoing-survey/','/posts/robot-kinematics/','/posts/flowmatching/']) {
      await navigate(origin+path,768,1000,'dark');
      check(await page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'), `Tablet dark ${path}: no horizontal overflow`);
      await screenshot(join(output, `${path.split('/').filter(Boolean).at(-1)}-768-dark.png`));
    }
    await navigate(`${origin}/posts/visual-servoing-survey/`,375,900);
    const before = requests.length;
    await page.evaluate(`document.querySelector('details:has([data-appendix])').open=true`);
    await delay(200);
    check(await page.evaluate(`document.querySelector('details:has([data-appendix])').querySelectorAll('template').length===0 && document.querySelector('details:has([data-appendix])').querySelectorAll('img[src]').length===2`), 'Opening handwritten appendix loads its preview images');
    check(!requests.slice(before).some(r=>/ibvs-handwritten-[12]\.jpg/.test(r.url)), 'Opening appendix still avoids full original JPEG downloads');
    await navigate(`${origin}/posts/transformer/`,1440,1000);
    await page.evaluate(`document.querySelector('.article-figure__link').scrollIntoView({block:'center',behavior:'instant'});document.querySelector('.article-figure__link').click()`);
    await delay(700);
    check(await page.evaluate(`document.querySelector('.glightbox-container')!==null`), 'Article image opens Chirpy lightbox');
    await page.send('Input.dispatchKeyEvent',{type:'rawKeyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27,nativeVirtualKeyCode:27});
    await page.send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27,nativeVirtualKeyCode:27});
    for(let i=0;i<50;i++) { if(await page.evaluate(`document.querySelector('.glightbox-container')===null`))break; await delay(50); }
    check(await page.evaluate(`document.querySelector('.glightbox-container')===null`), 'Escape closes image lightbox');
    for (const [path, id] of [['/posts/visual-servoing-survey/','fig-camera-projection'],['/posts/robot-kinematics/','demo-robot-kinematics-7dof'],['/posts/hilserl/','fig-hil-actor-critic'],['/posts/flowmatching/','fig-gaussian-flow-example']]) {
      await navigate(origin+path,1440,1000);
      const beforeHeight = await page.evaluate(`document.getElementById(${JSON.stringify(id)}).getBoundingClientRect().height`);
      await page.evaluate(`document.getElementById(${JSON.stringify(id)}).scrollIntoView({block:'center',behavior:'instant'})`);
      await delay(500);
      if(id.startsWith('demo-')) {
        await delay(2000);
        check(await page.evaluate(`document.getElementById(${JSON.stringify(id)}).querySelector('iframe')?.getBoundingClientRect().height===600`), 'Desktop demo honors its reviewed height');
        check(await page.evaluate(`Math.abs(document.getElementById(${JSON.stringify(id)}).getBoundingClientRect().height-${beforeHeight})<1`), 'Poster-to-demo replacement preserves reserved height');
      }
      await screenshot(join(output,`${id}-desktop.png`));
    }
    await page.send('Emulation.setScriptExecutionDisabled', { value:true });
    await navigate(`${origin}/blog/`,375,900);
    check(await page.evaluate(`document.querySelectorAll('[data-note-card]:not([hidden])').length===${expectedAll}`), `No-JS blog retains all ${expectedAll} articles`);
    await page.send('Emulation.setScriptExecutionDisabled', { value:false });
  }
  await writeFile(join(output, `${mode}-results.json`), JSON.stringify({mode,checks,failures,consoleErrors},null,2));
  console.log(`Finished ${checks.length} checks, ${failures.length} failures; report: .qa/browser/${mode}-results.json`);
  if (failures.length) process.exitCode=1;
} catch (error) {
  console.error(error.stack); process.exitCode=1;
} finally {
  if (browser) { try { await browser.send('Browser.close'); } catch { /* closed with the browser */ } browser.ws.close(); }
  if (page) page.ws.close();
  if (!browser) processHandle.kill();
}
