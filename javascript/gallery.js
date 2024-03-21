/* eslint-disable max-classes-per-file */

let ws;
let url;
let currentImage;
const el = {
  folders: undefined,
  files: undefined,
  search: undefined,
  status: undefined,
  btnSend: undefined,
};

// HTML Elements

class GalleryFolder extends HTMLElement {
  constructor(name) {
    super();
    this.name = name;
    this.shadow = this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    const style = document.createElement('style');
    style.textContent = `
      .gallery-folder {
        cursor: pointer;
        padding: 8px 6px 8px 6px;
      }
      .gallery-folder:hover {
        background-color: var(--button-primary-background-fill-hover);
      }
    `;
    this.shadow.appendChild(style);
    const div = document.createElement('div');
    div.className = 'gallery-folder';
    div.textContent = `\uf44a ${this.name}`;
    div.addEventListener('click', fetchFiles); // eslint-disable-line no-use-before-define
    this.shadow.appendChild(div);
  }
}

class GalleryFile extends HTMLElement {
  constructor({ folder, file, size, mtime }) {
    super();
    this.folder = folder;
    this.name = file;
    this.size = size;
    this.mtime = new Date(1000 * mtime);
    this.hash = undefined;
    this.exif = '';
    this.width = 0;
    this.height = 0;
    this.shadow = this.attachShadow({ mode: 'open' });
  }

  async connectedCallback() {
    const ext = this.name.split('.').pop().toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg'].includes(ext)) return;
    this.hash = await getHash(`${this.folder}/${this.name}/${this.size}/${this.mtime}`); // eslint-disable-line no-use-before-define
    const style = document.createElement('style');
    const width = opts.browser_fixed_width ? `${opts.extra_networks_card_size}px` : 'unset';
    style.textContent = `
      .gallery-file {
        object-fit: contain;
        cursor: pointer;
        height: ${opts.extra_networks_card_size}px;
        width: ${width};
      }
      .gallery-file:hover {
        filter: grayscale(100%);
      }
    `;

    this.shadow.appendChild(style);
    const img = document.createElement('img');
    img.className = 'gallery-file';
    img.loading = 'lazy';
    img.title = `Folder: ${this.folder}\nFile: ${this.name}\nSize: ${this.size.toLocaleString()} bytes\nModified: ${this.mtime.toLocaleString()}`;
    img.onload = async () => {
      this.width = img.naturalWidth;
      this.height = img.naturalHeight;
      img.title += `\nResolution: ${this.width} x ${this.height}`;
      this.title = img.title;
      // let exif = await getExif(img);
      // if (exif) this.exif = exif.replaceAll('<br>', '\n').replace(/<\/?[^>]+(>|$)/g, "");
    };
    img.src = `file=${this.folder}/${this.name}`;
    img.onclick = () => {
      currentImage = `${this.folder}/${this.name}`;
      el.btnSend.click();
    };
    this.title = img.title;
    this.style.display = this.title.toLowerCase().includes(el.search.value.toLowerCase()) ? 'unset' : 'none';
    this.shadow.appendChild(img);
  }
}

// methods

const gallerySendImage = (_images) => [currentImage]; // invoked by gadio button

async function getHash(str, algo = 'SHA-256') {
  const strBuf = new TextEncoder().encode(str);
  const hash = await crypto.subtle.digest(algo, strBuf);
  let hex = '';
  const view = new DataView(hash);
  for (let i = 0; i < hash.byteLength; i += 4) hex += (`00000000${view.getUint32(i).toString(16)}`).slice(-8);
  return hex;
}

async function wsConnect(socket, timeout = 2000) {
  const intrasleep = 100;
  const ttl = timeout / intrasleep;
  const isOpened = () => (socket.readyState === WebSocket.OPEN);
  if (socket.readyState !== WebSocket.CONNECTING) return isOpened();

  let loop = 0;
  while (socket.readyState === WebSocket.CONNECTING && loop < ttl) {
    await new Promise((resolve) => setTimeout(resolve, intrasleep)); // eslint-disable-line no-promise-executor-return
    loop++;
  }
  return isOpened();
}

async function gallerySearch(evt) {
  if (el.search.busy) clearTimeout(el.search.busy);
  el.search.busy = setTimeout(async () => {
    let numFound = 0;
    const all = Array.from(el.files.children);
    const str = el.search.value.toLowerCase();
    const r = /^(.+)([=<>])(.*)/;
    const t0 = performance.now();
    for (const f of all) {
      if (r.test(str)) {
        const match = str.match(r);
        const key = match[1].trim();
        const op = match[2].trim();
        let val = match[3].trim();
        if (key === 'mtime') val = new Date(val);
        console.log('HERE', key, op, val, f[key]);
        if (((op === '=') && (f[key] === val)) || ((op === '>') && (f[key] > val)) || ((op === '<') && (f[key] < val))) {
          f.style.display = 'unset';
          numFound++;
        } else {
          f.style.display = 'none';
        }
      } else if (f.title?.toLowerCase().includes(str) || f.exif?.toLowerCase().includes(str)) {
        f.style.display = 'unset';
        numFound++;
      } else {
        f.style.display = 'none';
      }
      const t1 = performance.now();
      el.status.innerText = `Filter | ${f.folder} | ${numFound.toLocaleString()}/${all.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
    }
  }, 250);
}

async function gallerySort(btn) {
  const t0 = performance.now();
  document.querySelectorAll('.gallery-separator').forEach((node) => el.files.removeChild(node)); // cannot sort separators
  const arr = Array.from(el.files.children);
  switch (btn.charCodeAt(0)) {
    case 61789:
      arr
        .sort((a, b) => a.name.localeCompare(b.name))
        .forEach((node) => el.files.appendChild(node));
      break;
    case 61790:
      arr
        .sort((b, a) => a.name.localeCompare(b.name))
        .forEach((node) => el.files.appendChild(node));
      break;
    case 61792:
      arr
        .sort((a, b) => a.size - b.size)
        .forEach((node) => el.files.appendChild(node));
      break;
    case 61793:
      arr
        .sort((b, a) => a.size - b.size)
        .forEach((node) => el.files.appendChild(node));
      break;
    case 61794:
      arr
        .sort((a, b) => a.width * a.height - b.width * b.height)
        .forEach((node) => el.files.appendChild(node));
      break;
    case 61795:
      arr
        .sort((b, a) => a.width * a.height - b.width * b.height)
        .forEach((node) => el.files.appendChild(node));
      break;
    case 61662:
      arr
        .sort((a, b) => a.mtime - b.mtime)
        .forEach((node) => el.files.appendChild(node));
      break;
    case 61661:
      arr
        .sort((b, a) => a.mtime - b.mtime)
        .forEach((node) => el.files.appendChild(node));
      break;
    default:
      break;
  }
  const t1 = performance.now();
  el.status.innerText = `Sort | ${arr.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
}

async function fetchFiles(evt) { // fetch file-by-file list over websockets
  el.files.innerHTML = '';
  if (!url) return;
  if (ws && ws.readyState === WebSocket.OPEN) ws.close(); // abort previous request
  ws = new WebSocket(`${url}/sdapi/v1/browser/files`);
  await wsConnect(ws);
  let numFiles = 0;
  el.status.innerText = `Folder | ${evt.target.name}`;
  const t0 = performance.now();
  let t1 = performance.now();
  let lastDir;
  ws.onmessage = (event) => { // time is 20% list 80% create item
    numFiles++;
    t1 = performance.now();
    if (event.data === '#END#') {
      ws.close();
    } else {
      const json = JSON.parse(event.data);
      const dir = json.file.match(/(.*)[\/\\]/) || '';
      if (dir?.[1] !== lastDir) { // create separator
        lastDir = dir[1];
        const sep = document.createElement('div');
        sep.className = 'gallery-separator';
        sep.innerText = lastDir;
        sep.title = lastDir;
        el.files.appendChild(sep);
      }
      const file = new GalleryFile(json);
      el.files.appendChild(file);
      el.status.innerText = `Folder | ${evt.target.name} | ${numFiles.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
    }
  };
  ws.onclose = (event) => {
    // log('gallery ws file enum', event);
  };
  ws.onerror = (event) => {
    log('gallery ws error', event);
  };
  ws.send(evt.target.name);
}

async function galleryVisible() {
  // if (el.folders.children.length > 0) return;
  const res = await fetch('/sdapi/v1/browser/folders');
  if (!res || res.status !== 200) return;
  el.folders.innerHTML = '';
  url = res.url.split('/sdapi')[0].replace('http', 'ws'); // update global url as ws need fqdn
  const folders = await res.json();
  for (const folder of folders) {
    const f = new GalleryFolder(folder);
    el.folders.appendChild(f);
  }
}

async function galleryHidden() { /**/ }

async function galleryObserve() { // triggered on gradio change to monitor when ui gets sufficiently constructed
  log('initBrowser');
  el.folders = gradioApp().getElementById('tab-gallery-folders');
  el.files = gradioApp().getElementById('tab-gallery-files');
  el.status = gradioApp().getElementById('tab-gallery-status');
  el.search = gradioApp().querySelector('#tab-gallery-search textarea');
  el.search.addEventListener('input', gallerySearch);
  el.btnSend = gradioApp().getElementById('tab-gallery-send-image');

  const intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio <= 0) galleryHidden();
    if (entries[0].intersectionRatio > 0) galleryVisible();
  });
  intersectionObserver.observe(el.folders);
}

// register on startup

customElements.define('gallery-folder', GalleryFolder);
customElements.define('gallery-file', GalleryFile);
onUiLoaded(galleryObserve);
