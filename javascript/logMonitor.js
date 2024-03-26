let logMonitorEl = null;
let logMonitorStatus = true;
let logWarnings = 0;
let logErrors = 0;

function dateToStr(ts) {
  const dt = new Date(1000 * ts);
  const year = dt.getFullYear();
  const mo = String(dt.getMonth() + 1).padStart(2, '0');
  const day = String(dt.getDate()).padStart(2, '0');
  const hour = String(dt.getHours()).padStart(2, '0');
  const min = String(dt.getMinutes()).padStart(2, '0');
  const sec = String(dt.getSeconds()).padStart(2, '0');
  const ms = String(dt.getMilliseconds()).padStart(3, '0');
  const s = `${year}-${mo}-${day} ${hour}:${min}:${sec}.${ms}`;
  return s;
}

async function logMonitor() {
  if (logMonitorStatus) setTimeout(logMonitor, opts.logmonitor_refresh_period);
  if (!opts.logmonitor_show) return;
  logMonitorStatus = false;
  let res;
  try { res = await fetch('/sdapi/v1/log?clear=True'); } catch {}
  if (res?.ok) {
    logMonitorStatus = true;
    if (!logMonitorEl) {
      logMonitorEl = document.getElementById('logMonitorData');
      logMonitorEl.onscrollend = () => {
        const at_bottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
        if (at_bottom) logMonitorEl.parentElement.style = '';
      };
    }
    if (!logMonitorEl) return;
    const at_bottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
    const lines = await res.json();
    if (logMonitorEl && lines?.length > 0) logMonitorEl.parentElement.parentElement.style.display = opts.logmonitor_show ? 'block' : 'none';
    for (const line of lines) {
      try {
        const l = JSON.parse(line);
        const row = document.createElement('tr');
        // row.style = 'padding: 10px; margin: 0;';
        const level = `<td style="color: var(--color-${l.level.toLowerCase()})">${l.level}</td>`;
        if (l.level === 'WARNING') logWarnings++;
        if (l.level === 'ERROR') logErrors++;
        const module = `<td style="color: var(--var(--neutral-400))">${l.module}</td>`;
        row.innerHTML = `<td>${dateToStr(l.created)}</td>${level}<td>${l.facility}</td>${module}<td>${l.msg}</td>`;
        logMonitorEl.appendChild(row);
      } catch (e) {
        console.log('logMonitor', e);
        console.error('logMonitor line', line);
      }
    }
    while (logMonitorEl.childElementCount > 100) logMonitorEl.removeChild(logMonitorEl.firstChild);
    if (at_bottom) logMonitorEl.scrollTop = logMonitorEl.scrollHeight;
    else if (lines?.length > 0) logMonitorEl.parentElement.style = 'border-bottom: 2px solid var(--highlight-color);';
    document.getElementById('logWarnings').innerText = logWarnings;
    document.getElementById('logErrors').innerText = logErrors;
  }
}

let logMonitorInitialized = false;

async function initLogMonitor() {
  if (logMonitorInitialized) return;
  const el = document.getElementsByTagName('footer')[0];
  if (!el) return;
  logMonitorInitialized = true;
  el.classList.add('log-monitor');
  el.innerHTML = `
    <table id="logMonitor" style="width: 100%;">
      <thead style="display: block; text-align: left; border-bottom: solid 1px var(--button-primary-border-color)">
        <tr>
          <th style="width: 144px">Time</th>
          <th>Level</th>
          <th style="width: 0"></th>
          <th style="width: 154px">Module</th>
          <th>Message</th>
          <th style="position: absolute; right: 7em">Warnings <span id="logWarnings">0</span></th>
          <th style="position: absolute; right: 1em">Errors <span id="logErrors">0</span></th>
        </tr>
      </thead>
      <tbody id="logMonitorData" style="white-space: nowrap; height: 10vh; width: 100vw; display: block; overflow-x: hidden; overflow-y: scroll; color: var(--neutral-400)">
      </tbody>
    </table>
  `;
  el.style.display = 'none';
  fetch(`/sdapi/v1/start?agent=${encodeURI(navigator.userAgent)}`);
  logMonitor();
  log('initLogMonitor');
}

onAfterUiUpdate(initLogMonitor);
