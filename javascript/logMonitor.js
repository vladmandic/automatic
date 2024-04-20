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
  const addLogLine = (line) => {
    try {
      const l = JSON.parse(line.replaceAll('\n', ' '));
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
  };

  const cleanupLog = (atBottom) => {
    while (logMonitorEl.childElementCount > 100) logMonitorEl.removeChild(logMonitorEl.firstChild);
    if (atBottom) logMonitorEl.scrollTop = logMonitorEl.scrollHeight;
    else logMonitorEl.parentElement.style = 'border-bottom: 2px solid var(--highlight-color);';
    document.getElementById('logWarnings').innerText = logWarnings;
    document.getElementById('logErrors').innerText = logErrors;
  };

  // addLogLine(`{ "created": ${Date.now() / 1000}, "level":"DEBUG", "module":"ui", "facility":"logMonitor", "msg":"Log refresh" }`);
  if (logMonitorStatus) setTimeout(logMonitor, opts.logmonitor_refresh_period);
  else setTimeout(logMonitor, 10 * opts.logmonitor_refresh_period);
  if (!opts.logmonitor_show) return;
  logMonitorStatus = false;
  if (!logMonitorEl) {
    logMonitorEl = document.getElementById('logMonitorData');
    logMonitorEl.onscrollend = () => {
      const atBottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
      if (atBottom) logMonitorEl.parentElement.style = '';
    };
  }
  if (!logMonitorEl) return;
  const atBottom = logMonitorEl.scrollHeight <= (logMonitorEl.scrollTop + logMonitorEl.clientHeight);
  try {
    const res = await fetch('/sdapi/v1/log?clear=False');
    if (res?.ok) {
      logMonitorStatus = true;
      const lines = await res.json();
      if (logMonitorEl && lines?.length > 0) logMonitorEl.parentElement.parentElement.style.display = opts.logmonitor_show ? 'block' : 'none';
      for (const line of lines) addLogLine(line);
    } else {
      addLogLine(`{ "created": ${Date.now()}, "level":"ERROR", "module":"logMonitor", "facility":"ui", "msg":"Failed to fetch log: ${res?.status} ${res?.statusText}" }`);
      logErrors++;
    }
    cleanupLog(atBottom);
  } catch (err) {
    addLogLine(`{ "created": ${Date.now()}, "level":"ERROR", "module":"logMonitor", "facility":"ui", "msg":"Failed to fetch log: server unreachable" }`);
    logErrors++;
    cleanupLog(atBottom);
  }
}

async function initLogMonitor() {
  const el = document.getElementsByTagName('footer')[0];
  if (!el) return;
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
