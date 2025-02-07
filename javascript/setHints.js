const localeData = {
  data: [],
  timeout: null,
  finished: false,
  type: 2,
  el: null,
};

async function tooltipCreate() {
  localeData.el = document.createElement('div');
  localeData.el.className = 'tooltip';
  localeData.el.id = 'tooltip-container';
  localeData.el.innerText = 'this is a hint';
  gradioApp().appendChild(localeData.el);
  if (window.opts.tooltips === 'None') localeData.type = 0;
  if (window.opts.tooltips === 'Browser default') localeData.type = 1;
  if (window.opts.tooltips === 'UI tooltips') localeData.type = 2;
}

async function tooltipShow(e) {
  if (e.target.dataset.hint) {
    localeData.el.classList.add('tooltip-show');
    localeData.el.innerHTML = `<b>${e.target.textContent}</b><br>${e.target.dataset.hint}`;
    if (e.clientX > window.innerWidth / 2) {
      localeData.el.classList.add('tooltip-left');
    } else {
      localeData.el.classList.remove('tooltip-left');
    }
  }
}

async function tooltipHide(e) {
  localeData.el.classList.remove('tooltip-show');
}

async function validateHints(json, elements) {
  json.missing = [];
  const data = Object.values(json).flat().filter((e) => e.hint.length > 0);
  for (const e of data) e.label = e.label.toLowerCase().trim();
  let original = elements.map((e) => e.textContent.toLowerCase().trim()).sort();
  let duplicateUI = original.filter((e, i, a) => a.indexOf(e) !== i).sort();
  original = [...new Set(original)]; // remove duplicates
  duplicateUI = [...new Set(duplicateUI)]; // remove duplicates
  const current = data.map((e) => e.label.toLowerCase().trim()).sort();
  log('all elements:', original);
  log('all hints:', current);
  log('hints-differences', { elements: original.length, hints: current.length });
  const missingHints = original.filter((e) => !current.includes(e)).sort();
  const orphanedHints = current.filter((e) => !original.includes(e)).sort();
  const duplicateHints = current.filter((e, i, a) => a.indexOf(e) !== i).sort();
  log('duplicate hints:', duplicateHints);
  log('duplicate labels:', duplicateUI);
  return [missingHints, orphanedHints];
}

async function addMissingHints(json, missingHints) {
  if (missingHints.length === 0) return;
  json.missing = [];
  for (const h of missingHints.sort()) {
    if (h.length <= 1) continue;
    json.missing.push({ id: '', label: h, localized: '', hint: h });
  }
  log('missing hints', missingHints);
  log('added missing hints:', { missing: json.missing });
}

async function removeOrphanedHints(json, orphanedHints) {
  const data = Object.values(json).flat().filter((e) => e.hint.length > 0);
  for (const e of data) e.label = e.label.toLowerCase().trim();
  const orphaned = data.filter((e) => orphanedHints.includes(e.label));
  log('orphaned hints:', { orphaned });
}

async function replaceButtonText(el) {
  // https://www.nerdfonts.com/cheat-sheet
  // use unicode of icon with format nf-md-<icon>_circle
  const textIcons = {
    Generate: '\uf144',
    Enqueue: '\udb81\udc17',
    Stop: '\udb81\ude66',
    Skip: '\udb81\ude61',
    Pause: '\udb80\udfe5',
    Restore: '\udb82\udd9b',
    Clear: '\udb80\udd59',
    Networks: '\uf261',
  };
  if (textIcons[el.innerText]) {
    el.classList.add('button-icon');
    el.innerText = textIcons[el.innerText];
  }
}

async function setHints(analyze = false) {
  let json = {};
  if (localeData.finished) return;
  if (localeData.data.length === 0) {
    const res = await fetch('/file=html/locale_en.json');
    json = await res.json();
    localeData.data = Object.values(json).flat().filter((e) => e.hint.length > 0);
    for (const e of localeData.data) e.label = e.label.toLowerCase().trim();
  }
  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
  ];
  if (elements.length === 0) return;
  if (Object.keys(opts).length === 0) return;
  if (!localeData.el) tooltipCreate();
  let localized = 0;
  let hints = 0;
  localeData.finished = true;
  const t0 = performance.now();
  for (const el of elements) {
    const found = localeData.data.find((l) => l.label === el.textContent.toLowerCase().trim());
    if (found?.localized?.length > 0) {
      localized++;
      el.textContent = found.localized;
    }
    // replaceButtonText(el);
    if (found?.hint?.length > 0) {
      hints++;
      if (localeData.type === 1) {
        el.title = found.hint;
      } else if (localeData.type === 2) {
        el.dataset.hint = found.hint;
        el.addEventListener('mouseover', tooltipShow);
        el.addEventListener('mouseout', tooltipHide);
      } else {
        // tooltips disabled
      }
    }
  }
  const t1 = performance.now();
  log('setHints', { type: localeData.type, elements: elements.length, localized, hints, data: localeData.data.length, time: t1 - t0 });
  // sortUIElements();
  if (analyze) {
    const [missingHints, orphanedHints] = await validateHints(json, elements);
    await addMissingHints(json, missingHints);
    await removeOrphanedHints(json, orphanedHints);
  }
}

const analyzeHints = async () => {
  localeData.finished = false;
  localeData.data = [];
  await setHints(true);
};

/*
onAfterUiUpdate(async () => {
  if (localeData.timeout) clearTimeout(localeData.timeout);
  localeData.timeout = setTimeout(setHints, 250);
});
*/
