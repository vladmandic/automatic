const allLocales = ['en', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pt', 'hr', 'ru', 'zh'];
const localeData = {
  prev: null,
  locale: null,
  data: [],
  timeout: null,
  finished: false,
  type: 2,
  hint: null,
  btn: null,
};

async function cycleLocale() {
  log('cycleLocale', localeData.prev, localeData.locale);
  const index = allLocales.indexOf(localeData.prev);
  localeData.locale = allLocales[(index + 1) % allLocales.length];
  localeData.btn.innerText = localeData.locale;
  localeData.btn.style.backgroundColor = localeData.locale !== 'en' ? 'var(--primary-500)' : '';
  localeData.finished = false;
  localeData.data = [];
  localeData.prev = localeData.locale;
  window.opts.ui_locale = localeData.locale;
  await setHints(); // eslint-disable-line no-use-before-define
}

async function tooltipCreate() {
  localeData.hint = document.createElement('div');
  localeData.hint.className = 'tooltip';
  localeData.hint.id = 'tooltip-container';
  localeData.hint.innerText = 'this is a hint';
  gradioApp().appendChild(localeData.hint);
  localeData.btn = document.createElement('div');
  localeData.btn.className = 'locale';
  localeData.btn.id = 'locale-container';
  localeData.btn.innerText = localeData.locale;
  localeData.btn.onclick = cycleLocale;
  gradioApp().appendChild(localeData.btn);
  if (window.opts.tooltips === 'None') localeData.type = 0;
  if (window.opts.tooltips === 'Browser default') localeData.type = 1;
  if (window.opts.tooltips === 'UI tooltips') localeData.type = 2;
}

async function tooltipShow(e) {
  if (e.target.dataset.hint) {
    localeData.hint.classList.add('tooltip-show');
    localeData.hint.innerHTML = `<b>${e.target.textContent}</b><br>${e.target.dataset.hint}`;
    if (e.clientX > window.innerWidth / 2) {
      localeData.hint.classList.add('tooltip-left');
    } else {
      localeData.hint.classList.remove('tooltip-left');
    }
  }
}

async function tooltipHide(e) {
  localeData.hint.classList.remove('tooltip-show');
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

async function getLocaleData(desiredLocale = null) {
  if (desiredLocale) desiredLocale = desiredLocale.split(':')[0];
  if (desiredLocale === 'Auto') {
    try {
      localeData.locale = navigator.languages && navigator.languages.length ? navigator.languages[0] : navigator.language;
      localeData.locale = localeData.locale.split('-')[0];
      localeData.prev = localeData.locale;
    } catch (e) {
      localeData.locale = 'en';
      log('getLocale', e);
    }
  } else {
    localeData.locale = desiredLocale || 'en';
    localeData.prev = localeData.locale;
  }
  log('getLocale', desiredLocale, localeData.locale);
  // primary
  let json = {};
  try {
    let res = await fetch(`/file=html/locale_${localeData.locale}.json`);
    if (!res || !res.ok) {
      localeData.locale = 'en';
      res = await fetch(`/file=html/locale_${localeData.locale}.json`);
    }
    json = await res.json();
  } catch { /**/ }

  try {
    const res = await fetch(`/file=html/override_${localeData.locale}.json`);
    if (res && res.ok) json.override = await res.json();
  } catch { /**/ }

  return json;
}

async function setHints(analyze = false) {
  let json = {};
  if (localeData.finished) return;
  if (Object.keys(opts).length === 0) return;
  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
    ...Array.from(gradioApp().querySelectorAll('.label-wrap > span')),
  ];
  if (elements.length === 0) return;
  if (localeData.data.length === 0) {
    json = await getLocaleData(window.opts.ui_locale);
    let overrideData = json.override || {};
    overrideData = Object.values(overrideData).flat().filter((e) => e.hint.length > 0);
    localeData.data = Object.values(json).flat().filter((e) => e.hint.length > 0);
    Object.assign(localeData.data, overrideData);
    for (const e of localeData.data) e.label = e.label.toLowerCase().trim();
  }
  if (!localeData.hint) tooltipCreate();
  let localized = 0;
  let hints = 0;
  localeData.finished = true;
  const t0 = performance.now();
  for (const el of elements) {
    let found;
    if (el.dataset.original) found = localeData.data.find((l) => l.label === el.dataset.original.toLowerCase().trim());
    else found = localeData.data.find((l) => l.label === el.textContent.toLowerCase().trim());
    if (found?.localized?.length > 0) {
      if (!el.dataset.original) el.dataset.original = el.textContent;
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
  localeData.btn.style.backgroundColor = localeData.locale !== 'en' ? 'var(--primary-500)' : '';
  log('setHints', { type: localeData.type, locale: localeData.locale, elements: elements.length, localized, hints, data: localeData.data.length, time: t1 - t0 });
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
