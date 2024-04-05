const log = (...msg) => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) window.logger.innerHTML += window.logPrettyPrint(...msg);
  console.log(ts, ...msg); // eslint-disable-line no-console
};

const debug = (...msg) => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) window.logger.innerHTML += window.logPrettyPrint(...msg);
  console.debug(ts, ...msg); // eslint-disable-line no-console
};

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms)); // eslint-disable-line no-promise-executor-return
}

function gradioApp() {
  const elems = document.getElementsByTagName('gradio-app');
  const elem = elems.length === 0 ? document : elems[0];
  if (elem !== document) elem.getElementById = (id) => document.getElementById(id);
  return elem.shadowRoot ? elem.shadowRoot : elem;
}

function getUICurrentTab() {
  return gradioApp().querySelector('#tabs button.selected');
}

function getUICurrentTabContent() {
  return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])');
}

const get_uiCurrentTabContent = getUICurrentTabContent;
const get_uiCurrentTab = getUICurrentTab;
const uiAfterUpdateCallbacks = [];
const uiUpdateCallbacks = [];
const uiLoadedCallbacks = [];
const uiReadyCallbacks = [];
const uiTabChangeCallbacks = [];
const optionsChangedCallbacks = [];
let uiCurrentTab = null;
let uiAfterUpdateTimeout = null;

function onAfterUiUpdate(callback) {
  uiAfterUpdateCallbacks.push(callback);
}

function onUiUpdate(callback) {
  uiUpdateCallbacks.push(callback);
}

function onUiLoaded(callback) {
  uiLoadedCallbacks.push(callback);
}

function onUiReady(callback) {
  uiReadyCallbacks.push(callback);
}

function onUiTabChange(callback) {
  uiTabChangeCallbacks.push(callback);
}

function onOptionsChanged(callback) {
  optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
  // if (!uiLoaded) return
  for (const callback of queue) {
    try {
      callback(arg);
    } catch (e) {
      console.error('error running callback', callback, ':', e);
    }
  }
}

function scheduleAfterUiUpdateCallbacks() {
  clearTimeout(uiAfterUpdateTimeout);
  uiAfterUpdateTimeout = setTimeout(() => executeCallbacks(uiAfterUpdateCallbacks, 500));
}

let executedOnLoaded = false;
const ignoreElements = ['logMonitorData', 'logWarnings', 'logErrors', 'tooltip-container', 'logger'];
const ignoreClasses = ['wrap'];

async function mutationCallback(mutations) {
  let validMutations = mutations;
  validMutations = validMutations.filter((m) => m.target.nodeName !== 'LABEL');
  validMutations = validMutations.filter((m) => ignoreElements.indexOf(m.target.id) === -1);
  validMutations = validMutations.filter((m) => m.target.id !== 'logWarnings' && m.target.id !== 'logErrors');
  validMutations = validMutations.filter((m) => !m.target.classList?.contains('wrap'));
  if (validMutations.length < 1) return;

  if (!executedOnLoaded && gradioApp().getElementById('txt2img_prompt')) { // execute once
    executedOnLoaded = true;
    executeCallbacks(uiLoadedCallbacks);
  }
  if (executedOnLoaded) { // execute on each mutation
    executeCallbacks(uiUpdateCallbacks, mutations);
    scheduleAfterUiUpdateCallbacks();
  }
  const newTab = getUICurrentTab();
  if (newTab && (newTab !== uiCurrentTab)) {
    uiCurrentTab = newTab;
    executeCallbacks(uiTabChangeCallbacks);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const mutationObserver = new MutationObserver(mutationCallback);
  mutationObserver.observe(gradioApp(), { childList: true, subtree: true });
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', (e) => {
  let handled = false;
  if (e.key !== undefined) {
    if ((e.key === 'Enter' && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
  } else if (e.keyCode !== undefined) {
    if ((e.keyCode === 13 && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
  }
  if (handled) {
    const button = getUICurrentTabContent().querySelector('button[id$=_generate]');
    if (button) button.click();
    e.preventDefault();
  }
});

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
  if (el === document) return true;
  const computedStyle = getComputedStyle(el);
  const isVisible = computedStyle.display !== 'none';
  if (!isVisible) return false;
  return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
  const clRect = el.getBoundingClientRect();
  const windowHeight = window.innerHeight;
  const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;
  return isOnScreen;
}
