/* eslint-disable no-undef */

async function initStartup() {
  log('initStartup');

  // all items here are non-blocking async calls
  initModels();
  getUIDefaults();
  initiGenerationParams();
  initNotifications();
  initPromptChecker();
  initLogMonitor();
  initContextMenu();
  initDragDrop();
  initSettings();
  initImageViewer();
  initGallery();
  setupControlUI();
  setupExtraNetworks();

  // reconnect server session
  await reconnectUI();

  // make sure all of the ui is ready and options are loaded
  while (Object.keys(window.opts).length === 0) await sleep(50);
  executeCallbacks(uiReadyCallbacks);

  // optinally wait for modern ui
  if (window.waitForUiUxReady) await window.waitForUiUxReady();
  removeSplash();

  // post startup tasks that may take longer but are not critical
  setHints();
  initIndexDB();
}

onUiLoaded(initStartup);
onUiReady(() => log('uiReady'));

// onAfterUiUpdate(() => log('evt onAfterUiUpdate'));
// onUiLoaded(() => log('evt onUiLoaded'));
// onOptionsChanged(() => log('evt onOptionsChanged'));
// onUiTabChange(() => log('evt onUiTabChange'));
// onUiUpdate(() => log('evt onUiUpdate'));
