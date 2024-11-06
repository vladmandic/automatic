/* eslint-disable no-undef */

async function initStartup() {
  log('initStartup');
  if (window.setupLogger) await setupLogger();

  // all items here are non-blocking async calls
  initModels();
  getUIDefaults();
  initPromptChecker();
  initLogMonitor();
  initContextMenu();
  initDragDrop();
  initAccordions();
  initSettings();
  initImageViewer();
  initGallery();
  initiGenerationParams();
  initChangelog();
  setupControlUI();

  // reconnect server session
  await reconnectUI();

  // make sure all of the ui is ready and options are loaded
  while (Object.keys(window.opts).length === 0) await sleep(50);
  executeCallbacks(uiReadyCallbacks);
  setupExtraNetworks();

  // optinally wait for modern ui
  if (window.waitForUiReady) await waitForUiReady();
  removeSplash();

  // post startup tasks that may take longer but are not critical
  showNetworks();
  setHints();
  applyStyles();
  initIndexDB();
}

onUiLoaded(initStartup);
onUiReady(() => log('uiReady'));

// onAfterUiUpdate(() => log('evt onAfterUiUpdate'));
// onUiLoaded(() => log('evt onUiLoaded'));
// onOptionsChanged(() => log('evt onOptionsChanged'));
// onUiTabChange(() => log('evt onUiTabChange'));
// onUiUpdate(() => log('evt onUiUpdate'));
