const serverTimeout = 5000;

const log = async (...msg) => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) window.logger.innerHTML += window.logPrettyPrint(...msg);
  console.log(ts, ...msg); // eslint-disable-line no-console
};

const debug = async (...msg) => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) window.logger.innerHTML += window.logPrettyPrint(...msg);
  console.debug(ts, ...msg); // eslint-disable-line no-console
};

const error = async (...msg) => {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  if (window.logger) window.logger.innerHTML += window.logPrettyPrint(...msg);
  console.error(ts, ...msg); // eslint-disable-line no-console
  const txt = msg.join(' ');
  if (!txt.includes('asctime') && !txt.includes('xhr.')) xhrPost('/sdapi/v1/log', { error: txt }); // eslint-disable-line no-use-before-define
};

const xhrInternal = (xhrObj, data, handler = undefined, errorHandler = undefined, ignore = false) => {
  const err = (msg) => {
    if (!ignore) {
      error(`${msg}: state=${xhrObj.readyState} status=${xhrObj.status} response=${xhrObj.responseText}`);
      if (errorHandler) errorHandler();
    }
  };

  xhrObj.setRequestHeader('Content-Type', 'application/json');
  xhrObj.timeout = serverTimeout;
  xhrObj.ontimeout = () => err('xhr.ontimeout');
  xhrObj.onerror = () => err('xhr.onerror');
  xhrObj.onabort = () => err('xhr.onabort');
  xhrObj.onreadystatechange = () => {
    if (xhrObj.readyState === 4) {
      if (xhrObj.status === 200) {
        try {
          const json = JSON.parse(xhrObj.responseText);
          if (handler) handler(json);
        } catch (e) {
          error(`xhr.onreadystatechange: ${e}`);
        }
      } else {
        err(`xhr.onreadystatechange: state=${xhrObj.readyState} status=${xhrObj.status} response=${xhrObj.responseText}`);
      }
    }
  };
  const req = JSON.stringify(data);
  xhrObj.send(req);
};

const xhrGet = (url, data, handler = undefined, errorHandler = undefined, ignore = false) => {
  const xhr = new XMLHttpRequest();
  const args = Object.keys(data).map((k) => `${encodeURIComponent(k)}=${encodeURIComponent(data[k])}`).join('&');
  xhr.open('GET', `${url}?${args}`, true);
  xhrInternal(xhr, data, handler, errorHandler, ignore);
};

function xhrPost(url, data, handler = undefined, errorHandler = undefined, ignore = false) {
  const xhr = new XMLHttpRequest();
  xhr.open('POST', url, true);
  xhrInternal(xhr, data, handler, errorHandler, ignore);
}
