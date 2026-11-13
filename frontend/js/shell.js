/* Oslian shell: Home, global view switching, the guided demo and the
 * workflow-transition disclosure. Medication Safety (#view-rx) is rendered by
 * the inline script in templates/index.html; the Synthetic FHIR Stream
 * (#view-stream) by /static/stream/main.js. Nothing crosses between them.
 */
(function () {
  'use strict';

  function el(id) { return document.getElementById(id); }

  function renderHome() {
    var root = el('view-home');
    if (!root) return;
    root.innerHTML =
      '<div class="shell"><div class="home">' +
        '<div class="home-lead">' +
          '<h1 class="display">Two independent research demonstrations</h1>' +
          '<p class="note">Oslian contains two separate research demonstrations. They use different cases, different ' +
          'engines and different evidence. They are not two stages of one patient, and neither reads the other\'s ' +
          'output. Pick the one you want to look at.</p>' +
        '</div>' +
        '<div class="workflow-cards">' +
          '<div class="wcard">' +
            '<div class="wcard-h"><span class="k">01</span><h2>Medication Safety</h2></div>' +
            '<ul class="wcard-list">' +
              '<li>Illustrative case scenarios</li>' +
              '<li>Symbolic medication-safety engine</li>' +
              '<li>Optional live openFDA labels</li>' +
              '<li>Independent of the forecast runtime</li>' +
              '<li>No automated prescribing</li>' +
            '</ul>' +
            '<div class="wcard-foot">' +
              '<button class="btn primary" type="button" data-shell="go-rx">Explore medication safety</button>' +
            '</div>' +
          '</div>' +
          '<div class="wcard">' +
            '<div class="wcard-h"><span class="k">02</span><h2>Monitoring</h2></div>' +
            '<ul class="wcard-list">' +
              '<li>Live synthetic FHIR R4 stream, not a hospital connection</li>' +
              '<li>Retrospective replay of a deidentified public eICU demo case</li>' +
              '<li>Anatomy and clinical-function navigation with a full chart for every signal</li>' +
              '<li>oslian/Model 2026-09-02 release, 41 cells, behind an explicit runtime gate</li>' +
              '<li>Calibrated 90% intervals only where the release validated them; no clinical alerts</li>' +
            '</ul>' +
            '<div class="wcard-foot">' +
              '<button class="btn primary" type="button" data-shell="go-stream">Open monitoring</button>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="home-foot">' +
          '<div><div class="h-sec">Guided demo</div>' +
          '<p class="note">Walks both chapters in order: the medication-safety method first, then the synthetic ' +
          'FHIR stream. The change of case and model pathway between them is disclosed and has to be ' +
          'acknowledged.</p></div>' +
          '<button class="btn" type="button" data-shell="start-guided">Start guided demo</button>' +
        '</div>' +
        '<p class="note home-note">This page loads no model and opens no stream. Nothing runs until you choose a workflow.</p>' +
      '</div></div>';
  }

  function setView(name) {
    document.body.setAttribute('data-view', name);
    Array.prototype.forEach.call(document.querySelectorAll('.nav-view'), function (b) {
      var on = b.getAttribute('data-view') === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-current', on ? 'true' : 'false');
    });
    if (name === 'home') renderHome();
    if (name === 'stream' && window.OslianStream) window.OslianStream.open();
  }

  var guided = null;
  function startGuided() {
    guided = { chapter: 1 };
    setView('rx');
    if (window.OslerRx && window.OslerRx.runDefaultScenario) window.OslerRx.runDefaultScenario();
    else if (window.OslerRx && window.OslerRx.openCasePicker) window.OslerRx.openCasePicker();
  }

  var returnFocus = null;
  function requestStreamTransition() {
    returnFocus = document.activeElement;
    var scrim = el('transitionScrim');
    if (!scrim) { commitTransition(); return; }
    scrim.setAttribute('data-open', '1');
    var ack = el('transitionAck');
    if (ack) ack.focus();
  }
  function closeTransition(opts) {
    var scrim = el('transitionScrim');
    if (scrim) scrim.setAttribute('data-open', '0');
    if (!(opts && opts.restoreFocus === false) && returnFocus && returnFocus.focus) returnFocus.focus();
    returnFocus = null;
  }
  function commitTransition() {
    closeTransition({ restoreFocus: false });
    if (guided) guided.chapter = 2;
    setView('stream');
  }

  document.addEventListener('click', function (e) {
    var nav = e.target.closest ? e.target.closest('.nav-view') : null;
    if (nav) { setView(nav.getAttribute('data-view')); return; }
    var t = e.target.closest ? e.target.closest('[data-shell]') : null;
    if (!t) return;
    var action = t.getAttribute('data-shell');
    if (action === 'go-rx') setView('rx');
    else if (action === 'go-stream') setView('stream');
    else if (action === 'start-guided') startGuided();
  });
  var ack = el('transitionAck'), cancel = el('transitionCancel'), close = el('transitionClose');
  if (ack) ack.addEventListener('click', commitTransition);
  if (cancel) cancel.addEventListener('click', closeTransition);
  if (close) close.addEventListener('click', closeTransition);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && el('transitionScrim') && el('transitionScrim').getAttribute('data-open') === '1') closeTransition();
  });

  renderHome();
  // deep link: #stream[?organ=<id>&run=<run_id>&view=all] opens the stream view directly
  var hash = String(window.location.hash || '');
  if (hash.indexOf('#stream') === 0) setView('stream');
  else if (hash === '#rx') setView('rx');
  else setView('home');

  window.OslianViews = { setView: setView, startGuided: startGuided, requestStreamTransition: requestStreamTransition };
})();
