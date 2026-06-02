  (function () {
    'use strict';

    var API_BASE = 'https://api.ongiini.ai/v1/learn';
    var LS_KEY = 'ongiini-learn:learner_id';

    // ── State ──────────────────────────────────────────────
    var state = {
      learner_id: localStorage.getItem(LS_KEY) || null,
      view: 'landing',                     // 'landing' | 'learn'
      intake_field: null,                  // string | null while in intake
      profile: null,
      goal: null,                          // active goal info
      goals: [],                           // all goals
      curriculum_outline: null,
      progress: { total_seen: 0, total_correct: 0, by_box: {} },
      thread: [],                          // persisted server-side messages
      active_card_id: null,                // latest unanswered exercise card_id
      module_progress: [],                 // per-module rollup from /turn
      active_module_id: null,
      busy: false,                         // /turn in flight
      // Pending language pair selected from a topic card before
      // creating a goal — flushed into /goals/new when intake finishes
      // or when the user creates a goal from the drawer.
      pending_target: null,
      pending_source: null,
      // The chat-focus fallback path: a returning learner with
      // completed intake but no captured objective is asked, in chat,
      // "What do you want to focus on for {target}?" — the composer's
      // btnSend handler checks this flag and routes the typed answer
      // to /goals/new instead of /turn or /intake.
      awaiting_focus: false,
    };

    var LANGUAGE_DISPLAY = {
      afrikaans: 'Afrikaans',
      english: 'English',
      german: 'German',
    };

    // ── i18n ────────────────────────────────────────────────
    // Single-locale-at-a-time: the active source_language drives the
    // UI language. Three bundles ship as static JSON next to the SPA;
    // the loader picks the right one and exposes t(key, vars) +
    // applyI18nToDom() for static elements tagged with data-i18n.
    var LOCALE_KEY = 'ongiini-learn:locale';
    var SUPPORTED_LOCALES = ['en', 'af', 'de'];
    var SOURCE_TO_LOCALE = {
      english: 'en', afrikaans: 'af', german: 'de',
    };
    var i18nBundle = {};
    var currentLocale = 'en';

    function detectInitialLocale() {
      // Priority: previously-saved locale → browser language (if it
      // maps to a supported one) → English.
      var saved = localStorage.getItem(LOCALE_KEY);
      if (saved && SUPPORTED_LOCALES.indexOf(saved) !== -1) return saved;
      var browser = (navigator.language || 'en').toLowerCase().slice(0, 2);
      if (SUPPORTED_LOCALES.indexOf(browser) !== -1) return browser;
      return 'en';
    }

    function lookupKey(obj, dottedKey) {
      var parts = dottedKey.split('.');
      var v = obj;
      for (var i = 0; i < parts.length; i++) {
        if (!v || typeof v !== 'object') return null;
        v = v[parts[i]];
      }
      return (typeof v === 'string') ? v : null;
    }

    function t(key, vars) {
      // Resolve from active bundle; fall back to the key itself so
      // missing translations are visible during development rather
      // than vanishing silently.
      var raw = lookupKey(i18nBundle, key);
      if (raw === null) return key;
      if (!vars) return raw;
      return raw.replace(/\{(\w+)\}/g, function (_, name) {
        return Object.prototype.hasOwnProperty.call(vars, name)
          ? String(vars[name]) : ('{' + name + '}');
      });
    }

    function applyI18nToDom() {
      // Substitute textContent for every element tagged
      // data-i18n="bundle.key" and the equivalent for attributes
      // (placeholder, aria-label, title) via data-i18n-attr.
      var nodes = document.querySelectorAll('[data-i18n]');
      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var key = n.getAttribute('data-i18n');
        var val = lookupKey(i18nBundle, key);
        if (val !== null) n.textContent = val;
      }
      var attrNodes = document.querySelectorAll('[data-i18n-attr]');
      for (var j = 0; j < attrNodes.length; j++) {
        var node = attrNodes[j];
        var spec = node.getAttribute('data-i18n-attr');   // "attr=key,attr=key"
        spec.split(',').forEach(function (pair) {
          var pieces = pair.split('=');
          if (pieces.length !== 2) return;
          var attr = pieces[0].trim();
          var key = pieces[1].trim();
          var val = lookupKey(i18nBundle, key);
          if (val !== null) node.setAttribute(attr, val);
        });
      }
      document.documentElement.lang = currentLocale;
    }

    async function loadLocale(locale) {
      if (SUPPORTED_LOCALES.indexOf(locale) === -1) locale = 'en';
      try {
        var resp = await fetch('i18n/' + locale + '.json', {
          credentials: 'omit',
        });
        if (!resp.ok) throw new Error('locale fetch ' + resp.status);
        i18nBundle = await resp.json();
        currentLocale = locale;
        localStorage.setItem(LOCALE_KEY, locale);
        applyI18nToDom();
      } catch (e) {
        // Network / parse failure — keep the previous bundle if any.
        // If this was the cold-load and even the default failed, t()
        // will echo keys, which is at least visible.
        console.warn('i18n load failed:', e);
      }
    }

    function setLocaleFromSource(source) {
      var locale = SOURCE_TO_LOCALE[source];
      if (locale && locale !== currentLocale) {
        loadLocale(locale);
      }
    }

    var urlParams = new URLSearchParams(window.location.search);
    var magicToken = urlParams.get('t');

    var $ = function (id) { return document.getElementById(id); };

    // DOM handles
    var topbar = $('topbar');
    var currTitleBtn = $('currTitleBtn');
    var currTitleText = $('currTitleText');
    var btnMenu = $('btnMenu');
    var mainSiteLink = $('mainSiteLink');
    var landing = $('landing');
    var learnSurface = $('learn-surface');
    var currPanel = $('currPanel');
    var currSummary = $('currSummary');
    var currModules = $('currModules');
    var currProgress = $('currProgress');
    var currProgressText = $('currProgressText');
    var currProgressBoxes = $('currProgressBoxes');
    var thread = $('thread');
    var progressStrip = $('progressStrip');
    var progressLabel = $('progressLabel');
    var progressFill = $('progressFill');
    var progressCount = $('progressCount');
    var errorBanner = $('errorBanner');
    var composerWrap = $('composerWrap');
    var composer = $('composer');
    var btnSend = $('btnSend');
    var drawerOverlay = $('drawerOverlay');
    var drawer = $('drawer');
    var btnDrawerClose = $('btnDrawerClose');
    var goalList = $('goalList');
    var btnNewGoal = $('btnNewGoal');
    var btnRestart = $('btnRestart');
    var btnArchive = $('btnArchive');
    var btnClear = $('btnClear');
    var topicAfrikaans = $('topic-afrikaans');
    var topicEnglish = $('topic-english');
    var topicGerman = $('topic-german');
    var btnStartClosing = $('btnStartClosing');
    var modalRoot = $('modalRoot');

    // ── HTTP helper ────────────────────────────────────────
    async function api(path, body) {
      var resp = await fetch(API_BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'omit',
        body: JSON.stringify(body || {}),
      });
      if (!resp.ok) {
        var detail = '';
        try { var j = await resp.json(); detail = j.detail || ''; } catch (e) {}
        throw new Error(detail || ('HTTP ' + resp.status));
      }
      return resp.json();
    }

    function showError(msg) {
      errorBanner.textContent = msg;
      errorBanner.classList.add('is-visible');
      setTimeout(function () { errorBanner.classList.remove('is-visible'); }, 6000);
    }

    // ── View transitions ───────────────────────────────────
    function enterLearnView() {
      state.view = 'learn';
      topbar.dataset.mode = 'learn';
      document.body.classList.add('learn-mode');   // lock body to viewport
      landing.hidden = true;
      learnSurface.hidden = false;
      composerWrap.hidden = false;
      btnMenu.hidden = false;
      mainSiteLink.hidden = true;
      currTitleBtn.hidden = false;
      window.scrollTo(0, 0);
    }

    function exitToLanding() {
      state.view = 'landing';
      topbar.dataset.mode = 'landing';
      document.body.classList.remove('learn-mode');
      landing.hidden = false;
      learnSurface.hidden = true;
      composerWrap.hidden = true;
      btnMenu.hidden = true;
      mainSiteLink.hidden = false;
      currTitleBtn.hidden = true;
      progressStrip.hidden = true;
      closeCurriculumPanel();
      thread.innerHTML = '';
    }

    // ── Renderers ──────────────────────────────────────────
    function el(tag, cls, text) {
      var n = document.createElement(tag);
      if (cls) n.className = cls;
      if (text != null) n.textContent = text;
      return n;
    }

    function renderCoachText(msg) {
      var wrap = el('div', 'msg msg--coach');
      if (msg.payload && msg.payload.meta && msg.payload.meta.error) {
        wrap.classList.add('has-error');
      }
      var bubble = el('div', 'bubble', String((msg.payload && msg.payload.text) || ''));
      wrap.appendChild(bubble);
      return wrap;
    }

    function renderLearnerText(msg) {
      var wrap = el('div', 'msg msg--learner');
      var bubble = el('div', 'bubble', String((msg.payload && msg.payload.text) || ''));
      wrap.appendChild(bubble);
      return wrap;
    }

    function renderLesson(msg) {
      var p = msg.payload || {};
      var wrap = el('div', 'lesson-card');
      wrap.appendChild(el('div', 'lesson-eyebrow', t('card.lesson_eyebrow')));
      if (p.title) wrap.appendChild(el('h3', 'lesson-title', String(p.title)));
      if (p.body) {
        var body = el('div', 'lesson-body');
        body.textContent = String(p.body);
        wrap.appendChild(body);
      }
      if (Array.isArray(p.examples) && p.examples.length) {
        var ul = el('ul', 'lesson-examples');
        p.examples.forEach(function (ex) {
          ul.appendChild(el('li', null, String(ex)));
        });
        wrap.appendChild(ul);
      }
      var action = el('div', 'lesson-action');
      var btn = el('button', null, t('card.got_it'));
      btn.type = 'button';
      btn.addEventListener('click', function () { sendTurn(null); });
      action.appendChild(btn);
      wrap.appendChild(action);
      return wrap;
    }

    function renderExercise(msg) {
      var p = msg.payload || {};
      var isActive = !msg.answered;
      var wrap = el('div', 'exercise-card');
      wrap.dataset.active = isActive ? 'true' : 'false';
      var ct = p.card_type || 'card';
      wrap.dataset.cardType = ct;
      var eyebrow;
      if (p.review_box) {
        // Re-review surfaced by the SRS scheduler — distinct visual
        // signal so the learner recognises "I've seen this one
        // before" vs a brand-new card from the model.
        eyebrow = t('card.review_eyebrow', { box: p.review_box }) + ' · ' +
          (isActive ? t('card.awaiting') : t('card.answered'));
      } else {
        // Per-type eyebrow label — every CARD_TYPES member has a
        // card.type.<ct> entry in i18n; if a future type slips through
        // without one, t() returns the raw key, which is fine.
        var labelKey = 'card.type.' + ct;
        var label = t(labelKey);
        if (label === labelKey) label = ct;
        eyebrow = label + ' · ' +
          (isActive ? t('card.awaiting') : t('card.answered'));
      }
      wrap.appendChild(el('div', 'exercise-eyebrow', eyebrow));

      // Per-type structural extras BEFORE the prompt: grammar drills
      // show their source sentence; dialogue cards show the
      // conversation so the prompt slot is contextually anchored.
      if (ct === 'grammar' && p.source_sentence) {
        var src = el('div', 'card-source-sentence');
        src.appendChild(el('span', 'card-source-eyebrow', t('card.grammar.source_label')));
        src.appendChild(el('p', 'card-source-text', String(p.source_sentence)));
        wrap.appendChild(src);
      }
      if (ct === 'dialogue' && Array.isArray(p.turns) && p.turns.length) {
        var convo = el('div', 'card-dialogue-turns');
        p.turns.forEach(function (turnObj) {
          if (!turnObj || typeof turnObj !== 'object') return;
          var spk = String(turnObj.speaker || '');
          var txt = String(turnObj.text || '');
          var bubble = el('div', 'dialogue-bubble');
          if (spk) bubble.appendChild(el('div', 'dialogue-speaker', spk));
          bubble.appendChild(el('div', 'dialogue-line', txt));
          convo.appendChild(bubble);
        });
        wrap.appendChild(convo);
      }

      if (p.prompt_text) wrap.appendChild(el('p', 'exercise-prompt', String(p.prompt_text)));

      // Per-type structural extras AFTER the prompt: multiple-choice
      // options become click-to-answer buttons; reorder shows a
      // token bank that feeds the composer; proverb optionally
      // surfaces a cultural-note panel once answered.
      if (ct === 'multiple_choice' && Array.isArray(p.options) && p.options.length) {
        var optsWrap = el('div', 'card-mc-options');
        p.options.forEach(function (opt) {
          if (!opt || typeof opt !== 'object') return;
          var lbl = String(opt.label || '');
          var txt = String(opt.text || '');
          var btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'mc-option';
          btn.dataset.label = lbl;
          var letterSpan = el('span', 'mc-letter', lbl);
          var textSpan = el('span', 'mc-text', txt);
          btn.appendChild(letterSpan);
          btn.appendChild(textSpan);
          if (isActive) {
            btn.addEventListener('click', function () {
              if (state.busy) return;
              // Disable every option button in this card to prevent
              // double-submit while the turn is in flight.
              optsWrap.querySelectorAll('.mc-option').forEach(function (b) {
                b.disabled = true;
              });
              btn.classList.add('mc-option-selected');
              appendMessage({ kind: 'learner_text', payload: { text: lbl } });
              sendTurn(lbl, { optimisticLearnerText: true });
            });
          } else {
            btn.disabled = true;
          }
          optsWrap.appendChild(btn);
        });
        wrap.appendChild(optsWrap);
      }
      if (ct === 'reorder' && Array.isArray(p.tokens) && p.tokens.length) {
        // Token bank — clicking a chip appends it to the composer so
        // the learner builds the sentence by tapping. They can still
        // type freely if they prefer.
        var bank = el('div', 'card-reorder-bank');
        var note = el('div', 'reorder-hint', t('card.reorder.hint'));
        bank.appendChild(note);
        var chipRow = el('div', 'reorder-chips');
        var shuffled = p.tokens.slice();
        // Light shuffle so the bank isn't already in the right order.
        // (deterministic-ish: rotate by length so re-renders are stable)
        var shift = (shuffled.length % 3) + 1;
        shuffled = shuffled.slice(shift).concat(shuffled.slice(0, shift));
        shuffled.forEach(function (tok) {
          var chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'reorder-chip';
          chip.textContent = String(tok);
          if (isActive) {
            chip.addEventListener('click', function () {
              if (state.busy) return;
              var existing = composer.value.trim();
              composer.value = (existing ? existing + ' ' : '') + String(tok);
              composer.dispatchEvent(new Event('input'));
              composer.focus();
            });
          } else {
            chip.disabled = true;
          }
          chipRow.appendChild(chip);
        });
        bank.appendChild(chipRow);
        wrap.appendChild(bank);
      }
      if (ct === 'proverb' && p.cultural_note && !isActive) {
        var note2 = el('div', 'card-cultural-note');
        note2.appendChild(el('span', 'card-cultural-eyebrow', t('card.proverb.cultural_label')));
        note2.appendChild(el('p', 'card-cultural-text', String(p.cultural_note)));
        wrap.appendChild(note2);
      }

      if (p.hint_text) {
        var hintRow = el('div', 'exercise-hint-row');
        var hintBtn = document.createElement('button');
        hintBtn.type = 'button';
        hintBtn.className = 'hint-btn';
        // SVG icon + translated label. innerHTML stays for the SVG
        // (closed string constant); the label is appended as a text
        // node to keep the t() value safe even if a translator slips
        // an HTML-looking character into a future locale.
        hintBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M12 18h.01M9 21h6M12 3a6 6 0 0 0-3.5 10.9c.5.4.9 1 1.1 1.6L10 18h4l.4-2.5c.2-.6.6-1.2 1.1-1.6A6 6 0 0 0 12 3Z"/></svg>';
        hintBtn.appendChild(document.createTextNode(' ' + t('card.show_hint')));
        var hintReveal = el('div', 'hint-revealed');
        hintReveal.textContent = String(p.hint_text);
        hintReveal.hidden = true;
        hintBtn.addEventListener('click', function () {
          hintReveal.hidden = false;
          hintBtn.hidden = true;
        });
        hintRow.appendChild(hintBtn);
        hintRow.appendChild(hintReveal);
        wrap.appendChild(hintRow);
      }
      if (isActive) state.active_card_id = msg.card_id || null;
      return wrap;
    }

    function renderFeedback(msg) {
      var p = msg.payload || {};
      var wrap = el('div', 'feedback-card');
      wrap.dataset.rating = String(p.rating || 'partial');
      wrap.appendChild(el('div', 'feedback-rating', String(p.rating || '').toUpperCase()));
      wrap.appendChild(el('p', 'feedback-text', String(p.feedback || '')));
      return wrap;
    }

    function renderProgress(msg) {
      var p = msg.payload || {};
      var seen = p.total_seen || 0;
      var correct = p.total_correct || 0;
      var pct = seen ? Math.round(100 * correct / seen) : 0;
      var chip = el('div', 'progress-chip');
      chip.appendChild(document.createTextNode(
        t('progress.split_template', { seen: seen, pct: pct, box: p.box || 1 })
      ));
      var boxes = el('span', 'boxes');
      for (var b = 1; b <= 5; b++) {
        var n = (p.by_box || {})[b] || 0;
        var bar = document.createElement('b');
        bar.dataset.box = String(b);
        bar.style.height = Math.max(3, Math.min(14, 3 + n * 2)) + 'px';
        boxes.appendChild(bar);
      }
      chip.appendChild(boxes);
      return chip;
    }

    function renderMessage(msg) {
      switch (msg.kind) {
        case 'coach_text':   return renderCoachText(msg);
        case 'learner_text': return renderLearnerText(msg);
        case 'lesson':       return renderLesson(msg);
        case 'exercise':     return renderExercise(msg);
        case 'feedback':     return renderFeedback(msg);
        case 'progress':     return renderProgress(msg);
        default:             return null;
      }
    }

    function scrollThreadToBottom() {
      // Double rAF: the first lets the just-appended DOM mutation
      // settle through layout; the second runs AFTER the reflow has
      // grown thread.scrollHeight to include the new content. Without
      // both, big cards (lessons / exercises with examples) can be
      // appended and the scroll fires before the thread's height has
      // grown, so the new card lands below the visible fold.
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          thread.scrollTop = thread.scrollHeight;
        });
      });
    }

    function appendMessage(msg) {
      var node = renderMessage(msg);
      if (node) {
        thread.appendChild(node);
        scrollThreadToBottom();
      }
    }

    function rerenderThread(messages) {
      thread.innerHTML = '';
      state.active_card_id = null;
      (messages || []).forEach(function (m) { appendMessage(m); });
    }

    function appendSkeleton() {
      var wrap = el('div', 'msg msg--coach');
      wrap.dataset.skeleton = '1';
      var skel = el('div', 'skeleton-bubble');
      skel.appendChild(el('span'));
      skel.appendChild(el('span'));
      skel.appendChild(el('span'));
      wrap.appendChild(skel);
      thread.appendChild(wrap);
      scrollThreadToBottom();
      return wrap;
    }

    function removeSkeleton(node) {
      if (node && node.parentNode) node.parentNode.removeChild(node);
    }

    // Append a CLIENT-SIDE coach bubble for intake prompts. Intake
    // questions aren't persisted in learner_messages (they pre-date
    // the goal), so the frontend renders them locally and forgets
    // them on refresh — the next /sessions response will surface the
    // correct next_intake_prompt to recover.
    function appendIntakePrompt(text) {
      appendMessage({ kind: 'coach_text', payload: { text: text } });
    }

    // ── Curriculum panel ───────────────────────────────────
    function renderCurriculumPanel() {
      if (!state.curriculum_outline) {
        currSummary.textContent = t('panel.summary_pending');
        currModules.innerHTML = '';
        currProgress.hidden = true;
        return;
      }
      var o = state.curriculum_outline;
      currSummary.textContent = String(o.summary || '');
      currModules.innerHTML = '';
      // Lookup table by module_id so each module row can grab its
      // per-module rollup from the latest /turn response.
      var modProgress = {};
      (state.module_progress || []).forEach(function (mp) {
        modProgress[mp.module_id] = mp;
      });
      (o.modules || []).forEach(function (m) {
        var row = el('div', 'curr-module');
        row.dataset.status = String(m.status || 'not_started');
        var head = el('div', 'curr-module-head');
        head.appendChild(el('h4', 'curr-module-title', String(m.title || '')));
        head.appendChild(el('span', 'curr-module-status', String(m.status || 'not started').replace('_', ' ')));
        var mp = modProgress[m.id];
        if (mp) {
          var done = (mp.exercises_attempted || 0) + (mp.lessons_given || 0);
          var denom = mp.estimated_cards
            || ((mp.exercises_emitted || 0) + (mp.lessons_given || 0))
            || null;
          var displayDone = denom ? Math.min(done, denom) : done;
          var count = el('span', 'curr-module-count',
            denom ? (displayDone + ' / ' + denom) : t('progress.cards', { n: done }));
          head.appendChild(count);
        }
        row.appendChild(head);
        if (mp && (mp.estimated_cards || mp.exercises_emitted)) {
          var done2 = (mp.exercises_attempted || 0) + (mp.lessons_given || 0);
          var denom2 = mp.estimated_cards
            || ((mp.exercises_emitted || 0) + (mp.lessons_given || 0));
          var pct = Math.min(100, Math.round(100 * done2 / denom2));
          var bar = el('div', 'curr-module-bar');
          var fill = el('b');
          fill.style.width = pct + '%';
          bar.appendChild(fill);
          row.appendChild(bar);
        }
        if (Array.isArray(m.topics) && m.topics.length) {
          var ul = el('ul', 'curr-topics');
          m.topics.forEach(function (t) {
            var li = el('li', null, String(t.title || ''));
            li.dataset.kind = String(t.kind || 'practice');
            ul.appendChild(li);
          });
          row.appendChild(ul);
        }
        currModules.appendChild(row);
      });

      // Progress strip
      var seen = state.progress.total_seen || 0;
      var correct = state.progress.total_correct || 0;
      var pct = seen ? Math.round(100 * correct / seen) : 0;
      currProgressText.textContent = seen
        ? t('progress.cards_pct', { n: seen, pct: pct })
        : t('progress.fresh_start');
      currProgressBoxes.innerHTML = '';
      for (var b = 1; b <= 5; b++) {
        var n = (state.progress.by_box || {})[b] || 0;
        var bar = document.createElement('b');
        bar.dataset.box = String(b);
        bar.style.height = Math.max(3, Math.min(18, 3 + n * 3)) + 'px';
        bar.title = 'Box ' + b + ': ' + n + ' cards';
        currProgressBoxes.appendChild(bar);
      }
      currProgress.hidden = false;
    }

    function setActiveGoalView(goal) {
      state.goal = goal;
      if (!goal) {
        currTitleText.textContent = t('topbar.curriculum_fallback');
        return;
      }
      // Once any goal is active, the first-curriculum focus prompt is
      // moot — clear the flag so the composer doesn't mis-route the
      // next typed message into /goals/new.
      state.awaiting_focus = false;
      // Prefer the goal title; otherwise build a language-pair label
      // like "Afrikaans for English speakers" so the user always knows
      // which curriculum they're in even before they name it.
      if (goal.title) {
        currTitleText.textContent = goal.title;
      } else if (goal.language) {
        var tgt = LANGUAGE_DISPLAY[goal.language] || goal.language;
        var src = LANGUAGE_DISPLAY[goal.source_language || 'english']
          || goal.source_language || 'English';
        currTitleText.textContent = t('topbar.title_pair_template',
          { target: tgt, source: src });
      } else {
        currTitleText.textContent = t('topbar.curriculum_fallback');
      }
      // Switching to a goal in a different source language updates
      // the UI locale automatically so the learner sees prompts in
      // their strongest language.
      if (goal.source_language) setLocaleFromSource(goal.source_language);
    }

    function renderProgressStrip() {
      // Active module's progress drives the slim bar. Denominator is
      // the outline's `estimated_cards` so the learner sees the SAME
      // number they saw on the plan — cap visible fill at 100% if the
      // model emits more than estimated (the learner has overflowed
      // the plan; the bar still reads sensibly).
      if (!state.active_module_id || !state.module_progress || !state.module_progress.length) {
        progressStrip.hidden = true;
        return;
      }
      var mod = null;
      for (var i = 0; i < state.module_progress.length; i++) {
        if (state.module_progress[i].module_id === state.active_module_id) {
          mod = state.module_progress[i]; break;
        }
      }
      if (!mod) { progressStrip.hidden = true; return; }
      var done = (mod.exercises_attempted || 0) + (mod.lessons_given || 0);
      var denom = mod.estimated_cards
        || ((mod.exercises_emitted || 0) + (mod.lessons_given || 0))
        || null;
      if (!denom) {
        // No denominator at all — show just the count without a bar.
        progressStrip.hidden = false;
        progressLabel.innerHTML = '<b>' + (mod.title || 'Module') + '</b>';
        progressCount.textContent = t('progress.cards', { n: done });
        progressFill.style.width = '0%';
        return;
      }
      var pct = Math.min(100, Math.round(100 * done / denom));
      // Clamp the displayed numerator to the denominator so the
      // counter reads "6 / 6" once the target is hit instead of
      // "10 / 6". Auto-advance on the backend will switch us to the
      // next module on the next turn; until then the bar stays full.
      var doneDisplay = Math.min(done, denom);
      progressStrip.hidden = false;
      progressLabel.innerHTML = '<b>' + (mod.title || 'Module') + '</b>';
      progressCount.textContent = doneDisplay + ' / ' + denom;
      progressFill.style.width = pct + '%';
    }

    // ── Drawer ─────────────────────────────────────────────
    function renderGoalList() {
      goalList.innerHTML = '';
      var goals = state.goals || [];
      if (!goals.length) {
        var empty = el('div', null, t('drawer.none_yet'));
        empty.style.fontSize = '13px';
        empty.style.color = 'var(--ink-soft)';
        goalList.appendChild(empty);
        return;
      }
      goals.forEach(function (g) {
        var row = el('div', 'goal-row');
        row.dataset.active = (g.status === 'active') ? 'true' : 'false';
        var head = el('div', 'goal-row-head');
        head.appendChild(el('div', 'goal-row-title', g.title || 'Untitled curriculum'));
        head.appendChild(el('span', 'goal-row-status', g.status));
        row.appendChild(head);
        if (g.status !== 'active') {
          var actions = el('div', 'goal-row-actions');
          var actBtn = el('button', null, t('drawer.activate'));
          actBtn.type = 'button';
          actBtn.addEventListener('click', function () { switchGoal(g.goal_id); });
          actions.appendChild(actBtn);
          row.appendChild(actions);
        }
        goalList.appendChild(row);
      });
    }

    function openDrawer() {
      renderGoalList();
      drawerOverlay.hidden = false;
      drawer.hidden = false;
    }
    function closeDrawer() {
      drawerOverlay.hidden = true;
      drawer.hidden = true;
    }

    // ── Source-language pick (after a topic card click) ──────
    function openSourcePickModal(target) {
      modalRoot.innerHTML = '';
      var ov = el('div', 'modal-overlay');
      var mo = el('div', 'modal');
      var sources = Object.keys(LANGUAGE_DISPLAY).filter(function (s) {
        return s !== target;
      });
      var hh = document.createElement('h3');
      hh.textContent = t('modal.source_pick_heading', {
        target: LANGUAGE_DISPLAY[target],
      });
      var pp = document.createElement('p');
      pp.textContent = t('modal.source_pick_blurb');
      mo.appendChild(hh);
      mo.appendChild(pp);
      sources.forEach(function (s, i) {
        var label = document.createElement('label');
        label.className = 'source-radio';
        var input = document.createElement('input');
        input.type = 'radio'; input.name = 'srcLang'; input.value = s;
        if (i === 0) input.checked = true;
        var span = document.createElement('span');
        span.textContent = LANGUAGE_DISPLAY[s];
        label.appendChild(input);
        label.appendChild(span);
        mo.appendChild(label);
      });
      var actions = document.createElement('div');
      actions.className = 'modal-actions';
      var cancelBtn = document.createElement('button');
      cancelBtn.type = 'button'; cancelBtn.id = 'srcCancel';
      cancelBtn.textContent = t('modal.source_pick_back');
      var startBtn = document.createElement('button');
      startBtn.type = 'button'; startBtn.id = 'srcStart';
      startBtn.className = 'primary';
      startBtn.textContent = t('modal.source_pick_start');
      actions.appendChild(cancelBtn);
      actions.appendChild(startBtn);
      mo.appendChild(actions);
      ov.appendChild(mo);
      modalRoot.appendChild(ov);

      function onCancel() {
        modalRoot.innerHTML = '';
        state.pending_target = null;
        state.pending_source = null;
      }
      mo.querySelector('#srcCancel').addEventListener('click', onCancel);
      ov.addEventListener('click', function (e) {
        if (e.target === ov) onCancel();
      });
      mo.querySelector('#srcStart').addEventListener('click', async function () {
        var picked = mo.querySelector('input[name="srcLang"]:checked');
        if (!picked) return;
        state.pending_target = target;
        state.pending_source = picked.value;
        modalRoot.innerHTML = '';
        // Swap UI locale to the picked source immediately so the
        // intake bubbles + any subsequent error toasts render in the
        // learner's strongest language. Awaited so startSession sees
        // the freshly-loaded bundle.
        var nextLocale = SOURCE_TO_LOCALE[picked.value];
        if (nextLocale && nextLocale !== currentLocale) {
          await loadLocale(nextLocale);
        }
        startSession();
      });
    }

    function onTopicCardClick(target) {
      // Reset any stale resume state so a returning learner who
      // clicks a new topic gets a fresh source pick instead of
      // jumping into their last curriculum.
      state.pending_target = target;
      state.pending_source = null;
      openSourcePickModal(target);
    }

    // ── New-goal modal ─────────────────────────────────────
    function openNewGoalModal(opts) {
      modalRoot.innerHTML = '';
      opts = opts || {};
      var isInitial = !!opts.initial;   // first-curriculum / no-active-goal flow
      var heading = isInitial
        ? t('modal.new_initial_heading')
        : t('modal.new_secondary_heading');
      var blurb = isInitial
        ? t('modal.new_initial_blurb')
        : t('modal.new_secondary_blurb');
      var titlePlaceholder = isInitial
        ? t('modal.focus_placeholder_initial')
        : t('modal.focus_placeholder_secondary');
      var ov = el('div', 'modal-overlay');
      var mo = el('div', 'modal');
      // Prefer the active goal's pair; fall through to the source-pick
      // selection (state.pending_target/source) before the hard-coded
      // default — otherwise this modal would silently drop the
      // language the user just picked on the landing page.
      var defTarget = (state.goal && state.goal.language)
                      || state.pending_target
                      || 'afrikaans';
      var defSource = (state.goal && state.goal.source_language)
                      || state.pending_source
                      || 'english';

      function makeLabel(forId, text) {
        var l = document.createElement('label');
        l.htmlFor = forId;
        l.textContent = text;
        return l;
      }
      function makeSelect(id, options, selected) {
        var s = document.createElement('select');
        s.id = id;
        options.forEach(function (opt) {
          var o = document.createElement('option');
          o.value = opt.value;
          o.textContent = opt.label;
          if (opt.value === selected) o.selected = true;
          s.appendChild(o);
        });
        return s;
      }

      var hh = document.createElement('h3'); hh.textContent = heading;
      var pp = document.createElement('p'); pp.textContent = blurb;
      var titleInput = document.createElement('input');
      titleInput.id = 'newGoalTitle';
      titleInput.type = 'text';
      titleInput.maxLength = 80;
      titleInput.placeholder = titlePlaceholder;
      var ctxInput = document.createElement('textarea');
      ctxInput.id = 'newGoalContext';
      ctxInput.rows = 3;
      ctxInput.maxLength = 600;
      ctxInput.placeholder = t('modal.context_placeholder');

      var langOpts = Object.keys(LANGUAGE_DISPLAY).map(function (k) {
        return { value: k, label: LANGUAGE_DISPLAY[k] };
      });
      var targetSel = makeSelect('newGoalTarget', langOpts, defTarget);
      var sourceSel = makeSelect('newGoalSource', langOpts, defSource);
      var levelSel = makeSelect('newGoalLevel', [
        { value: 'beginner',     label: t('modal.level_beginner') },
        { value: 'elementary',   label: t('modal.level_elementary') },
        { value: 'intermediate', label: t('modal.level_intermediate') },
        { value: 'advanced',     label: t('modal.level_advanced') },
      ], 'beginner');

      var pairRow = document.createElement('div');
      pairRow.className = 'modal-pair-row';
      var pairLeft = document.createElement('div');
      pairLeft.appendChild(makeLabel('newGoalTarget', t('modal.target_label')));
      pairLeft.appendChild(targetSel);
      var pairRight = document.createElement('div');
      pairRight.appendChild(makeLabel('newGoalSource', t('modal.source_label')));
      pairRight.appendChild(sourceSel);
      pairRow.appendChild(pairLeft);
      pairRow.appendChild(pairRight);

      var actions = document.createElement('div');
      actions.className = 'modal-actions';
      var cancelBtn = document.createElement('button');
      cancelBtn.type = 'button'; cancelBtn.id = 'newGoalCancel';
      cancelBtn.textContent = isInitial ? t('modal.back') : t('modal.cancel');
      var createBtn = document.createElement('button');
      createBtn.type = 'button'; createBtn.id = 'newGoalCreate';
      createBtn.className = 'primary';
      createBtn.textContent = isInitial ? t('modal.start') : t('modal.create');
      actions.appendChild(cancelBtn);
      actions.appendChild(createBtn);

      mo.appendChild(hh);
      mo.appendChild(pp);
      mo.appendChild(makeLabel('newGoalTitle', t('modal.focus_label')));
      mo.appendChild(titleInput);
      mo.appendChild(pairRow);
      mo.appendChild(makeLabel('newGoalLevel', t('modal.level_label')));
      mo.appendChild(levelSel);
      mo.appendChild(makeLabel('newGoalContext', t('modal.context_label')));
      mo.appendChild(ctxInput);
      mo.appendChild(actions);

      ov.appendChild(mo);
      modalRoot.appendChild(ov);
      titleInput.focus();

      function onCancel() {
        modalRoot.innerHTML = '';
        if (isInitial) exitToLanding();
      }
      mo.querySelector('#newGoalCancel').addEventListener('click', onCancel);
      ov.addEventListener('click', function (e) {
        if (e.target === ov) onCancel();
      });
      createBtn.addEventListener('click', async function () {
        var title = titleInput.value.trim();
        var ctx = ctxInput.value.trim();
        if (!title) { titleInput.focus(); return; }
        if (targetSel.value === sourceSel.value) {
          showError(t('modal.pick_different'));
          return;
        }
        createBtn.disabled = true;
        try {
          var resp = await api('/goals/new', {
            learner_id: state.learner_id,
            title: title,
            context: ctx || null,
            language: targetSel.value,
            source_language: sourceSel.value,
            current_level: levelSel.value,
            activate: true,
          });
          modalRoot.innerHTML = '';
          state.goals = resp.goals || [];
          setActiveGoalView(resp.goal);
          state.curriculum_outline = null;
          state.progress = { total_seen: 0, total_correct: 0, by_box: {} };
          state.thread = [];
          rerenderThread([]);
          renderCurriculumPanel();
          closeDrawer();
          // Kick off the first /turn for the new goal so the outline
          // + first lesson land immediately.
          await sendTurn(null);
        } catch (e) {
          showError(e.message || t('errors.could_not_create_short'));
        } finally {
          createBtn.disabled = false;
        }
      });
    }

    // ── Goal operations ────────────────────────────────────
    async function switchGoal(goalId) {
      closeDrawer();
      try {
        var resp = await api('/goals/activate', {
          learner_id: state.learner_id,
          goal_id: goalId,
        });
        state.goals = resp.goals || [];
        setActiveGoalView(resp.goal);
        state.curriculum_outline = resp.curriculum_outline || null;
        state.progress = resp.progress || state.progress;
        state.thread = resp.thread || [];
        // Reset module-scoped state from the PREVIOUS goal before
        // applying the new one. Without this the slim bar kept
        // showing the previous curriculum's module + counter even
        // after switching, and lingering active_card_id could mark a
        // stale exercise from the prior goal as active during the
        // re-render.
        state.module_progress = resp.module_progress || [];
        state.active_module_id = resp.active_module_id || null;
        state.active_card_id = null;
        rerenderThread(state.thread);
        renderCurriculumPanel();
        renderProgressStrip();
      } catch (e) {
        showError(e.message || t('errors.could_not_switch'));
      }
    }

    async function restartCurrentGoal() {
      if (!state.goal) return;
      if (!confirm(t('confirm.restart', {
        title: state.goal.title || t('confirm.this_curriculum_fallback'),
      }))) return;
      closeDrawer();
      try {
        await api('/goals/restart', {
          learner_id: state.learner_id,
          goal_id: state.goal.goal_id,
        });
        state.thread = [];
        state.progress = { total_seen: 0, total_correct: 0, by_box: {} };
        state.active_card_id = null;
        // The outline is preserved across restart but progress is
        // wiped — drop module counts to zero too so the slim bar +
        // panel badges match what the user sees.
        state.module_progress = [];
        rerenderThread([]);
        renderCurriculumPanel();
        renderProgressStrip();
        await sendTurn(null);
      } catch (e) {
        showError(e.message || t('errors.could_not_restart'));
      }
    }

    async function archiveCurrentGoal() {
      if (!state.goal) return;
      if (!confirm(t('confirm.archive'))) return;
      closeDrawer();
      try {
        var resp = await api('/goals/archive', {
          learner_id: state.learner_id,
          goal_id: state.goal.goal_id,
        });
        state.goals = resp.goals || [];
        if (resp.active_goal_id) {
          // Some other goal got activated upstream (not currently
          // possible — see backend) OR there's at least a paused
          // sibling we should now activate.
          await switchGoal(resp.active_goal_id);
        } else if (state.goals.length === 0) {
          // No curriculums left — back to landing.
          exitToLanding();
        } else {
          // Paused siblings remain; pick the first one to surface.
          await switchGoal(state.goals[0].goal_id);
        }
      } catch (e) {
        showError(e.message || t('errors.could_not_delete'));
      }
    }

    async function clearAllData() {
      if (!confirm(t('confirm.clear_all'))) return;
      try {
        if (state.learner_id) await api('/clear', { learner_id: state.learner_id });
      } catch (e) {}
      localStorage.removeItem(LS_KEY);
      state = {
        learner_id: null, view: 'landing', intake_field: null,
        profile: null, goal: null, goals: [],
        curriculum_outline: null,
        progress: { total_seen: 0, total_correct: 0, by_box: {} },
        thread: [], active_card_id: null,
        module_progress: [], active_module_id: null,
        pending_target: null, pending_source: null,
        busy: false,
      };
      closeDrawer();
      exitToLanding();
    }

    // ── Turn dispatch ──────────────────────────────────────
    async function sendTurn(text, opts) {
      if (state.busy) return;
      state.busy = true;
      btnSend.disabled = true;
      composer.disabled = true;
      var skipOptimisticEcho = !!(opts && opts.optimisticLearnerText);

      var skel = appendSkeleton();
      try {
        var body = { learner_id: state.learner_id, text: text };
        if (state.goal && state.goal.goal_id) body.goal_id = state.goal.goal_id;
        var resp = await api('/turn', body);
        removeSkeleton(skel);
        state.progress = resp.progress || state.progress;
        state.curriculum_outline = resp.curriculum_outline || state.curriculum_outline;
        // Refresh the drawer's goals list each turn — otherwise the
        // first auto-created curriculum stays invisible until the user
        // manually creates or switches a goal.
        if (resp.goals) state.goals = resp.goals;
        state.module_progress = resp.module_progress || [];
        state.active_module_id = resp.active_module_id || state.active_module_id;
        if (resp.goal) setActiveGoalView(resp.goal);
        var skippedOnce = false;
        (resp.messages || []).forEach(function (m) {
          state.thread.push(m);   // persisted history for rehydration
          // The server echoes back the learner_text it just persisted.
          // When the composer rendered the bubble optimistically we
          // must NOT re-render it on the response or the user sees
          // their text twice (the screenshot bug).
          if (skipOptimisticEcho && !skippedOnce && m.kind === 'learner_text') {
            skippedOnce = true;
            return;
          }
          appendMessage(m);
        });
        renderCurriculumPanel();
        renderProgressStrip();
      } catch (e) {
        removeSkeleton(skel);
        showError(e.message || t('errors.turn_failed'));
      } finally {
        state.busy = false;
        btnSend.disabled = !composer.value.trim();
        composer.disabled = false;
        composer.focus();
      }
    }

    async function sendIntakeAnswer(value) {
      var field = state.intake_field;
      if (!field) return;
      // Show the learner bubble client-side immediately (intake
      // messages aren't persisted in learner_messages).
      appendMessage({ kind: 'learner_text', payload: { text: String(value) } });
      // Skeleton while the LLM parser interprets the reply.
      var skel = appendSkeleton();
      try {
        var resp = await api('/intake', {
          learner_id: state.learner_id, field: field, value: value,
        });
        removeSkeleton(skel);
        // The LLM intermediary either extracted a value (advance) OR
        // wrote a natural-voice follow-up question (stay on field).
        // Either way the frontend just renders the next coach bubble —
        // no more "Hmm — must be a positive integer Try that again?"
        // mechanical string. clarify_text is the authoritative copy
        // when present.
        if (resp.clarify_text) {
          appendMessage({
            kind: 'coach_text',
            payload: { text: resp.clarify_text },
          });
          return;
        }
        state.profile = resp.profile || state.profile;
        if (resp.intake_complete) {
          state.intake_field = null;
          appendMessage({
            kind: 'coach_text',
            payload: { text: t('intake.putting_plan') },
          });
          // Create the goal explicitly with the language pair the user
          // picked on the landing topic-card click flow. Falls through
          // to the legacy /turn auto-create when no pending pair —
          // matches the magic-link / cold-resume path where the pair
          // isn't supplied client-side.
          if (state.pending_target && state.pending_source) {
            try {
              var ngResp = await api('/goals/new', {
                learner_id: state.learner_id,
                title: state.profile && state.profile.objective
                  ? state.profile.objective : null,
                language: state.pending_target,
                source_language: state.pending_source,
                current_level: state.profile && state.profile.current_level
                  ? state.profile.current_level : null,
                activate: true,
              });
              state.goals = ngResp.goals || [];
              setActiveGoalView(ngResp.goal);
              state.pending_target = null;
              state.pending_source = null;
            } catch (e) {
              // /goals/new failed — DO NOT fall through to sendTurn,
              // because the backend's /turn auto-create would silently
              // spawn a default Afrikaans-from-English goal and the
              // user would land in the wrong curriculum. Show the
              // error and bail.
              state.pending_target = null;
              state.pending_source = null;
              showError(t('errors.could_not_create', { detail: e.message || '' }));
              return;
            }
          }
          // Bootstrap the first real turn — designs outline + first
          // card. From here on every message is persisted server-side.
          await sendTurn(null);
        } else {
          state.intake_field = resp.next_intake_field;
          appendIntakePrompt(resp.next_intake_prompt || '');
        }
      } catch (e) {
        removeSkeleton(skel);
        showError(e.message || t('errors.could_not_save'));
      }
    }

    // ── First-curriculum, post-intake creation ─────────────
    // Three small helpers that replace the openNewGoalModal({initial:true})
    // call. They keep the onboarding chat-first: once the source-pick
    // modal is closed and intake is already done, the user shouldn't
    // see another form — the assistant just spins up their curriculum
    // (or asks the one missing thing in chat).

    async function _createGoalAndBootstrap(title) {
      // Shared body for both auto-create and chat-focus paths. Mirrors
      // sendTurn's busy-state pattern so a spam-click or Enter-spam
      // during the in-flight /goals/new can't fire a second create.
      if (state.busy) return;
      state.busy = true;
      btnSend.disabled = true;
      composer.disabled = true;
      try {
        var ngResp = await api('/goals/new', {
          learner_id: state.learner_id,
          title: title || null,
          language: state.pending_target,
          source_language: state.pending_source,
          current_level: state.profile && state.profile.current_level
            ? state.profile.current_level : null,
          activate: true,
        });
        state.goals = ngResp.goals || [];
        setActiveGoalView(ngResp.goal);
        state.pending_target = null;
        state.pending_source = null;
        // Only release busy/composer AFTER success — sendTurn(null)
        // will re-acquire them. If we released before sendTurn, an
        // Enter-spam could fire a /turn with no goal.
        await sendTurn(null);
      } catch (e) {
        // Same defensive policy as sendIntakeAnswer: don't silently
        // fall through to /turn (the backend's auto-create would
        // spawn a default Afrikaans goal). Surface the error and
        // keep awaiting_focus armed so the next typed message
        // retries through the focus branch instead of /turn.
        state.pending_target = null;
        state.pending_source = null;
        showError(t('errors.could_not_create', { detail: e.message || '' }));
      } finally {
        state.busy = false;
        btnSend.disabled = !composer.value.trim();
        composer.disabled = false;
      }
    }

    async function autoCreateGoalFromPending() {
      // We know the language pair AND the focus (from earlier intake).
      // Show the "putting your plan together…" coach bubble and create
      // the goal silently — no modal, no extra questions. Early-return
      // if busy so a rapid-fire startSession can't append the bubble
      // twice.
      if (state.busy) return;
      appendMessage({
        kind: 'coach_text',
        payload: { text: t('intake.putting_plan') },
      });
      var title = state.profile && state.profile.objective
        ? state.profile.objective : null;
      await _createGoalAndBootstrap(title);
    }

    function askForFocusInChat() {
      // Edge case: returning learner, intake marked complete, but
      // profile.objective is empty (legacy data). Ask the one missing
      // thing in chat and route the answer to /goals/new via the
      // composer's awaiting_focus branch.
      state.awaiting_focus = true;
      var target = state.pending_target || 'language';
      var displayTarget = LANGUAGE_DISPLAY[target] || target;
      // LANGUAGE_DISPLAY values are already capitalised in en.json but
      // belt-and-braces in case a future locale routes via this code
      // path with a lowercase name.
      if (displayTarget && displayTarget.length) {
        displayTarget = displayTarget.charAt(0).toUpperCase() + displayTarget.slice(1);
      }
      var greeting = t('intake.welcome_back_short');
      var question = t('intake.focus_prompt_template', { target: displayTarget });
      appendIntakePrompt(greeting + ' ' + question);
    }

    async function createGoalWithFocus(text) {
      // Composer routed here when state.awaiting_focus is true. Note:
      // we DON'T clear awaiting_focus eagerly — if /goals/new fails
      // inside _createGoalAndBootstrap, the user's retry should stay
      // in this branch (typing again hits createGoalWithFocus, not
      // sendTurn(/turn), which would trigger the backend's auto-create
      // and spawn an unintended default Afrikaans goal).
      // Early-return if busy so a spam-Enter can't append double
      // bubbles before _createGoalAndBootstrap acquires the lock.
      if (state.busy) return;
      appendMessage({ kind: 'learner_text', payload: { text: String(text) } });
      appendMessage({
        kind: 'coach_text',
        payload: { text: t('intake.putting_plan') },
      });
      await _createGoalAndBootstrap(String(text));
      // On success, setActiveGoalView (inside _createGoalAndBootstrap)
      // has already cleared awaiting_focus. On error it's still true,
      // which is what we want for the retry path.
    }

    // ── Session bootstrap ──────────────────────────────────
    async function startSession() {
      try {
        var body = {};
        if (magicToken) body.token = magicToken;
        if (state.learner_id) body.learner_id = state.learner_id;
        var s = await api('/sessions', body);
        state.learner_id = s.learner_id;
        localStorage.setItem(LS_KEY, s.learner_id);
        state.profile = s.profile;
        state.goals = s.goals || [];
        state.curriculum_outline = s.curriculum_outline;
        state.progress = s.progress || state.progress;
        state.thread = s.thread || [];
        setActiveGoalView(s.active_goal || null);
        enterLearnView();
        renderCurriculumPanel();

        if (!s.intake_complete) {
          // Intake mode — render the welcome bubble + the first
          // question. The thread is empty pre-intake by definition.
          state.intake_field = s.next_intake_field;
          rerenderThread([]);
          appendIntakePrompt(
            t('intake.welcome_prefix') + (s.next_intake_prompt || "")
          );
        } else if (!s.active_goal) {
          // Intake done but no active curriculum. Stay chat-first:
          // if we have a pending language pair (came from the
          // source-pick modal) AND we know the learner's focus
          // (captured during intake), spin up the goal silently.
          // If the pair is missing — magic-link / cold-resume with
          // no topic-card click — fall back to the new-goal modal.
          // If we have the pair but no objective on file (legacy
          // data), ask in chat instead of opening a form.
          rerenderThread([]);
          if (state.pending_target && state.pending_source) {
            var have_objective = !!(state.profile && state.profile.objective);
            if (have_objective) {
              await autoCreateGoalFromPending();
            } else {
              askForFocusInChat();
            }
          } else {
            openNewGoalModal({ initial: true });
          }
        } else if (state.thread.length === 0) {
          // Active goal exists but no thread yet — first /turn for
          // this goal. Bootstrap.
          rerenderThread([]);
          await sendTurn(null);
        } else {
          rerenderThread(state.thread);
        }
      } catch (e) {
        showError(t('errors.could_not_start', { detail: e.message || '' }));
      }
    }

    // ── Wire-up ────────────────────────────────────────────
    topicAfrikaans.addEventListener('click', function (e) {
      e.preventDefault();
      onTopicCardClick('afrikaans');
    });
    topicEnglish.addEventListener('click', function (e) {
      e.preventDefault();
      onTopicCardClick('english');
    });
    topicGerman.addEventListener('click', function (e) {
      e.preventDefault();
      onTopicCardClick('german');
    });
    btnStartClosing.addEventListener('click', function () {
      onTopicCardClick('afrikaans');
    });

    composer.addEventListener('input', function () {
      btnSend.disabled = !composer.value.trim() || state.busy;
      // Auto-resize textarea.
      composer.style.height = 'auto';
      composer.style.height = Math.min(composer.scrollHeight, 180) + 'px';
    });
    composer.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        btnSend.click();
      }
    });
    btnSend.addEventListener('click', function () {
      var v = composer.value.trim();
      if (!v) return;
      composer.value = '';
      composer.style.height = 'auto';
      btnSend.disabled = true;
      if (state.awaiting_focus) {
        // First-curriculum focus prompt — route directly to /goals/new
        // (NOT /intake — intake is already complete by definition when
        // this branch fires) and bootstrap the first card.
        createGoalWithFocus(v);
      } else if (state.intake_field) {
        sendIntakeAnswer(v);
      } else {
        // Optimistic learner bubble — `optimisticLearnerText: true`
        // tells sendTurn to drop the server's echo of this message
        // when the response lands, so the user sees ONE bubble.
        appendMessage({ kind: 'learner_text', payload: { text: v } });
        sendTurn(v, { optimisticLearnerText: true });
      }
    });

    var _currDimmer = null;

    function openCurriculumPanel() {
      renderCurriculumPanel();
      currPanel.hidden = false;
      currTitleBtn.setAttribute('aria-expanded', 'true');
      // Click-anywhere-else-to-close dimmer overlay between thread and
      // panel. Removed when panel closes.
      if (!_currDimmer) {
        _currDimmer = document.createElement('div');
        _currDimmer.className = 'curr-panel-dimmer';
        _currDimmer.addEventListener('click', closeCurriculumPanel);
        learnSurface.insertBefore(_currDimmer, currPanel);
      }
    }
    function closeCurriculumPanel() {
      currPanel.hidden = true;
      currTitleBtn.setAttribute('aria-expanded', 'false');
      if (_currDimmer && _currDimmer.parentNode) {
        _currDimmer.parentNode.removeChild(_currDimmer);
        _currDimmer = null;
      }
    }

    currTitleBtn.addEventListener('click', function () {
      if (currPanel.hidden) openCurriculumPanel();
      else closeCurriculumPanel();
    });

    btnMenu.addEventListener('click', openDrawer);
    btnDrawerClose.addEventListener('click', closeDrawer);
    drawerOverlay.addEventListener('click', closeDrawer);

    btnNewGoal.addEventListener('click', openNewGoalModal);
    btnRestart.addEventListener('click', restartCurrentGoal);
    btnArchive.addEventListener('click', archiveCurrentGoal);
    btnClear.addEventListener('click', clearAllData);

    // Load the chosen locale BEFORE deciding whether to auto-enter
    // via magic-link — startSession can render intake bubbles that
    // need t() to be ready.
    loadLocale(detectInitialLocale());

    // Magic-link arrival → auto-enter (they came from a specific link
    // and the landing would be confusing). Otherwise default to the
    // landing — even for returning learners — so they always see the
    // marketing entrance and pick "Start with Afrikaans" deliberately.
    // The session/state is still recovered server-side when they
    // click Start, so there's no progress loss.
    if (magicToken) {
      startSession();
    }
  })();
