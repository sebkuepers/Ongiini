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
    };

    var LANGUAGE_DISPLAY = {
      afrikaans: 'Afrikaans',
      english: 'English',
      german: 'German',
    };

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
      wrap.appendChild(el('div', 'lesson-eyebrow', 'Lesson'));
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
      var btn = el('button', null, 'Got it →');
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
      var eyebrow = (p.card_type || 'card') + (isActive ? ' · awaiting your answer' : ' · answered');
      wrap.appendChild(el('div', 'exercise-eyebrow', eyebrow));
      if (p.prompt_text) wrap.appendChild(el('p', 'exercise-prompt', String(p.prompt_text)));
      if (p.hint_text) {
        var hintRow = el('div', 'exercise-hint-row');
        var hintBtn = document.createElement('button');
        hintBtn.type = 'button';
        hintBtn.className = 'hint-btn';
        hintBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M12 18h.01M9 21h6M12 3a6 6 0 0 0-3.5 10.9c.5.4.9 1 1.1 1.6L10 18h4l.4-2.5c.2-.6.6-1.2 1.1-1.6A6 6 0 0 0 12 3Z"/></svg>' +
          ' Show hint';
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
      chip.appendChild(document.createTextNode(seen + ' cards · ' + pct + '% right · box ' + (p.box || 1)));
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
        currSummary.textContent = 'Your plan is still being put together.';
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
          var count = el('span', 'curr-module-count',
            denom ? (done + ' / ' + denom) : (done + ' cards'));
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
      currProgressText.textContent = seen ? (seen + ' cards · ' + pct + '% right') : 'fresh start';
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
        currTitleText.textContent = 'Curriculum';
        return;
      }
      // Prefer the goal title; otherwise build a language-pair label
      // like "Afrikaans for English speakers" so the user always knows
      // which curriculum they're in even before they name it.
      if (goal.title) {
        currTitleText.textContent = goal.title;
      } else if (goal.language) {
        var tgt = LANGUAGE_DISPLAY[goal.language] || goal.language;
        var src = LANGUAGE_DISPLAY[goal.source_language || 'english']
          || goal.source_language || 'English';
        currTitleText.textContent = tgt + ' for ' + src + ' speakers';
      } else {
        currTitleText.textContent = 'Curriculum';
      }
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
        progressCount.textContent = done + ' cards';
        progressFill.style.width = '0%';
        return;
      }
      var pct = Math.min(100, Math.round(100 * done / denom));
      progressStrip.hidden = false;
      progressLabel.innerHTML = '<b>' + (mod.title || 'Module') + '</b>';
      progressCount.textContent = done + ' / ' + denom;
      progressFill.style.width = pct + '%';
    }

    // ── Drawer ─────────────────────────────────────────────
    function renderGoalList() {
      goalList.innerHTML = '';
      var goals = state.goals || [];
      if (!goals.length) {
        var empty = el('div', null, 'No curriculums yet.');
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
          var actBtn = el('button', null, 'Activate');
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
      var radios = sources.map(function (s, i) {
        return (
          '<label style="display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid var(--rule);border-radius:10px;cursor:pointer;margin-bottom:8px;">' +
          '<input type="radio" name="srcLang" value="' + s + '" ' + (i === 0 ? 'checked' : '') + ' />' +
          '<span>' + LANGUAGE_DISPLAY[s] + '</span>' +
          '</label>'
        );
      }).join('');
      mo.innerHTML =
        '<h3>You picked ' + LANGUAGE_DISPLAY[target] + '.</h3>' +
        '<p>Which language do you already speak well? The coach will use it for explanations and to translate from.</p>' +
        radios +
        '<div class="modal-actions">' +
        '<button type="button" id="srcCancel">Back</button>' +
        '<button type="button" class="primary" id="srcStart">Start</button>' +
        '</div>';
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
      mo.querySelector('#srcStart').addEventListener('click', function () {
        var picked = mo.querySelector('input[name="srcLang"]:checked');
        if (!picked) return;
        state.pending_target = target;
        state.pending_source = picked.value;
        modalRoot.innerHTML = '';
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
        ? 'What do you want to focus on?'
        : 'New curriculum';
      var blurb = isInitial
        ? "Tell me what you actually want to be able to do in Afrikaans. " +
          "I'll write a plan around that and start the first lesson."
        : "What do you want to focus on? Give it a short name; the model writes the plan around it.";
      var titlePlaceholder = isInitial
        ? 'e.g. Job interview at SPAR'
        : 'e.g. Interview at SPAR';
      var ov = el('div', 'modal-overlay');
      var mo = el('div', 'modal');
      var defTarget = (state.goal && state.goal.language) || 'afrikaans';
      var defSource = (state.goal && state.goal.source_language) || 'english';

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
      ctxInput.placeholder =
        'One sentence about why — the coach uses this to shape the cards.';

      var langOpts = Object.keys(LANGUAGE_DISPLAY).map(function (k) {
        return { value: k, label: LANGUAGE_DISPLAY[k] };
      });
      var targetSel = makeSelect('newGoalTarget', langOpts, defTarget);
      var sourceSel = makeSelect('newGoalSource', langOpts, defSource);
      var levelSel = makeSelect('newGoalLevel', [
        { value: 'beginner', label: 'Beginner' },
        { value: 'elementary', label: 'Elementary' },
        { value: 'intermediate', label: 'Intermediate' },
        { value: 'advanced', label: 'Advanced' },
      ], 'beginner');

      var pairRow = document.createElement('div');
      pairRow.style.display = 'flex'; pairRow.style.gap = '10px';
      var pairLeft = document.createElement('div'); pairLeft.style.flex = '1';
      pairLeft.appendChild(makeLabel('newGoalTarget', 'Language to learn'));
      pairLeft.appendChild(targetSel);
      var pairRight = document.createElement('div'); pairRight.style.flex = '1';
      pairRight.appendChild(makeLabel('newGoalSource', 'From'));
      pairRight.appendChild(sourceSel);
      pairRow.appendChild(pairLeft);
      pairRow.appendChild(pairRight);

      var actions = document.createElement('div');
      actions.className = 'modal-actions';
      var cancelBtn = document.createElement('button');
      cancelBtn.type = 'button'; cancelBtn.id = 'newGoalCancel';
      cancelBtn.textContent = isInitial ? 'Back' : 'Cancel';
      var createBtn = document.createElement('button');
      createBtn.type = 'button'; createBtn.id = 'newGoalCreate';
      createBtn.className = 'primary';
      createBtn.textContent = isInitial ? 'Start' : 'Create';
      actions.appendChild(cancelBtn);
      actions.appendChild(createBtn);

      mo.appendChild(hh);
      mo.appendChild(pp);
      mo.appendChild(makeLabel('newGoalTitle', 'Focus'));
      mo.appendChild(titleInput);
      mo.appendChild(pairRow);
      mo.appendChild(makeLabel('newGoalLevel', 'Your level'));
      mo.appendChild(levelSel);
      mo.appendChild(makeLabel('newGoalContext', 'Context (optional)'));
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
          showError("Pick a different source and target language.");
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
          showError(e.message || 'Could not create curriculum.');
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
        showError(e.message || 'Could not switch curriculum.');
      }
    }

    async function restartCurrentGoal() {
      if (!state.goal) return;
      if (!confirm('This wipes the cards and chat history for "' + (state.goal.title || 'this curriculum') + '" — the plan stays. Continue?')) return;
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
        showError(e.message || 'Could not restart.');
      }
    }

    async function archiveCurrentGoal() {
      if (!state.goal) return;
      if (!confirm('Delete this curriculum? You can always start a new one.')) return;
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
        showError(e.message || 'Could not delete curriculum.');
      }
    }

    async function clearAllData() {
      if (!confirm('This deletes everything we have stored about your learning. Continue?')) return;
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
        showError(e.message || 'Turn failed — try again.');
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
            payload: { text: "Perfect. Putting your plan together now…" },
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
              showError("Could not create curriculum: " + (e.message || ''));
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
        showError(e.message || 'Could not save that.');
      }
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
            "Welcome — let's set you up. " + (s.next_intake_prompt || "")
          );
        } else if (!s.active_goal) {
          // Intake done but no active curriculum — the learner either
          // archived all their goals and came back, or this is their
          // very first /sessions after intake. ASK what they want to
          // learn instead of silently spawning a generic Afrikaans
          // goal (the previous bug: get_or_create_active_goal would
          // re-seed from profile.objective and the result felt like
          // 'the old curriculum came back').
          rerenderThread([]);
          openNewGoalModal({ initial: true });
        } else if (state.thread.length === 0) {
          // Active goal exists but no thread yet — first /turn for
          // this goal. Bootstrap.
          rerenderThread([]);
          await sendTurn(null);
        } else {
          rerenderThread(state.thread);
        }
      } catch (e) {
        showError("Couldn't start: " + (e.message || ''));
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
      if (state.intake_field) {
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
