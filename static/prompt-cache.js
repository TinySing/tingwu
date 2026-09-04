(function () {
  var KEY = 'tingwu.meeting_custom_prompt';
  var el = document.querySelector('textarea[name="custom_prompt"]');
  if (!el) return;

  // 服务端有回填（例如提交失败）时优先用服务端值，否则用上次本地缓存
  if (!el.value.trim()) {
    try {
      var cached = localStorage.getItem(KEY);
      if (cached) el.value = cached;
    } catch (e) {}
  }

  function save() {
    try {
      if (el.value.trim()) localStorage.setItem(KEY, el.value);
    } catch (e) {}
  }

  el.addEventListener('input', save);
  el.addEventListener('change', save);
  var form = el.closest('form');
  if (form) form.addEventListener('submit', save);
})();

(function () {
  var fields = [
    { id: 'pt-task-id', key: 'tingwu.prompt_tuner.task_id' },
    { id: 'pt-goal', key: 'tingwu.prompt_tuner.goal' },
    { id: 'pt-initial', key: 'tingwu.prompt_tuner.initial' },
    { id: 'pt-rounds', key: 'tingwu.prompt_tuner.rounds' },
    { id: 'pt-feedback', key: 'tingwu.prompt_tuner.feedback' },
  ];

  fields.forEach(function (item) {
    var el = document.getElementById(item.id);
    if (!el) return;
    try {
      var cached = localStorage.getItem(item.key);
      if (cached != null && cached !== '' && !el.value.trim()) el.value = cached;
    } catch (e) {}

    function save() {
      try {
        localStorage.setItem(item.key, el.value);
      } catch (e) {}
    }
    el.addEventListener('input', save);
    el.addEventListener('change', save);
  });
})();
