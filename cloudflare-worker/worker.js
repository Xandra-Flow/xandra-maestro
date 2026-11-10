/*
Xandra Hope - Cloudflare Worker (v4.5 SIMPLIFIED)
Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud & Александра Евгеньевна Острикова
All Rights Reserved. License: XHNSL v1.0

v4.5: Simplified to fix syntax errors. No IIFE in template literals.
      All HTML built with string concatenation (no template literals in critical sections).
*/

function jsonResponse(data, status) {
  return new Response(JSON.stringify(data), {
    status: status || 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Content-Type': 'application/json'
    }
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

async function handleStats(env) {
  try {
    // Fetch state.json
    let state = {};
    try {
      const hfRes = await fetch('https://huggingface.co/smart-force/xandra-training-state/resolve/main/state.json?v=' + Date.now(), {
        headers: { 'Authorization': 'Bearer ' + env.HF_TOKEN_OWNER }
      });
      if (hfRes.status === 200) {
        state = await hfRes.json();
      }
    } catch (e) {}

    var trainingStep = state.current_step || 0;
    var targetSteps = state.target_steps || 500000;
    var currentLoss = (state.current_loss !== null && state.current_loss !== undefined) ? state.current_loss.toFixed(4) : 'N/A';
    var lastUpdated = state.updated_at ? new Date(state.updated_at).toLocaleString('en-US') : 'N/A';
    var gpuHoursUsed = state.gpu_hours_used || 0;
    var totalProjectHours = state.total_project_hours || gpuHoursUsed;
    var projectStartDate = state.project_start_date;
    var projectAgeDays = projectStartDate ? Math.ceil((Date.now() - new Date(projectStartDate).getTime()) / (1000 * 60 * 60 * 24)) : 0;
    var avgHoursPerDay = projectAgeDays > 0 ? (totalProjectHours / projectAgeDays).toFixed(1) : '0.0';
    var history = state.history || [];
    var accountUsage = state.account_usage || {};
    var kernelStatusMap = state.kernel_status_map || {};
    var lossHistory = state.loss_history || [];
    var trainingSamples = state.training_samples || [];
    var lock = state.training_lock || {};
    var lockStatus = lock.is_locked ? 'busy' : 'available';
    var lockBy = lock.locked_by || '';
    var activeAccount = '';
    if (lockBy.indexOf('maestro-') === 0) activeAccount = lockBy.substring(8);
    else if (lockBy.indexOf('kaggle-') === 0) activeAccount = lockBy.substring(7);
    else activeAccount = lockBy;

    var progressPercent = targetSteps > 0 ? ((trainingStep / targetSteps) * 100).toFixed(2) : 0;
    var totalSessions = history.length;
    var bestLoss = 'N/A';
    if (history.length > 0) {
      var minLoss = 999;
      for (var i = 0; i < history.length; i++) {
        if (history[i].final_loss && history[i].final_loss < minLoss) minLoss = history[i].final_loss;
      }
      if (minLoss < 999) bestLoss = minLoss.toFixed(4);
    }

    // Build account cards HTML
    var accountList = [];
    for (var key in accountUsage) accountList.push({name: key, data: accountUsage[key]});
    var accountsHtml = '';
    var totalWeeklyHours = 0;
    var totalRemaining = 0;
    var swarmMaxHours = accountList.length * 30;

    if (accountList.length > 0) {
      for (var k = 0; k < accountList.length; k++) {
        var a = accountList[k].data;
        var usagePercent = (a.used / 30) * 100;
        var usageColor = a.used > 20 ? '#f56565' : a.used > 10 ? '#ed8936' : '#48bb78';
        var remColor = a.remaining > 10 ? '#48bb78' : a.remaining > 2 ? '#ed8936' : '#f56565';
        var isActive = (accountList[k].name === activeAccount);
        var kEntry = kernelStatusMap[accountList[k].name] || {};
        var kStatus = kEntry.kernel_status || 'unknown';

        totalWeeklyHours += a.used;
        totalRemaining += a.remaining;

        accountsHtml += '<div class="account-card' + (isActive ? ' card-active' : '') + '">';
        accountsHtml += '<div class="account-header"><strong>' + escapeHtml(accountList[k].name) + '</strong>';
        accountsHtml += '<span class="kernel-badge k-' + kStatus + '">' + kStatus + '</span></div>';
        accountsHtml += '<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:' + usagePercent + '%;background:' + usageColor + ';"></div></div>';
        accountsHtml += '<div class="account-info">Used: <strong>' + a.used.toFixed(1) + 'h</strong> / 30h';
        accountsHtml += '<span style="float:right;color:' + remColor + ';">Remaining: <strong>' + a.remaining.toFixed(1) + 'h</strong></span></div>';
        accountsHtml += '</div>';
      }
    } else {
      accountsHtml = '<p style="color:#666;text-align:center;">No accounts registered yet</p>';
    }

    // Build loss curve HTML
    var lossChartHtml = '';
    if (lossHistory.length === 0) {
      lossChartHtml = '<p style="color:#666;text-align:center;line-height:180px;">No loss data yet (training just started)</p>';
    } else {
      var maxLoss = -Infinity, minLoss = Infinity;
      for (var i = 0; i < lossHistory.length; i++) {
        if (lossHistory[i].loss > maxLoss) maxLoss = lossHistory[i].loss;
        if (lossHistory[i].loss < minLoss) minLoss = lossHistory[i].loss;
      }
      var range = maxLoss - minLoss || 1;
      var pointsArr = [];
      for (var i = 0; i < lossHistory.length; i++) {
        var x = (i / (lossHistory.length - 1 || 1)) * 100;
        var y = 100 - ((lossHistory[i].loss - minLoss) / range) * 100;
        pointsArr.push(x.toFixed(1) + ',' + y.toFixed(1));
      }
      var points = pointsArr.join(' ');
      var latestLoss = lossHistory[lossHistory.length - 1].loss;
      var firstLoss = lossHistory[0].loss;
      var improvement = firstLoss > latestLoss ? (((firstLoss - latestLoss) / firstLoss) * 100).toFixed(1) : '0.0';
      lossChartHtml = '<div style="font-size:0.85em;color:#666;margin-bottom:5px;">';
      lossChartHtml += 'Start: <strong>' + firstLoss.toFixed(4) + '</strong> &rarr; ';
      lossChartHtml += 'Current: <strong>' + latestLoss.toFixed(4) + '</strong> ';
      lossChartHtml += '(<span style="color:#48bb78;">&#9660; ' + improvement + '%</span>)</div>';
      lossChartHtml += '<svg width="100%" height="150" viewBox="0 0 100 100" preserveAspectRatio="none" style="background:#fff;border:1px solid #e2e8f0;border-radius:5px;">';
      lossChartHtml += '<polyline points="' + points + '" fill="none" stroke="#48bb78" stroke-width="0.5" vector-effect="non-scaling-stroke" />';
      lossChartHtml += '</svg>';
      lossChartHtml += '<div style="font-size:0.75em;color:#a0aec0;text-align:right;margin-top:3px;">' + lossHistory.length + ' data points</div>';
    }

    // Build training samples HTML
    var samplesHtml = '';
    if (trainingSamples.length === 0) {
      samplesHtml = '<p style="color:#666;text-align:center;padding:30px;">No training samples logged yet</p>';
    } else {
      var recent = trainingSamples.slice(-10).reverse();
      for (var i = 0; i < recent.length; i++) {
        var s = recent[i];
        var preview = (s.text || '').substring(0, 150);
        var ellipsis = (s.text && s.text.length > 150) ? '...' : '';
        var source = s.source || 'unknown';
        var time = s.timestamp ? new Date(s.timestamp).toLocaleString('en-US') : '';
        var num = trainingSamples.length - i;
        samplesHtml += '<div style="background:white;padding:10px;border-radius:5px;margin-bottom:8px;border-right:3px solid #667eea;">';
        samplesHtml += '<div style="font-size:0.8em;color:#667eea;font-weight:bold;">#' + num + ' &bull; ' + escapeHtml(source) + ' &bull; ' + time + '</div>';
        samplesHtml += '<div style="margin-top:5px;font-size:0.9em;color:#4a5568;">' + escapeHtml(preview) + ellipsis + '</div>';
        samplesHtml += '<div style="font-size:0.75em;color:#a0aec0;margin-top:3px;">Step ' + (s.step || '?') + ' &bull; loss: ' + (s.loss || 'N/A') + '</div>';
        samplesHtml += '</div>';
      }
    }

    // Build session info
    var sessionStart = lock.locked_at ? new Date(lock.locked_at).toLocaleString('en-US') : 'N/A';
    var sessionEnd = 'N/A';
    if (history.length > 0) {
      var lastSession = history[history.length - 1];
      sessionEnd = lastSession.timestamp ? new Date(lastSession.timestamp).toLocaleString('en-US') : 'N/A';
    }

    // Assemble the full HTML
    var html = '';
    html += '<!DOCTYPE html>\n<html lang="en" dir="ltr">\n<head>\n';
    html += '<meta charset="UTF-8">\n';
    html += '<meta name="viewport" content="width=device-width,initial-scale=1.0">\n';
    html += '<title>Xandra Hope Dashboard</title>\n';
    html += '<style>\n';
    html += '*{margin:0;padding:0;box-sizing:border-box}\n';
    html += 'body{font-family:Segoe UI,Tahoma,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;color:#333;padding:20px}\n';
    html += 'body.rtl{direction:rtl}\n';
    html += '.container{max-width:960px;margin:0 auto}\n';
    html += 'header{text-align:center;background:white;padding:30px 20px;border-radius:20px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.1)}\n';
    html += 'header h1{font-size:2.5em;color:#4a3a8a}\n';
    html += '.lang-switcher{display:flex;gap:10px;justify-content:center;margin-bottom:20px}\n';
    html += '.lang-btn{background:white;border:none;padding:5px 15px;border-radius:20px;cursor:pointer;font-weight:bold;color:#667eea;box-shadow:0 2px 5px rgba(0,0,0,0.1)}\n';
    html += '.lang-btn.active{background:#667eea;color:white}\n';
    html += '.btn-group{display:flex;gap:15px;justify-content:center;margin-bottom:20px;flex-wrap:wrap}\n';
    html += '.btn{color:white;border:none;padding:10px 25px;border-radius:10px;font-weight:bold;cursor:pointer;font-size:1.1em}\n';
    html += '.btn-refresh{background:#667eea}\n';
    html += '.btn-force{background:#ed8936}\n';
    html += '.btn:disabled{opacity:0.6;cursor:not-allowed}\n';
    html += '.section-card{background:white;padding:30px;border-radius:15px;margin-bottom:20px;box-shadow:0 5px 20px rgba(0,0,0,0.08)}\n';
    html += '.section-card h2{color:#4a3a8a;margin-bottom:20px;font-size:1.4em}\n';
    html += '.progress-bar-bg{background:#e2e8f0;border-radius:20px;overflow:hidden;height:30px;margin:15px 0}\n';
    html += '.progress-bar-fill{background:linear-gradient(90deg,#48bb78,#38a169);height:100%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;transition:width 1s ease}\n';
    html += '.progress-info{display:flex;justify-content:space-between;margin-top:10px;color:#4a5568;font-size:0.95em}\n';
    html += '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:30px}\n';
    html += '.card{background:white;padding:25px;border-radius:15px;text-align:center;box-shadow:0 5px 20px rgba(0,0,0,0.08)}\n';
    html += '.card .number{font-size:2.2em;font-weight:bold;color:#667eea}\n';
    html += '.card .label{color:#666;margin-top:5px;font-size:0.9em}\n';
    html += '.account-card{background:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:15px;border:1px solid #e2e8f0}\n';
    html += '.account-card.card-active{border-color:#48bb78;box-shadow:0 0 0 2px rgba(72,187,120,0.3);background:#f0fff4}\n';
    html += '.account-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}\n';
    html += '.kernel-badge{padding:4px 10px;border-radius:12px;font-size:0.8em;font-weight:bold;text-transform:uppercase}\n';
    html += '.kernel-badge.k-running{background:#c6f6d5;color:#22543d;animation:pulse 2s infinite}\n';
    html += '.kernel-badge.k-complete{background:#bee3f8;color:#2a4365}\n';
    html += '.kernel-badge.k-error{background:#fed7d7;color:#742a2a}\n';
    html += '.kernel-badge.k-unknown{background:#e2e8f0;color:#4a5568}\n';
    html += '.account-info{font-size:0.9em;color:#4a5568}\n';
    html += '.session-info{display:flex;justify-content:space-between;margin-top:15px;padding:15px;background:#f7fafc;border-radius:10px;font-size:0.9em}\n';
    html += '.footer{text-align:center;color:white;padding:30px}\n';
    html += '.project-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}\n';
    html += '.stat-row{background:#f7fafc;padding:10px;border-radius:8px;border-right:3px solid #667eea;font-size:0.95em}\n';
    html += '@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.6}}\n';
    html += '</style>\n';
    html += '</head>\n<body>\n';
    html += '<div class="container">\n';

    // Header
    html += '<header>\n';
    html += '<h1><span data-ml="title">Xandra Hope Dashboard</span></h1>\n';
    html += '<p data-ml="subtitle">The Sovereign Arabic AI Model</p>\n';
    html += '<p style="font-size:0.7em;color:#a0aec0;margin-top:5px;">v4.5 Simplified</p>\n';
    html += '</header>\n';

    // Language switcher
    html += '<div class="lang-switcher">\n';
    html += '<button class="lang-btn active" onclick="setLang(\'en\', event)">EN</button>\n';
    html += '<button class="lang-btn" onclick="setLang(\'ar\', event)">AR</button>\n';
    html += '<button class="lang-btn" onclick="setLang(\'ru\', event)">RU</button>\n';
    html += '</div>\n';

    // Buttons
    html += '<div class="btn-group">\n';
    html += '<button class="btn btn-refresh" onclick="refreshPage()" data-ml="refresh">Refresh</button>\n';
    html += '<button class="btn btn-force" onclick="triggerUpdate()" data-ml="force_update">Force Update</button>\n';
    html += '</div>\n';

    // Training progress
    html += '<div class="section-card">\n';
    html += '<h2 data-ml="training_progress">Training Progress</h2>\n';
    html += '<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:' + progressPercent + '%;">' + progressPercent + '%</div></div>\n';
    html += '<div class="progress-info"><span data-ml="step">Step</span>: <strong>' + trainingStep.toLocaleString() + '</strong></div>\n';
    html += '<div class="progress-info"><span data-ml="target">Target</span>: <strong>' + targetSteps.toLocaleString() + '</strong></div>\n';
    html += '<div class="progress-info"><span data-ml="loss">Loss</span>: <strong>' + currentLoss + '</strong></div>\n';
    html += '<div class="progress-info"><span data-ml="last_update">Last Update</span>: <strong>' + lastUpdated + '</strong></div>\n';
    html += '</div>\n';

    // Stats grid
    html += '<div class="grid">\n';
    html += '<div class="card"><div class="number" style="font-size:1.3em;" data-ml="' + lockStatus + '">' + lockStatus + '</div><div class="label">' + (activeAccount ? 'By: ' + activeAccount : '') + '</div></div>\n';
    html += '<div class="card"><div class="number">' + gpuHoursUsed.toFixed(1) + '</div><div class="label" data-ml="total_gpu">Total GPU Hours</div></div>\n';
    html += '<div class="card"><div class="number">' + totalSessions + '</div><div class="label">Total Sessions</div></div>\n';
    html += '</div>\n';

    // Swarm status
    html += '<div class="section-card">\n';
    html += '<h2 data-ml="swarm_status">Swarm Status</h2>\n';
    html += '<div class="progress-info"><span data-ml="weekly_available">Weekly Available</span>: <strong>' + swarmMaxHours + 'h</strong></div>\n';
    html += '<div class="progress-info"><span data-ml="weekly_used">Weekly Used</span>: <strong style="color:#f56565;">' + totalWeeklyHours.toFixed(1) + 'h</strong></div>\n';
    html += '<div class="progress-info"><span data-ml="remaining">Remaining</span>: <strong style="color:#48bb78;">' + totalRemaining.toFixed(1) + 'h</strong></div>\n';
    html += '<div style="margin-top:20px;">' + accountsHtml + '</div>\n';
    html += '</div>\n';

    // Project stats
    html += '<div class="section-card">\n';
    html += '<h2 data-ml="project_stats">Project Statistics (All-Time)</h2>\n';
    html += '<div class="project-stats">\n';
    html += '<div class="stat-row"><span data-ml="project_age">Project Age</span>: <strong>' + projectAgeDays + ' days</strong></div>\n';
    html += '<div class="stat-row"><span data-ml="total_project_hours">Total Training Hours</span>: <strong style="color:#48bb78;font-size:1.2em;">' + totalProjectHours.toFixed(1) + 'h</strong></div>\n';
    html += '<div class="stat-row"><span data-ml="avg_hours_day">Avg Hours/Day</span>: <strong>' + avgHoursPerDay + 'h</strong></div>\n';
    html += '<div class="stat-row"><span data-ml="best_loss">Best Loss</span>: <strong>' + bestLoss + '</strong></div>\n';
    html += '</div>\n';
    html += '</div>\n';

    // Loss curve
    html += '<div class="section-card">\n';
    html += '<h2 data-ml="loss_curve">Performance Evolution (Loss Curve)</h2>\n';
    html += '<div style="width:100%;min-height:200px;background:#f7fafc;border-radius:10px;padding:10px;">\n';
    html += lossChartHtml;
    html += '</div>\n';
    html += '</div>\n';

    // Training samples
    html += '<div class="section-card">\n';
    html += '<h2 data-ml="training_samples">What the Model is Learning</h2>\n';
    html += '<div style="max-height:300px;overflow-y:auto;background:#f7fafc;border-radius:10px;padding:10px;">\n';
    html += samplesHtml;
    html += '</div>\n';
    html += '</div>\n';

    // Session info
    html += '<div class="section-card">\n';
    html += '<h2 data-ml="sessions_info">Sessions Info</h2>\n';
    html += '<div class="session-info"><div><strong data-ml="current_start">Current Session Start</strong>:<br>' + sessionStart + '</div><div><strong data-ml="last_end">Last Session End</strong>:<br>' + sessionEnd + '</div></div>\n';
    html += '</div>\n';

    html += '<div class="footer"><p>(c) 2026 Xandra Flow - All Rights Reserved</p></div>\n';
    html += '</div>\n';

    // Script — using string concatenation, NO template literals
    html += '<script>\n';
    html += 'var translations = {\n';
    html += '  en: { title:"Xandra Hope Dashboard", subtitle:"The Sovereign Arabic AI Model", refresh:"Refresh", force_update:"Force Update",\n';
    html += '        training_progress:"Training Progress", step:"Step", target:"Target", loss:"Loss", last_update:"Last Update",\n';
    html += '        available:"Available", busy:"Busy", total_gpu:"Total GPU Hours", total_sessions:"Total Sessions",\n';
    html += '        swarm_status:"Swarm Status", weekly_available:"Weekly Available", weekly_used:"Weekly Used", remaining:"Remaining",\n';
    html += '        used:"Used", sessions_info:"Sessions Info", current_start:"Current Session Start", last_end:"Last Session End",\n';
    html += '        running:"Running", exhausted:"Exhausted",\n';
    html += '        project_stats:"Project Statistics (All-Time)", project_age:"Project Age", total_project_hours:"Total Training Hours",\n';
    html += '        avg_hours_day:"Avg Hours/Day", best_loss:"Best Loss",\n';
    html += '        loss_curve:"Performance Evolution (Loss Curve)", training_samples:"What the Model is Learning" },\n';
    html += '  ar: { title:"Xandra Hope Dashboard", subtitle:"The Sovereign Arabic AI Model", refresh:"Refresh", force_update:"Force Update",\n';
    html += '        training_progress:"Training Progress", step:"Step", target:"Target", loss:"Loss", last_update:"Last Update",\n';
    html += '        available:"Available", busy:"Busy", total_gpu:"Total GPU Hours", total_sessions:"Total Sessions",\n';
    html += '        swarm_status:"Swarm Status", weekly_available:"Weekly Available", weekly_used:"Weekly Used", remaining:"Remaining",\n';
    html += '        used:"Used", sessions_info:"Sessions Info", current_start:"Current Session Start", last_end:"Last Session End",\n';
    html += '        running:"Running", exhausted:"Exhausted",\n';
    html += '        project_stats:"Project Statistics (All-Time)", project_age:"Project Age", total_project_hours:"Total Training Hours",\n';
    html += '        avg_hours_day:"Avg Hours/Day", best_loss:"Best Loss",\n';
    html += '        loss_curve:"Performance Evolution (Loss Curve)", training_samples:"What the Model is Learning" },\n';
    html += '  ru: { title:"Xandra Hope Dashboard", subtitle:"The Sovereign Arabic AI Model", refresh:"Refresh", force_update:"Force Update",\n';
    html += '        training_progress:"Training Progress", step:"Step", target:"Target", loss:"Loss", last_update:"Last Update",\n';
    html += '        available:"Available", busy:"Busy", total_gpu:"Total GPU Hours", total_sessions:"Total Sessions",\n';
    html += '        swarm_status:"Swarm Status", weekly_available:"Weekly Available", weekly_used:"Weekly Used", remaining:"Remaining",\n';
    html += '        used:"Used", sessions_info:"Sessions Info", current_start:"Current Session Start", last_end:"Last Session End",\n';
    html += '        running:"Running", exhausted:"Exhausted",\n';
    html += '        project_stats:"Project Statistics (All-Time)", project_age:"Project Age", total_project_hours:"Total Training Hours",\n';
    html += '        avg_hours_day:"Avg Hours/Day", best_loss:"Best Loss",\n';
    html += '        loss_curve:"Performance Evolution (Loss Curve)", training_samples:"What the Model is Learning" }\n';
    html += '};\n';
    html += 'function setLang(lang, event) {\n';
    html += '  var dict = translations[lang] || translations.en;\n';
    html += '  document.querySelectorAll("[data-ml]").forEach(function(el) {\n';
    html += '    var key = el.getAttribute("data-ml");\n';
    html += '    if (dict[key]) el.innerText = dict[key];\n';
    html += '  });\n';
    html += '  document.body.className = lang === "ar" ? "rtl" : "ltr";\n';
    html += '  document.querySelectorAll(".lang-btn").forEach(function(btn) { btn.classList.remove("active"); });\n';
    html += '  if (event && event.target) event.target.classList.add("active");\n';
    html += '}\n';
    html += 'function showToast(message, color) {\n';
    html += '  var existing = document.getElementById("xandra-toast");\n';
    html += '  if (existing) existing.remove();\n';
    html += '  var toast = document.createElement("div");\n';
    html += '  toast.id = "xandra-toast";\n';
    html += '  toast.style.cssText = "position:fixed;top:20px;right:20px;background:" + (color || "#667eea") + ";color:white;padding:12px 20px;border-radius:8px;box-shadow:0 4px 15px rgba(0,0,0,0.2);z-index:9999;font-family:Arial;font-size:14px;";\n';
    html += '  toast.textContent = message;\n';
    html += '  document.body.appendChild(toast);\n';
    html += '  setTimeout(function() {\n';
    html += '    if (toast && toast.parentNode) {\n';
    html += '      toast.style.opacity = "0";\n';
    html += '      toast.style.transition = "opacity 0.5s";\n';
    html += '      setTimeout(function(){ if (toast.parentNode) toast.remove(); }, 500);\n';
    html += '    }\n';
    html += '  }, 3000);\n';
    html += '}\n';
    html += 'function refreshPage() {\n';
    html += '  var btn = document.querySelector(".btn-refresh");\n';
    html += '  if (btn) { btn.disabled = true; btn.textContent = "Loading..."; }\n';
    html += '  showToast("Refreshing...", "#667eea");\n';
    html += '  setTimeout(function(){ window.location.reload(); }, 300);\n';
    html += '}\n';
    html += 'function triggerUpdate() {\n';
    html += '  var btn = document.querySelector(".btn-force");\n';
    html += '  var originalText = btn ? btn.textContent : "";\n';
    html += '  if (btn) { btn.disabled = true; btn.textContent = "Sending..."; }\n';
    html += '  showToast("Sending request...", "#667eea");\n';
    html += '  fetch("/api/force-update", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" } })\n';
    html += '    .then(function(r) { return r.json(); })\n';
    html += '    .then(function(d) {\n';
    html += '      if (d.success) {\n';
    html += '        showToast("Success!", "#48bb78");\n';
    html += '        setTimeout(function(){ window.location.reload(); }, 1500);\n';
    html += '      } else {\n';
    html += '        var msg = "Failed!\\n";\n';
    html += '        if (d.error) msg += "Error: " + d.error + "\\n";\n';
    html += '        if (d.detail) msg += "Detail: " + d.detail + "\\n";\n';
    html += '        showToast("Failed!", "#f56565");\n';
    html += '        alert(msg);\n';
    html += '        if (btn) { btn.disabled = false; btn.textContent = originalText; }\n';
    html += '      }\n';
    html += '    })\n';
    html += '    .catch(function(e) {\n';
    html += '      showToast("Network error", "#f56565");\n';
    html += '      alert("Network error: " + e.message);\n';
    html += '      if (btn) { btn.disabled = false; btn.textContent = originalText; }\n';
    html += '    });\n';
    html += '}\n';
    html += 'setTimeout(function(){ window.location.reload(); }, 60000);\n';
    html += '</script>\n';
    html += '</body>\n</html>';

    return new Response(html, {
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });
  } catch (error) {
    return jsonResponse({ error: error.message }, 500);
  }
}

async function handleForceUpdate(env) {
  try {
    const listRes = await fetch('https://api.github.com/repos/' + env.GITHUB_REPO + '/actions/workflows', {
      method: 'GET',
      headers: { 'Authorization': 'token ' + env.GH_PAT, 'Accept': 'application/vnd.github.v3+json' }
    });

    if (listRes.status === 200) {
      const listData = await listRes.json();
      const workflows = listData.workflows || [];
      for (const w of workflows) {
        if (w.state !== 'active') continue;
        const fileName = w.path.split('/').pop();
        const ghRes = await fetch('https://api.github.com/repos/' + env.GITHUB_REPO + '/actions/workflows/' + w.id + '/dispatches', {
          method: 'POST',
          headers: { 'Authorization': 'token ' + env.GH_PAT, 'Accept': 'application/vnd.github.v3+json' },
          body: JSON.stringify({ ref: 'main' })
        });
        if (ghRes.status === 204) {
          return jsonResponse({ success: true, message: 'Triggered via ' + fileName, workflow: fileName });
        }
      }
    }

    const fallbackFiles = ['maestro.yml', 'maestro_multi.yml', 'main.yml'];
    for (const fileName of fallbackFiles) {
      const ghRes = await fetch('https://api.github.com/repos/' + env.GITHUB_REPO + '/actions/workflows/' + fileName + '/dispatches', {
        method: 'POST',
        headers: { 'Authorization': 'token ' + env.GH_PAT, 'Accept': 'application/vnd.github.v3+json' },
        body: JSON.stringify({ ref: 'main' })
      });
      if (ghRes.status === 204) {
        return jsonResponse({ success: true, message: 'Triggered via ' + fileName, workflow: fileName });
      }
    }

    return jsonResponse({ success: false, error: 'Could not trigger any workflow' }, 500);
  } catch (error) {
    return jsonResponse({ success: false, error: error.message }, 500);
  }
}

export default {
  async fetch(request, env) {
    var url = new URL(request.url);
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }
      });
    }
    if (request.method === 'GET' && (url.pathname === '/' || url.pathname === '/stats')) {
      var authHeader = request.headers.get('Authorization');
      var expectedAuth = 'Basic ' + btoa(env.ADMIN_USER + ':' + env.ADMIN_PASS);
      if (authHeader !== expectedAuth) {
        return new Response('Unauthorized', {
          status: 401,
          headers: { 'WWW-Authenticate': 'Basic realm="Xandra Dashboard"' }
        });
      }
      return handleStats(env);
    }
    if (url.pathname === '/api/force-update') {
      if (request.method !== 'POST') {
        return jsonResponse({ success: false, error: 'Use POST' }, 405);
      }
      var authHeaderPost = request.headers.get('Authorization');
      var expectedAuthPost = 'Basic ' + btoa(env.ADMIN_USER + ':' + env.ADMIN_PASS);
      if (authHeaderPost !== expectedAuthPost) {
        return jsonResponse({ success: false, error: 'Unauthorized' }, 401);
      }
      return handleForceUpdate(env);
    }
    return jsonResponse({ success: false, error: 'Not Found' }, 404);
  }
};
