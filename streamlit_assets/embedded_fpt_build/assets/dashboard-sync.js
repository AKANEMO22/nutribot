((globalScope) => {
  const DAILY_TARGET = 2000;
  const METRIC_TARGETS = {
    calories: 2000,
    protein: 100,
    carbs: 250,
    fat: 65,
  };

  const HOURLY_WARNING_LIMITS = {
    calories: 900,
    protein: 60,
    carbs: 120,
    fat: 35,
  };

  const INTAKE_HISTORY_KEY = "nutribot_intake_history_v1";
  const HISTORY_RETENTION_MS = 24 * 60 * 60 * 1000;
  const ONE_HOUR_MS = 60 * 60 * 1000;

  const FOOD_NUTRITION_DB = {
    "banh mi trung": { calories: 280, protein: 12, carbs: 30, fat: 12 },
    "sua dau nanh": { calories: 120, protein: 8, carbs: 10, fat: 4 },
    "com ga": { calories: 450, protein: 32, carbs: 52, fat: 11 },
    "rau xao": { calories: 140, protein: 3, carbs: 10, fat: 8 },
    canh: { calories: 90, protein: 4, carbs: 9, fat: 3 },
    tao: { calories: 95, protein: 0.5, carbs: 25, fat: 0.3 },
    "hat hanh nhan": { calories: 170, protein: 6, carbs: 6, fat: 15 },
    "pho bo": { calories: 420, protein: 24, carbs: 45, fat: 16 },
    "cha gio": { calories: 160, protein: 6, carbs: 12, fat: 10 },
    "com tam": { calories: 520, protein: 25, carbs: 62, fat: 17 },
    "bun bo": { calories: 480, protein: 24, carbs: 55, fat: 16 },
    "bun cha": { calories: 520, protein: 26, carbs: 54, fat: 21 },
    "bun rieu": { calories: 430, protein: 22, carbs: 48, fat: 14 },
    "mi xao": { calories: 510, protein: 14, carbs: 63, fat: 20 },
    "goi cuon": { calories: 90, protein: 5, carbs: 10, fat: 2 },
    "ca hoi": { calories: 350, protein: 34, carbs: 0, fat: 22 },
    "uc ga": { calories: 250, protein: 35, carbs: 0, fat: 9 },
    sua: { calories: 110, protein: 6, carbs: 9, fat: 5 },
    chuoi: { calories: 105, protein: 1.3, carbs: 27, fat: 0.3 },
    trung: { calories: 78, protein: 6.3, carbs: 0.6, fat: 5.3 },
    salad: { calories: 150, protein: 4, carbs: 12, fat: 9 },
    "sua chua": { calories: 100, protein: 5, carbs: 13, fat: 3 },
  };

  const CARD_CONFIG = {
    calories: {
      labelMatchers: ["calories hom nay", "calories hôm nay"],
      color: "linear-gradient(to right, #ff7a18, #ff3d3d)",
      formatter: (value) => formatKcal(value),
      target: METRIC_TARGETS.calories,
    },
    protein: {
      labelMatchers: ["protein"],
      color: "linear-gradient(to right, #3b82f6, #06b6d4)",
      formatter: (value) => String(Math.round(value)),
      target: METRIC_TARGETS.protein,
    },
    carbs: {
      labelMatchers: ["carbs"],
      color: "linear-gradient(to right, #22c55e, #10b981)",
      formatter: (value) => String(Math.round(value)),
      target: METRIC_TARGETS.carbs,
    },
    fat: {
      labelMatchers: ["chat beo", "chất béo"],
      color: "linear-gradient(to right, #a855f7, #ec4899)",
      formatter: (value) => String(Math.round(value)),
      target: METRIC_TARGETS.fat,
    },
  };

  function parseKcal(text) {
    const digits = (text || "").replace(/[^\d]/g, "");
    return digits ? Number(digits) : 0;
  }

  function formatKcal(value) {
    return Math.round(value).toLocaleString("vi-VN");
  }

  function normalizeFoodName(text) {
    return (text || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/đ/g, "d")
      .replace(/Đ/g, "D")
      .toLowerCase()
      .trim()
      .replace(/\s+/g, " ");
  }

  function splitFoodInput(text) {
    return (text || "")
      .split(/[;,\n]/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function zeroTotals() {
    return {
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
    };
  }

  function sanitizeTotals(totals) {
    const base = zeroTotals();
    if (!totals || typeof totals !== "object") {
      return base;
    }

    Object.keys(base).forEach((k) => {
      const v = Number(totals[k]);
      base[k] = Number.isFinite(v) ? v : 0;
    });

    return base;
  }

  function diffPositive(nextTotals, prevTotals) {
    const out = zeroTotals();
    Object.keys(out).forEach((k) => {
      out[k] = Math.max(0, Number(nextTotals[k] || 0) - Number(prevTotals[k] || 0));
    });
    return out;
  }

  function addTotals(acc, inc) {
    Object.keys(acc).forEach((k) => {
      acc[k] += Number(inc[k] || 0);
    });
    return acc;
  }

  function createDashboardSync(doc = globalScope.document, win = globalScope.window) {
    let isApplying = false;
    let lastAppliedSignature = null;
    const pendingLocalAnswers = [];
    let localRequestInFlight = false;
    let syncScheduled = false;
    let lastSyncTs = 0;

    function getGreenChatInput() {
      if (!doc || typeof doc.querySelector !== "function") {
        return null;
      }
      return doc.querySelector('input[placeholder*="Nhập câu hỏi về dinh dưỡng"]');
    }

    function getGreenChatSendButton(inputEl) {
      if (!inputEl) {
        return null;
      }

      let node = inputEl;
      for (let i = 0; i < 5 && node; i += 1) {
        const buttons = Array.from(node.parentElement ? node.parentElement.querySelectorAll("button") : []);
        const sendBtn = buttons.find((btn) => (btn.textContent || "").trim().includes("Gửi"));
        if (sendBtn) {
          return sendBtn;
        }
        node = node.parentElement;
      }
      return null;
    }

    function getGreenChatMessageList(inputEl) {
      if (!inputEl || typeof inputEl.closest !== "function") {
        return null;
      }
      return inputEl
        .closest(".h-screen")
        ?.querySelector(".flex-1.overflow-y-auto.p-6 .max-w-4xl.mx-auto.space-y-6");
    }

    function getBotRows(messageList) {
      if (!messageList) {
        return [];
      }

      return Array.from(messageList.querySelectorAll("div.flex.gap-3"))
        .filter((row) => {
          const cls = String(row.className || "");
          if (cls.includes("flex-row-reverse")) {
            return false;
          }
          return !!row.querySelector("p.whitespace-pre-line.leading-relaxed");
        });
    }

    async function sendPortalFeedback(question, answer, rating, upBtn, downBtn) {
      try {
        await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source: "green_portal_chat",
            question,
            answer,
            rating,
          }),
        });

        upBtn.disabled = true;
        downBtn.disabled = true;
        upBtn.style.opacity = "0.5";
        downBtn.style.opacity = "0.5";
      } catch (error) {
        // Ignore feedback send failures to keep UI responsive.
      }
    }

    function attachFeedbackButtons(botRow, question, answer) {
      if (!botRow || botRow.querySelector(".nutribot-feedback-row")) {
        return;
      }

      const contentWrap = Array.from(botRow.querySelectorAll("div"))
        .find((el) => {
          const cls = String(el.className || "");
          return cls.includes("flex-col") && cls.includes("max-w-[70%]");
        });
      if (!contentWrap) {
        return;
      }

      const row = doc.createElement("div");
      row.className = "nutribot-feedback-row flex gap-2 mt-1";

      const upBtn = doc.createElement("button");
      upBtn.textContent = "👍";
      upBtn.className = "px-3 py-1 bg-white border border-green-200 text-green-700 rounded-full text-xs";

      const downBtn = doc.createElement("button");
      downBtn.textContent = "👎";
      downBtn.className = "px-3 py-1 bg-white border border-green-200 text-green-700 rounded-full text-xs";

      upBtn.addEventListener("click", () => sendPortalFeedback(question, answer, "up", upBtn, downBtn));
      downBtn.addEventListener("click", () => sendPortalFeedback(question, answer, "down", upBtn, downBtn));

      row.appendChild(upBtn);
      row.appendChild(downBtn);
      contentWrap.appendChild(row);
    }

    function tryApplyPendingAnswer(inputEl) {
      if (!pendingLocalAnswers.length) {
        return;
      }

      const now = Date.now();
      while (pendingLocalAnswers.length && now - pendingLocalAnswers[0].ts > 120000) {
        pendingLocalAnswers.shift();
      }
      if (!pendingLocalAnswers.length) {
        return;
      }

      const messageList = getGreenChatMessageList(inputEl);
      const botRows = getBotRows(messageList);
      if (!botRows.length) {
        return;
      }

      const pending = pendingLocalAnswers[0];
      const ageMs = now - pending.ts;
      const hasExpectedNewBotRow = botRows.length > pending.baselineBotCount;
      const canForcePatchLatest = ageMs > 1500;
      if (!hasExpectedNewBotRow && !canForcePatchLatest) {
        return;
      }

      const latestBotRow = botRows[botRows.length - 1];
      const textNode = latestBotRow.querySelector("p.whitespace-pre-line.leading-relaxed");
      if (!textNode) {
        return;
      }

      textNode.textContent = pending.answer;
      latestBotRow.setAttribute("data-local-ai-patched", "1");
      attachFeedbackButtons(latestBotRow, pending.question, pending.answer);
      pendingLocalAnswers.shift();
      localRequestInFlight = false;
    }

    async function requestPortalLocalAnswer(question, baselineBotCount, inputEl) {
      let answer = "Không có phản hồi từ AI local.";
      try {
        const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
        const timeoutId = controller
          ? win.setTimeout(() => controller.abort(), 45000)
          : null;

        const response = await fetch("/api/local-chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: question }),
          signal: controller ? controller.signal : undefined,
        });

        if (timeoutId) {
          win.clearTimeout(timeoutId);
        }

        const data = await response.json();
        if (data && data.answer) {
          answer = String(data.answer);
        }
      } catch (error) {
        answer = "AI local phản hồi chậm hoặc lỗi kết nối. Vui lòng thử lại với câu hỏi ngắn hơn.";
      }

      pendingLocalAnswers.push({ question, answer, baselineBotCount, ts: Date.now() });
      tryApplyPendingAnswer(inputEl);

      // Ensure input is not locked forever if UI structure changed and patch did not apply.
      win.setTimeout(() => {
        tryApplyPendingAnswer(inputEl);
        if (localRequestInFlight) {
          localRequestInFlight = false;
        }
      }, 2200);
    }

    function handlePortalSubmit(inputEl) {
      const question = ((inputEl?.value || "").trim() || (inputEl?.getAttribute("data-local-draft") || "").trim());
      if (!question) {
        return;
      }

      const now = Date.now();
      const lastQuestion = inputEl.getAttribute("data-local-last-question") || "";
      const lastTs = Number(inputEl.getAttribute("data-local-last-ts") || "0");
      if (lastQuestion === question && now - lastTs < 800) {
        return;
      }

      if (localRequestInFlight) {
        return;
      }

      inputEl.setAttribute("data-local-last-question", question);
      inputEl.setAttribute("data-local-last-ts", String(now));
      localRequestInFlight = true;

      const baselineBotCount = getBotRows(getGreenChatMessageList(inputEl)).length;
      requestPortalLocalAnswer(question, baselineBotCount, inputEl);
    }

    function wirePortalChatToLocalAI() {
      const inputEl = getGreenChatInput();
      if (!inputEl) {
        return;
      }

      if (!inputEl.getAttribute("data-local-ai-bound")) {
        inputEl.setAttribute("data-local-ai-bound", "1");
        inputEl.addEventListener("input", () => {
          inputEl.setAttribute("data-local-draft", inputEl.value || "");
        });
        inputEl.addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            handlePortalSubmit(inputEl);
          }
        }, true);
      }

      const sendBtn = getGreenChatSendButton(inputEl);
      if (sendBtn && !sendBtn.getAttribute("data-local-ai-bound")) {
        sendBtn.setAttribute("data-local-ai-bound", "1");
        sendBtn.addEventListener("click", () => {
          handlePortalSubmit(inputEl);
        }, true);
      }

      if (doc.body && !doc.body.getAttribute("data-local-ai-click-bound")) {
        doc.body.setAttribute("data-local-ai-click-bound", "1");
        doc.body.addEventListener("click", (event) => {
          const target = event && event.target;
          if (!target || typeof target.closest !== "function") {
            return;
          }
          const btn = target.closest("button");
          if (!btn) {
            return;
          }
          const label = (btn.textContent || "").trim();
          if (!label.includes("Gửi")) {
            return;
          }
          const activeInput = getGreenChatInput();
          if (activeInput) {
            handlePortalSubmit(activeInput);
          }
        }, true);
      }

      tryApplyPendingAnswer(inputEl);
    }

    function loadHistory() {
      if (!win || !win.localStorage) {
        return [];
      }

      try {
        const raw = win.localStorage.getItem(INTAKE_HISTORY_KEY);
        if (!raw) {
          return [];
        }

        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
          return [];
        }

        return parsed
          .filter((item) => item && Number.isFinite(item.ts) && item.totals)
          .map((item) => ({
            ts: Number(item.ts),
            signature: String(item.signature || ""),
            totals: sanitizeTotals(item.totals),
          }));
      } catch (error) {
        return [];
      }
    }

    function saveHistory(history) {
      if (!win || !win.localStorage) {
        return;
      }

      try {
        win.localStorage.setItem(INTAKE_HISTORY_KEY, JSON.stringify(history));
      } catch (error) {
        // Ignore localStorage quota and serialization errors.
      }
    }

    function updateHistoryAndGetHourlyIntake(currentTotals, signature, nowMs) {
      const cutoffRetention = nowMs - HISTORY_RETENTION_MS;
      const cutoffHour = nowMs - ONE_HOUR_MS;

      let history = loadHistory().filter((item) => item.ts >= cutoffRetention);
      const latest = history.length ? history[history.length - 1] : null;

      if (!latest || latest.signature !== signature) {
        history.push({
          ts: nowMs,
          signature,
          totals: sanitizeTotals(currentTotals),
        });
      }

      if (history.length > 500) {
        history = history.slice(-500);
      }
      saveHistory(history);

      const hourly = zeroTotals();
      for (let i = 1; i < history.length; i += 1) {
        const prev = history[i - 1];
        const next = history[i];
        if (next.ts < cutoffHour) {
          continue;
        }
        addTotals(hourly, diffPositive(next.totals, prev.totals));
      }

      return hourly;
    }

    function getOrCreateWarningBox(mealCard) {
      if (!mealCard) {
        return null;
      }

      let box = mealCard.querySelector(".dashboard-sync-hourly-warning");
      if (box) {
        return box;
      }

      box = doc.createElement("div");
      box.className = "dashboard-sync-hourly-warning";
      box.style.display = "none";
      box.style.marginTop = "10px";
      box.style.padding = "10px 12px";
      box.style.borderRadius = "10px";
      box.style.background = "#FFF4E5";
      box.style.color = "#9A3412";
      box.style.border = "1px solid #FED7AA";
      box.style.fontSize = "13px";
      box.style.lineHeight = "1.45";

      const anchor =
        mealCard.querySelector(".space-y-4") ||
        mealCard.querySelector(".text-2xl.bg-gradient-to-r") ||
        mealCard;

      if (anchor && anchor.parentNode && typeof anchor.parentNode.insertBefore === "function") {
        if (anchor.nextSibling) {
          anchor.parentNode.insertBefore(box, anchor.nextSibling);
        } else {
          anchor.parentNode.appendChild(box);
        }
      } else if (typeof mealCard.appendChild === "function") {
        mealCard.appendChild(box);
      }

      return box;
    }

    function renderHourlyWarning(mealCard, hourlyTotals) {
      const warningBox = getOrCreateWarningBox(mealCard);
      if (!warningBox) {
        return;
      }

      const exceeded = [];
      Object.keys(HOURLY_WARNING_LIMITS).forEach((metric) => {
        if (Number(hourlyTotals[metric] || 0) >= HOURLY_WARNING_LIMITS[metric]) {
          exceeded.push(metric);
        }
      });

      if (!exceeded.length) {
        warningBox.style.display = "none";
        warningBox.textContent = "";
        return;
      }

      const metricNames = {
        calories: "Calories",
        protein: "Protein",
        carbs: "Carbs",
        fat: "Chất béo",
      };

      const detail = exceeded
        .map((metric) => {
          const value = Math.round(Number(hourlyTotals[metric] || 0));
          const unit = metric === "calories" ? "kcal" : "g";
          return `${metricNames[metric]}: ${value}${unit}`;
        })
        .join(" | ");

      warningBox.style.display = "block";
      warningBox.textContent = `⚠ Cảnh báo: Trong 1 giờ gần nhất, bạn đã nạp cao ở các chỉ số: ${detail}`;
    }

    function getMealCardRoot() {
      const textareas = Array.from(doc.querySelectorAll("textarea"));
      for (const textarea of textareas) {
        const root = textarea.closest(".bg-white.rounded-2xl.shadow-xl.p-6");
        if (!root) {
          continue;
        }
        if (root.querySelector(".text-2xl") && root.querySelector(".text-xl")) {
          return root;
        }
      }
      return null;
    }

    function readMealTotalCalories() {
      const mealCard = getMealCardRoot();
      if (!mealCard) {
        return null;
      }

      // Chọn chính xác class chứa text giá trị tổng cộng (chữ màu xanh gradient)
      const totalValue = mealCard.querySelector(".text-2xl.bg-gradient-to-r");
      if (!totalValue) {
        return null;
      }

      return parseKcal(totalValue.textContent);
    }

    function readAllMealItems() {
      const mealCard = getMealCardRoot();
      if (!mealCard) {
        return [];
      }

      const itemRows = Array.from(mealCard.querySelectorAll(".text-sm.text-gray-600.truncate"));
      if (!itemRows.length) {
        return [];
      }

      return itemRows.flatMap((row) => splitFoodInput(row.textContent));
    }

    function calculateNutrition(items, fallbackCalories) {
      const totals = {
        calories: 0,
        protein: 0,
        carbs: 0,
        fat: 0,
      };

      if (!items.length) {
        totals.calories = fallbackCalories;
        return totals;
      }

      items.forEach((item) => {
        const key = normalizeFoodName(item);
        const nutrition = FOOD_NUTRITION_DB[key];
        if (!nutrition) {
          return;
        }
        totals.calories += nutrition.calories;
        totals.protein += nutrition.protein;
        totals.carbs += nutrition.carbs;
        totals.fat += nutrition.fat;
      });

      if (fallbackCalories !== null && fallbackCalories !== undefined) {
        totals.calories = fallbackCalories;
      }

      return totals;
    }

    function getMetricCards() {
      return Array.from(
        doc.querySelectorAll(".grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-4 > .bg-white.rounded-2xl.shadow-xl.p-6")
      );
    }

    function findMetricCard(metricCards, metricKey) {
      const config = CARD_CONFIG[metricKey];
      if (!config) {
        return null;
      }

      return (
        metricCards.find((card) => {
          const titleEl = card.querySelector("p");
          const titleText = normalizeFoodName(titleEl ? titleEl.textContent : card.textContent);
          return config.labelMatchers.some((matcher) => titleText.includes(normalizeFoodName(matcher)));
        }) || null
      );
    }

    function setTextIfChanged(el, nextText) {
      if (!el) {
        return false;
      }
      if (el.textContent === nextText) {
        return false;
      }
      el.textContent = nextText;
      return true;
    }

    function setWidthIfChanged(el, nextWidth) {
      if (!el) {
        return false;
      }
      if (el.style.width === nextWidth) {
        return false;
      }
      el.style.setProperty("position", "absolute");
      el.style.setProperty("top", "0");
      el.style.setProperty("left", "0");
      el.style.setProperty("height", "100%");
      el.style.setProperty("width", nextWidth, "important");
      el.style.setProperty("transition", "none", "important");
      el.style.setProperty("transform", "none", "important");
      el.style.setProperty("animation", "none", "important");
      el.style.setProperty("pointer-events", "none");
      el.style.setProperty("z-index", "3");
      el.style.setProperty("border-radius", "9999px");
      el.style.setProperty("opacity", "1", "important");
      return true;
    }

    function getProgressTrack(card) {
      return card.querySelector(".relative.w-full.h-2.bg-gray-100.rounded-full.overflow-hidden");
    }

    function getOrCreateSyncFill(track, color) {
      if (!track) {
        return null;
      }

      // Hide React/Framer native fill to avoid visual conflict with synced fill.
      const nativeFills = track.querySelectorAll(".absolute.top-0.left-0.h-full");
      nativeFills.forEach((el) => {
        if (el.classList.contains("dashboard-sync-fill")) {
          return;
        }
        el.style.setProperty("opacity", "0", "important");
        el.style.setProperty("width", "0%", "important");
        el.style.setProperty("transition", "none", "important");
        el.style.setProperty("transform", "none", "important");
        el.style.setProperty("animation", "none", "important");
      });

      let fill = track.querySelector(".dashboard-sync-fill");
      if (fill) {
        fill.style.setProperty("background", color);
        return fill;
      }

      fill = doc.createElement("div");
      fill.className = "dashboard-sync-fill absolute top-0 left-0 h-full rounded-full";
      fill.style.background = color;
      track.appendChild(fill);
      return fill;
    }

    function updateMetricCard(card, metricKey, value) {
      if (!card || !CARD_CONFIG[metricKey]) {
        return false;
      }

      const config = CARD_CONFIG[metricKey];
      const ratio = Math.max(0, Math.min(100, (value / config.target) * 100));
      const valueEl = card.querySelector(".text-4xl");
      const percentText = card.querySelector(".text-xs.text-gray-500.mt-2.text-right");
      const track = getProgressTrack(card);
      const progressFill = getOrCreateSyncFill(track, config.color);

      return [
        setTextIfChanged(valueEl, config.formatter(value)),
        setWidthIfChanged(progressFill, `${ratio}%`),
        setTextIfChanged(percentText, `${Math.round(ratio)}% hoàn thành`),
      ].some(Boolean);
    }

    function updateOverviewCards(nutritionTotals) {
      const metricCards = getMetricCards();
      if (!metricCards.length) {
        return false;
      }

      const caloriesCard = findMetricCard(metricCards, "calories");
      const proteinCard = findMetricCard(metricCards, "protein");
      const carbsCard = findMetricCard(metricCards, "carbs");
      const fatCard = findMetricCard(metricCards, "fat");

      return [
        updateMetricCard(caloriesCard, "calories", nutritionTotals.calories),
        updateMetricCard(proteinCard, "protein", nutritionTotals.protein),
        updateMetricCard(carbsCard, "carbs", nutritionTotals.carbs),
        updateMetricCard(fatCard, "fat", nutritionTotals.fat),
      ].some(Boolean);
    }

    function syncDashboard() {
      if (isApplying) {
        return;
      }

      wirePortalChatToLocalAI();

      const totalCalories = readMealTotalCalories();
      const mealCard = getMealCardRoot();
      if (totalCalories === null) {
        return;
      }

      const mealItems = readAllMealItems();
      const signature = `${totalCalories}|${mealItems.join("|")}`;

      if (lastAppliedSignature === signature) {
        return;
      }

      const nutritionTotals = calculateNutrition(mealItems, totalCalories);
      const hourlyTotals = updateHistoryAndGetHourlyIntake(nutritionTotals, signature, Date.now());

      isApplying = true;
      try {
        const changed = updateOverviewCards(nutritionTotals);
        renderHourlyWarning(mealCard, hourlyTotals);
        if (changed) {
          lastAppliedSignature = signature;
        }
      } finally {
        isApplying = false;
      }
    }

    function scheduleSync() {
      if (syncScheduled) {
        return;
      }

      const now = Date.now();
      const elapsed = now - lastSyncTs;
      const minGapMs = 250;
      const delay = elapsed >= minGapMs ? 0 : (minGapMs - elapsed);
      syncScheduled = true;

      win.setTimeout(() => {
        syncScheduled = false;
        if (typeof doc?.visibilityState === "string" && doc.visibilityState === "hidden") {
          return;
        }
        lastSyncTs = Date.now();
        syncDashboard();
      }, delay);
    }

    function startSync() {
      scheduleSync();

      const observer = new MutationObserver(() => {
        scheduleSync();
      });

      observer.observe(doc.body, {
        childList: true,
        subtree: true,
      });

      const intervalId = win.setInterval(scheduleSync, 2500);
      return { observer, intervalId };
    }

    return {
      syncDashboard,
      startSync,
      _test: {
        updateOverviewCards,
        readMealTotalCalories,
      },
    };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { createDashboardSync };
  }

  if (typeof window !== "undefined" && typeof document !== "undefined") {
    const syncInstance = createDashboardSync(document, window);
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => syncInstance.startSync(), { once: true });
    } else {
      syncInstance.startSync();
    }
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
