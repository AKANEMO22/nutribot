((globalScope) => {
  const DAILY_TARGET = 2000;
  const METRIC_TARGETS = {
    calories: 2000,
    protein: 100,
    carbs: 250,
    fat: 65,
  };

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

  function createDashboardSync(doc = globalScope.document, win = globalScope.window) {
    let isApplying = false;
    let lastAppliedSignature = null;

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

      const totalCalories = readMealTotalCalories();
      if (totalCalories === null) {
        return;
      }

      const mealItems = readAllMealItems();
      const signature = `${totalCalories}|${mealItems.join("|")}`;

      if (lastAppliedSignature === signature) {
        return;
      }

      const nutritionTotals = calculateNutrition(mealItems, totalCalories);

      isApplying = true;
      try {
        const changed = updateOverviewCards(nutritionTotals);
        if (changed) {
          lastAppliedSignature = signature;
        }
      } finally {
        isApplying = false;
      }
    }

    function startSync() {
      syncDashboard();

      const observer = new MutationObserver(() => {
        syncDashboard();
      });

      observer.observe(doc.body, {
        childList: true,
        subtree: true,
        characterData: true,
      });

      const intervalId = win.setInterval(syncDashboard, 1200);
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
