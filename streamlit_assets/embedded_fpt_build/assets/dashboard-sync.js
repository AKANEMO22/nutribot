((globalScope) => {
  const DAILY_TARGET = 2000;

  function parseKcal(text) {
    const digits = (text || "").replace(/[^\d]/g, "");
    return digits ? Number(digits) : 0;
  }

  function formatKcal(value) {
    return Math.round(value).toLocaleString("vi-VN");
  }

  function createDashboardSync(doc = globalScope.document, win = globalScope.window) {
    let isApplying = false;
    let lastAppliedCalories = null;

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

      const totalValue = mealCard.querySelector(".text-2xl");
      if (!totalValue) {
        return null;
      }

      return parseKcal(totalValue.textContent);
    }

    function getOverviewCaloriesCard() {
      const metricCards = Array.from(
        doc.querySelectorAll(".grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-4 > .bg-white.rounded-2xl.shadow-xl.p-6")
      );

      if (!metricCards.length) {
        return null;
      }

      return metricCards[0];
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
      el.style.width = nextWidth;
      return true;
    }

    function updateOverviewCard(calories) {
      const card = getOverviewCaloriesCard();
      if (!card) {
        return;
      }

      const valueEl = card.querySelector(".text-4xl");
      const ratio = Math.max(0, Math.min(100, (calories / DAILY_TARGET) * 100));
      const progressFill = card.querySelector(".h-2 .absolute");
      const percentText = card.querySelector(".text-xs.text-gray-500.mt-2.text-right");

      const changed = [
        setTextIfChanged(valueEl, formatKcal(calories)),
        setWidthIfChanged(progressFill, `${ratio}%`),
        setTextIfChanged(percentText, `${Math.round(ratio)}% hoàn thành`),
      ].some(Boolean);

      if (changed) {
        lastAppliedCalories = calories;
      }
    }

    function syncDashboard() {
      if (isApplying) {
        return;
      }

      const totalCalories = readMealTotalCalories();
      if (totalCalories === null) {
        return;
      }

      if (lastAppliedCalories === totalCalories) {
        return;
      }

      isApplying = true;
      try {
        updateOverviewCard(totalCalories);
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
        updateOverviewCard,
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
