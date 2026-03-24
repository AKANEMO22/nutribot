const assert = require("assert");
const { createDashboardSync } = require("./streamlit_assets/embedded_fpt_build/assets/dashboard-sync.js");

function createMockElement(initialText = "") {
  return {
    textContent: initialText,
    style: { width: "" },
    _writes: 0,
  };
}

function run() {
  const totalEl = createMockElement("1790 kcal");
  const cardValueEl = createMockElement("0");
  const cardPercentEl = createMockElement("0% hoàn thành");
  const cardBarEl = createMockElement();

  const mealCard = {
    querySelector: (selector) => {
      if (selector === ".text-2xl.bg-gradient-to-r") return totalEl;
      if (selector === ".text-2xl") return totalEl;
      if (selector === ".text-xl") return { textContent: "dummy" };
      return null;
    },
  };

  const overviewCard = {
    querySelector: (selector) => {
      if (selector === ".text-4xl") return cardValueEl;
      if (selector === ".h-2 .absolute") return cardBarEl;
      if (selector === ".text-xs.text-gray-500.mt-2.text-right") return cardPercentEl;
      return null;
    },
  };

  const textarea = {
    closest: () => mealCard,
  };

  const doc = {
    body: {},
    querySelectorAll: (selector) => {
      if (selector === "textarea") return [textarea];
      if (selector === ".grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-4 > .bg-white.rounded-2xl.shadow-xl.p-6") {
        return [overviewCard];
      }
      return [];
    },
  };

  class MockMutationObserver {
    constructor(cb) {
      this.cb = cb;
      this.observeCalled = 0;
    }
    observe() {
      this.observeCalled += 1;
    }
  }

  const win = {
    setInterval: () => 1,
    MutationObserver: MockMutationObserver,
  };

  global.MutationObserver = MockMutationObserver;

  const sync = createDashboardSync(doc, win);

  sync.syncDashboard();
  const firstValue = cardValueEl.textContent;
  const firstPercent = cardPercentEl.textContent;
  const firstWidth = cardBarEl.style.width;

  sync.syncDashboard();

  assert.strictEqual(cardValueEl.textContent, firstValue, "Calories text should remain stable on repeated sync");
  assert.strictEqual(cardPercentEl.textContent, firstPercent, "Percent text should remain stable on repeated sync");
  assert.strictEqual(cardBarEl.style.width, firstWidth, "Progress width should remain stable on repeated sync");

  totalEl.textContent = "2000 kcal";
  sync.syncDashboard();

  assert.strictEqual(cardValueEl.textContent, "2.000", "Calories text should update when input changes");
  assert.strictEqual(cardPercentEl.textContent, "100% hoàn thành", "Percent should update correctly");

  console.log("feedbackloop_test: PASS");
}

run();
