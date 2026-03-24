const assert = require("assert");
const { createDashboardSync } = require("./streamlit_assets/embedded_fpt_build/assets/dashboard-sync.js");

function createMockElement(initialText = "") {
  return {
    textContent: initialText,
    style: {
      width: "",
      setProperty(prop, value) {
        this[prop] = value;
      },
    },
    children: [],
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
    _writes: 0,
  };
}

function run() {
  const totalEl = createMockElement("1790 kcal");
  const cardValueEl = createMockElement("0");
  const cardPercentEl = createMockElement("0% hoàn thành");
  const cardBarEl = createMockElement();
  cardBarEl.classList = {
    contains: () => false,
  };

  const progressTrack = createMockElement();
  progressTrack.querySelector = (selector) => {
    if (selector === ".dashboard-sync-fill") return cardBarEl;
    return null;
  };
  progressTrack.querySelectorAll = () => [];

  const mealCard = {
    querySelector: (selector) => {
      if (selector === ".text-2xl.bg-gradient-to-r") return totalEl;
      if (selector === ".text-2xl") return totalEl;
      if (selector === ".text-xl") return { textContent: "dummy" };
      if (selector === ".space-y-4") return null;
      if (selector === ".dashboard-sync-hourly-warning") return null;
      return null;
    },
    querySelectorAll: (selector) => {
      if (selector === ".text-sm.text-gray-600.truncate") return [];
      return [];
    },
    appendChild: () => null,
  };

  const overviewCard = {
    querySelector: (selector) => {
      if (selector === ".text-4xl") return cardValueEl;
      if (selector === ".relative.w-full.h-2.bg-gray-100.rounded-full.overflow-hidden") return progressTrack;
      if (selector === ".text-xs.text-gray-500.mt-2.text-right") return cardPercentEl;
      if (selector === "p") return { textContent: "Calories hom nay" };
      return null;
    },
  };

  const textarea = {
    closest: () => mealCard,
  };

  const doc = {
    body: {},
    createElement: () => createMockElement(),
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
    localStorage: {
      _store: {},
      getItem(key) {
        return this._store[key] || null;
      },
      setItem(key, value) {
        this._store[key] = value;
      },
    },
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
