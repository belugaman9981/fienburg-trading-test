// Headless smoke test: loads index.html in jsdom and drives the real UI.
// Run:  npm install && npm test
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(e.message));
const dom = new JSDOM(html, { url: "http://localhost/", runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc });
const w = dom.window, d = w.document;
const $ = (s) => d.querySelector(s);
const click = (el) => el.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
const row = (sym) => [...d.querySelectorAll("#wl .row")].find((r) => r.dataset.s === sym);
const setType = (t) => { $("#otype").value = t; $("#otype").dispatchEvent(new w.Event("change")); };
const order = (side, type, qty, lim) => {
  click($(`#side button[data-s=${side}]`)); setType(type);
  $("#qty").value = qty; if (lim !== undefined) $("#lim").value = lim;
  click($("#go")); return $("#msg").textContent;
};
const stopTimers = () => { /* the sim interval keeps the process alive; we exit explicitly */ };

let failed = 0;
function test(name, fn) {
  try { fn(); console.log("  ok  ", name); } catch (e) { failed++; console.log("  FAIL", name, "\n      ", e.message); }
}

setTimeout(() => {
  console.log("PaperTrade smoke tests");
  w.clearInterval(w.iv); // we drive time ourselves via step()

  test("page boots with 13 symbols and $100k", () => {
    assert.strictEqual(d.querySelectorAll("#wl .row").length, 13);
    assert.ok($("#stats").textContent.includes("$100,000.00"));
  });

  test("market buy then position shows up", () => {
    click(row("AAPL"));
    assert.strictEqual(order("buy", "market", 10), "Order placed.");
    assert.ok($("#tbl").textContent.includes("AAPL"));
  });

  test("cannot oversell / short", () => {
    assert.match(order("sell", "market", 999), /only have 10/);
  });

  test("limit buy rests, reserves buying power, then fills when price drops", () => {
    click(row("MSFT"));
    const bpBefore = w.bp();
    assert.strictEqual(order("buy", "limit", 5, 300), "Order placed.");
    assert.ok(Math.abs(bpBefore - w.bp() - 1500) < 1e-6, "buying power should drop by 5*300");
    w.st.px.MSFT = 295; w.step();
    assert.strictEqual(w.held("MSFT"), 5);
  });

  test("stop validation: sell stop must be below market, buy stop above", () => {
    click(row("AAPL"));
    assert.match(order("sell", "stop", 1, w.st.px.AAPL * 1.05), /below the current price/);
    assert.match(order("buy", "stop", 1, w.st.px.AAPL * 0.95), /above the current price/);
  });

  test("sell stop stays open, then triggers when price falls through it", () => {
    const stop = +(w.st.px.AAPL * 0.97).toFixed(2);
    assert.strictEqual(order("sell", "stop", 10, stop), "Order placed.");
    const o = w.st.ord[w.st.ord.length - 1];
    assert.strictEqual(o.status, "open");
    w.st.px.AAPL = stop * 0.9; w.step();
    assert.strictEqual(o.status, "filled");
    assert.strictEqual(w.held("AAPL"), 0);
  });

  test("buy stop reserves cash and fills on breakout", () => {
    click(row("NVDA"));
    const stop = +(w.st.px.NVDA * 1.03).toFixed(2), bp0 = w.bp();
    assert.strictEqual(order("buy", "stop", 3, stop), "Order placed.");
    assert.ok(Math.abs(bp0 - w.bp() - 3 * stop) < 1e-6);
    w.st.px.NVDA = stop * 1.02; w.step();
    assert.strictEqual(w.held("NVDA"), 3);
  });

  test("cancelling an order releases reserved cash", () => {
    click(row("TSLA"));
    const bp0 = w.bp();
    order("buy", "limit", 2, w.st.px.TSLA * 0.5);
    assert.ok(w.bp() < bp0);
    click($("#tabs button[data-t=ord]")); click($("#tbl [data-c]"));
    assert.ok(Math.abs(w.bp() - bp0) < 1e-6);
  });

  test("fractional BTC/USD with max button", () => {
    click(row("BTC/USD")); click($("#side button[data-s=buy]")); setType("market");
    click($("#max")); assert.ok(parseFloat($("#qty").value) > 0);
    click($("#go")); assert.ok(w.held("BTC/USD") > 0);
  });

  test("Close button liquidates a position", () => {
    click($("#tabs button[data-t=pos]"));
    const btn = [...d.querySelectorAll("#tbl [data-x]")].find((b) => b.dataset.x === "BTC/USD");
    click(btn); assert.strictEqual(w.held("BTC/USD"), 0);
  });

  test("state is flushed to localStorage when the page is hidden", () => {
    const cash = w.st.cash; w.st.cash = cash + 1; // unsaved change
    w.dispatchEvent(new w.Event("pagehide"));
    assert.strictEqual(JSON.parse(w.localStorage.getItem("papertrade.v1")).cash, cash + 1);
  });

  test("reset restores $100k", () => {
    click($("#reset")); click($("#reset"));
    assert.ok($("#stats").textContent.includes("$100,000.00"));
    assert.strictEqual(w.st.ord.length, 0);
  });

  test("no runtime errors were thrown", () => assert.deepStrictEqual(errors, []));

  console.log(failed ? `\n${failed} failed` : "\nall passed");
  process.exit(failed ? 1 : 0);
}, 600);
