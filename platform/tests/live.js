// Exercises the Alpaca live-price code path against a mocked REST API and WebSocket.
// No network or real keys needed.  Run:  node tests/live.js
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(e.message));

const SYMS = ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","AMD","NFLX","SPY","QQQ","COIN"];
const now = new Date().toISOString();
const bar = (p) => ({ t: now, o: p, h: p, l: p, c: p, v: 100 });
const snap = (p) => ({ latestTrade: { p, t: now }, minuteBar: bar(p), prevDailyBar: { c: p * 0.99 } });
const calls = [];
let sockets = [];

const dom = new JSDOM(html, {
  url: "http://localhost/", runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
  beforeParse(w) {
    w.localStorage.setItem("papertrade.mode", "live");
    w.localStorage.setItem("papertrade.keys", JSON.stringify({ id: "KEY", sec: "SECRET" }));
    w.fetch = async (url, opts) => {
      calls.push(url);
      assert.ok(opts.headers["APCA-API-KEY-ID"] === "KEY", "auth header sent");
      const ok = (j) => ({ ok: true, status: 200, json: async () => j });
      if (url.includes("/crypto/") && url.includes("snapshots")) return ok({ snapshots: { "BTC/USD": snap(64000) } });
      if (url.includes("/crypto/") && url.includes("bars")) return ok({ bars: { "BTC/USD": [bar(63900), bar(64000)] } });
      if (url.includes("snapshots")) { const o = {}; SYMS.forEach((s, i) => (o[s] = snap(100 + i))); return ok({ snapshots: o }); }
      if (url.includes("/bars")) return ok({ bars: [bar(101), bar(100)] });
      return { ok: false, status: 404, json: async () => ({}) };
    };
    w.WebSocket = class {
      constructor(url) { this.url = url; this.sent = []; sockets.push(this); setTimeout(() => this.onmessage({ data: JSON.stringify([{ T: "success", msg: "connected" }]) }), 5); }
      send(m) { this.sent.push(JSON.parse(m)); const a = this.sent[this.sent.length - 1];
        if (a.action === "auth") setTimeout(() => this.onmessage({ data: JSON.stringify([{ T: "success", msg: "authenticated" }]) }), 5); }
      close() { if (this.onclose) this.onclose(); }
    };
  },
});
const w = dom.window, d = w.document, $ = (s) => d.querySelector(s);

setTimeout(() => {
  try {
    assert.strictEqual($("#mb").textContent, "LIVE PRICES");
    assert.ok(calls.some((u) => u.includes("snapshots")), "polled snapshots");
    assert.ok(calls.some((u) => u.includes("/bars")), "loaded history bars");
    assert.strictEqual(w.liveReady, true);
    assert.strictEqual(w.st.px.AAPL, 100);
    assert.strictEqual(w.st.px["BTC/USD"], 64000);
    assert.strictEqual(sockets.length, 2, "stock + crypto streams opened");
    assert.ok(sockets.every((s) => s.sent.some((m) => m.action === "subscribe")), "both subscribed after auth");

    // a streamed trade updates the price, and trading works in live mode
    sockets[0].onmessage({ data: JSON.stringify([{ T: "t", S: "AAPL", p: 123.45, s: 10, t: new Date().toISOString() }]) });
    assert.strictEqual(w.st.px.AAPL, 123.45);
    $("#qty").value = 2; $("#go").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
    assert.strictEqual($("#msg").textContent, "Order placed.");
    assert.strictEqual(w.held("AAPL"), 2);
    assert.deepStrictEqual(errors, []);
    console.log("live-mode mock test: all passed");
    process.exit(0);
  } catch (e) { console.log("live-mode mock test FAILED:", e.message, errors); process.exit(1); }
}, 900);
