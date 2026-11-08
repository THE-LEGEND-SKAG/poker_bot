// End-to-end smoke test: play hands through the real UI and open the Lab.
// usage: (server running on :8000)  npm run e2e   [E2E_URL=http://host:port]
// First time on a new machine: npx playwright install chromium
import { chromium } from "playwright";

const URL = process.env.E2E_URL ?? "http://127.0.0.1:8000/";
const fail = (msg) => {
  console.error(`E2E FAILED: ${msg}`);
  process.exit(1);
};

const browser = await chromium.launch();
const errors = [];
try {
  for (const [label, bots, viewport] of [["6-max", 5, { width: 1440, height: 900 }], ["heads-up (phone)", 1, { width: 390, height: 844 }]]) {
    const page = await browser.newPage({ viewport });
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    await page.goto(URL);
    await page.locator("input[type=number]").first().fill(String(bots));
    const selects = page.locator("select");
    await selects.nth((await selects.count()) - 1).selectOption({ label: "Instant" });
    await page.getByRole("button", { name: "Sit down" }).click();
    let hands = 0;
    let decisions = 0;
    for (let i = 0; i < 600 && hands < 5; i++) {
      await page.waitForTimeout(100);
      const rebuy = page.getByRole("button", { name: "Rebuy" });
      if (await rebuy.isVisible().catch(() => false)) await rebuy.click();
      const next = page.getByRole("button", { name: /Next hand/ });
      if (await next.isVisible().catch(() => false)) {
        hands++;
        await next.click();
        continue;
      }
      const passive = page.getByRole("button", { name: /^(Check|Call)/ }).first();
      if (await passive.isVisible().catch(() => false)) {
        // every 5th decision raise when a raise is offered, otherwise check/call
        const raise = page.getByRole("button", { name: /^(Raise to|Bet|All-in)/ }).first();
        const wantRaise = decisions % 5 === 4 && (await raise.isVisible().catch(() => false));
        decisions++;
        await (wantRaise ? raise : passive).click();
      }
    }
    if (hands < 5) fail(`${label}: only ${hands} hands completed`);
    if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) fail(`${label}: horizontal overflow`);
    console.log(`ok  ${label}: ${hands} hands, ${decisions} decisions`);
    await page.close();
  }
  const lab = await browser.newPage();
  lab.on("pageerror", (e) => errors.push(String(e)));
  await lab.goto(URL + "#lab");
  await lab.getByText("Leaderboard").first().waitFor({ timeout: 15000 });
  await lab.getByText("CFR exploitability").waitFor({ timeout: 15000 });
  console.log("ok  Lab: leaderboard and training charts rendered");
} catch (e) {
  fail(String(e));
} finally {
  await browser.close();
}
if (errors.length) fail(`browser errors: ${errors.join(" | ")}`);
console.log("E2E PASSED");
