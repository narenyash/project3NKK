// Manual Playwright driver (chromium-cli not available in this environment) - per the
// run skill's fallback guidance: import { chromium }, launch with --no-sandbox, drive
// the already-running dev server, screenshot at each meaningful step.
import { chromium } from "playwright";
import fs from "fs";

const SCREENSHOT_DIR = "./playwright_screenshots";
fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

const HUMAN_ESSAY = fs.readFileSync("./test_essays/human.txt", "utf-8");
const AI_ESSAY = fs.readFileSync("./test_essays/ai.txt", "utf-8");
const FALSE_POSITIVE_ESSAY = fs.readFileSync("./test_essays/false_positive.txt", "utf-8");
const MIXED_ESSAY = fs.readFileSync("./test_essays/mixed_human_ai.txt", "utf-8");

async function analyzeAndScreenshot(page, label, text) {
  await page.goto("http://localhost:5173");
  await page.waitForSelector("text=Analyze an essay");
  await page.screenshot({ path: `${SCREENSHOT_DIR}/${label}_0_idle.png` });

  await page.fill("textarea.essay-textarea", text);
  await page.click("button.btn-primary");

  try {
    await page.waitForSelector("text=Analyzing essay", { timeout: 2000 });
    await page.screenshot({ path: `${SCREENSHOT_DIR}/${label}_1_loading.png` });
  } catch {
    // loading state may have already passed, that's fine
  }

  await page.waitForSelector(".summary-score-value", { timeout: 60000 });
  await page.screenshot({ path: `${SCREENSHOT_DIR}/${label}_2_results.png`, fullPage: true });

  const scoreText = await page.textContent(".summary-score-value");
  const description = await page.textContent(".summary-score-description");
  console.log(`[${label}] essay-level score shown: ${scoreText} | description: "${description}"`);

  // Highlighting is now a solid background above ESSAY_HIGHLIGHT_CUTOFF (inline style,
  // no more .sentence-red/.sentence-orange classes) - find the first sentence with a
  // non-transparent computed background color.
  const highlightedCount = await page.evaluate(() => {
    return Array.from(document.querySelectorAll(".sentence-span")).filter((el) => {
      const bg = getComputedStyle(el).backgroundColor;
      return bg && bg !== "rgba(0, 0, 0, 0)" && bg !== "transparent";
    }).length;
  });
  console.log(`[${label}] highlighted sentence count: ${highlightedCount}`);

  const paragraphInfo = await page.evaluate(() => {
    const blocks = Array.from(document.querySelectorAll(".paragraph-block"));
    return {
      count: blocks.length,
      firstHeaderText: blocks[0] ? blocks[0].querySelector(".paragraph-header").textContent : null,
      notableCount: blocks.filter((b) => b.classList.contains("paragraph-block-notable")).length,
    };
  });
  console.log(
    `[${label}] paragraph blocks: ${paragraphInfo.count}, notable (>=cutoff): ${paragraphInfo.notableCount}, first header: "${paragraphInfo.firstHeaderText}"`
  );

  const outlierBadgeCount = await page.locator(".outlier-badge").count();
  console.log(`[${label}] outlier badges present: ${outlierBadgeCount}`);
  if (outlierBadgeCount > 0) {
    await page.locator(".outlier-badge").first().click();
    const noteVisible = await page.locator(".outlier-note").first().isVisible();
    const noteText = noteVisible ? await page.locator(".outlier-caveat").first().textContent() : null;
    console.log(
      `[${label}] outlier note expands on click: ${noteVisible}, caveat mentions "not a verdict": ${noteText ? noteText.includes("should not be read as") : false}`
    );
    await page.screenshot({ path: `${SCREENSHOT_DIR}/${label}_1b_outlier.png`, fullPage: true });
  }

  const infoIconHandle = await page.evaluateHandle(() => {
    const spans = Array.from(document.querySelectorAll(".sentence-span"));
    const target = spans.find((el) => {
      const bg = getComputedStyle(el).backgroundColor;
      return bg && bg !== "rgba(0, 0, 0, 0)" && bg !== "transparent";
    });
    if (!target) return null;
    return target.parentElement.querySelector(".info-icon");
  });
  const infoIcon = infoIconHandle.asElement();

  if (infoIcon) {
    await infoIcon.click();
    await page.waitForSelector(".detail-sentence-text");
    await page.screenshot({ path: `${SCREENSHOT_DIR}/${label}_3_detail.png`, fullPage: true });

    const primaryCardCount = await page.$$eval(
      ".detail-section-primary .feature-card",
      (els) => els.length
    );
    const secondaryCardCount = await page.$$eval(
      ".detail-section-secondary .feature-card",
      (els) => els.length
    );
    const detailText = await page.textContent(".detail-card");
    const hasMissingRaw = detailText.includes("Actual value unavailable");
    const hasNamedMetrics = detailText.includes("Measured using 7 factors");

    console.log(
      `[${label}] detail panel opened via info icon. primary factor cards: ${primaryCardCount}, secondary factor cards: ${secondaryCardCount}, named-metrics summary present: ${hasNamedMetrics}, any "unavailable" raw values: ${hasMissingRaw}`
    );
  } else {
    const firstSentence = await page.$(".sentence-span");
    if (firstSentence) {
      await firstSentence.click();
      await page.waitForSelector(".detail-sentence-text");
      await page.screenshot({ path: `${SCREENSHOT_DIR}/${label}_3_detail_none.png`, fullPage: true });
    }
    console.log(`[${label}] NOTE: no highlighted sentence found in this essay.`);
  }

  return { scoreText, description };
}

(async () => {
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });

  console.log("=== TEST 1: Human essay ===");
  await analyzeAndScreenshot(page, "human", HUMAN_ESSAY);

  console.log("\n=== TEST 2: AI essay ===");
  await analyzeAndScreenshot(page, "ai", AI_ESSAY);

  console.log("\n=== TEST 3: Known false-positive essay ===");
  const fpResult = await analyzeAndScreenshot(page, "false_positive", FALSE_POSITIVE_ESSAY);

  console.log("\n=== TEST 4: Mixed essay (2 human paragraphs + 1 AI paragraph) ===");
  await analyzeAndScreenshot(page, "mixed", MIXED_ESSAY);

  const bodyText = await page.textContent("body");
  const verdictPhrases = ["IS AI-written", "is AI written", "IS AI written", "this essay is AI"];
  const foundVerdict = verdictPhrases.filter((p) => bodyText.toLowerCase().includes(p.toLowerCase()));

  console.log(`\nFalse-positive essay score: ${fpResult.scoreText}`);
  console.log(`Verdict-language check: ${foundVerdict.length === 0 ? "PASS - no verdict phrases found" : "FAIL - found: " + foundVerdict.join(", ")}`);

  console.log(`\nConsole/page errors across all tests: ${errors.length === 0 ? "none" : errors.join(" | ")}`);

  await browser.close();
})();
