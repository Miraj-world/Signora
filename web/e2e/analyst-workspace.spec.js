const { test, expect } = require("@playwright/test");
const { answered, abstained } = require("./fixtures");

async function preparePage(page, payload, onRequest = () => {}) {
  await page.route("**/ready", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ status: "ready" }),
  }));
  await page.route("**/v1/answers", async (route) => {
    onRequest(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(payload),
    });
  });
  await page.goto("/");
}

test("serializes filters and connects citations to evidence", async ({ page }) => {
  let requestBody;
  await preparePage(page, answered, (body) => { requestBody = body; });

  await page.getByLabel("Customer segment").selectOption("enterprise");
  await page.getByLabel("Product area").selectOption("onboarding");
  await page.getByLabel("Severity").selectOption("high");
  await page.getByLabel("Source type").selectOption("support_ticket");
  await page.getByLabel("Top K").fill("12");
  await page.locator("#query").fill("Why is enterprise onboarding difficult?");
  await page.getByRole("button", { name: "Search" }).click();

  await expect(page.locator("#resultStatus")).toHaveText("Answered");
  await expect(page.locator("#citationCount")).toHaveText("2");
  await expect(page.locator(".evidence-row")).toHaveCount(2);
  expect(requestBody).toMatchObject({
    query: "Why is enterprise onboarding difficult?",
    top_k: 12,
    customer_segment: "enterprise",
    product_area: "onboarding",
    severity: "high",
    source_type: "support_ticket",
  });

  await page.locator("#answerCopy").getByRole("button", { name: "[atom_sso]" }).click();
  await expect(page.locator('.evidence-row[data-atom="atom_sso"]')).toHaveClass(/selected/);
});

test("renders a below-threshold abstention without citations", async ({ page }) => {
  await preparePage(page, abstained);
  await page.locator("#query").fill("What do customers say about cryptocurrency payments?");
  await page.getByRole("button", { name: "Search" }).click();

  await expect(page.locator("#resultStatus")).toHaveText("Abstained");
  await expect(page.locator("#confidence")).toHaveText("0.177");
  await expect(page.getByRole("heading", { name: "No confident answer" })).toBeVisible();
  await expect(page.locator(".evidence-row")).toHaveCount(0);
});

test("reset restores the broad default retrieval scope", async ({ page }) => {
  await preparePage(page, answered);
  await page.getByLabel("Customer segment").selectOption("enterprise");
  await page.getByLabel("Product area").selectOption("onboarding");
  await page.getByLabel("Top K").fill("20");
  await page.getByRole("button", { name: "Reset" }).click();

  await expect(page.getByLabel("Customer segment")).toHaveValue("");
  await expect(page.getByLabel("Product area")).toHaveValue("");
  await expect(page.getByLabel("Top K")).toHaveValue("8");
});
