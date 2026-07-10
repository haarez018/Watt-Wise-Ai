import { expect, test } from "@playwright/test";

test("landing page links to signup and login", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /know your next electricity bill/i }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "Get started for free" })).toHaveAttribute(
    "href",
    "/signup",
  );
  await expect(page.getByRole("link", { name: "Log in with existing account" })).toHaveAttribute(
    "href",
    "/login",
  );
});

test("signup form validates required fields", async ({ page }) => {
  await page.goto("/signup");
  await page.getByRole("button", { name: "Sign up" }).click();
  await expect(page.getByText("Enter a valid email address")).toBeVisible();
});

test("visiting a protected route while signed out redirects to login", async ({ page }) => {
  await page.goto("/dashboard");
  await expect(page).toHaveURL(/\/login/);
});
