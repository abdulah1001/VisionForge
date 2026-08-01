import { expect,test } from "@playwright/test";
test("navigates core routes",async({page})=>{await page.goto("/");await expect(page.getByRole("heading",{name:/Track the subject/i})).toBeVisible();await page.getByRole("link",{name:"Open studio"}).click();await expect(page).toHaveURL(/studio/);await expect(page.getByText("Analysis studio")).toBeVisible()});
test.describe("GPU runtime",()=>{test.describe.configure({mode:"serial"});test.skip("requires configured GPU backend",async()=>{})});
