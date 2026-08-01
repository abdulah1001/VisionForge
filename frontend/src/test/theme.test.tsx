import { render,screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider } from "@/theme/ThemeProvider";
import { ThemeSwitcher } from "@/components/layout/ThemeSwitcher";
describe("theme",()=>{it("switches and persists theme",async()=>{render(<ThemeProvider><ThemeSwitcher/></ThemeProvider>);await userEvent.click(screen.getByRole("button"));expect(document.documentElement.dataset.theme).toBe("porcelain");expect(localStorage.getItem("visionforge-theme")).toBe("porcelain")})});
