import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/theme/ThemeProvider";

export function ThemeSwitcher() {
  const { theme, toggle } = useTheme();
  return (
    <button
      type="button"
      className="btn btn-ghost h-11 w-11 shrink-0 p-0"
      onClick={toggle}
      aria-label={`Switch to ${theme === "carbon" ? "porcelain" : "carbon"} theme`}
    >
      {theme === "carbon" ? <Sun size={17} /> : <Moon size={17} />}
    </button>
  );
}
