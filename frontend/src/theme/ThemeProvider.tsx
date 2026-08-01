import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
export type Theme="carbon"|"porcelain";
interface ThemeContextValue{theme:Theme;setTheme:(theme:Theme)=>void;toggle:()=>void}
const ThemeContext=createContext<ThemeContextValue|null>(null);
const initialTheme=():Theme=>document.documentElement.dataset.theme==="porcelain"?"porcelain":"carbon";
export function ThemeProvider({children}:{children:ReactNode}){
  const [theme,setTheme]=useState<Theme>(initialTheme);
  useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem("visionforge-theme",theme);const meta=document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');if(meta)meta.content=theme==="carbon"?"#090B10":"#F3EFE7"},[theme]);
  const value=useMemo(()=>({theme,setTheme,toggle:()=>setTheme(v=>v==="carbon"?"porcelain":"carbon")}),[theme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
export function useTheme(){const value=useContext(ThemeContext);if(!value)throw new Error("useTheme must be used within ThemeProvider");return value}
