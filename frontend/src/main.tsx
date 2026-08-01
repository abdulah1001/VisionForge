import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient,QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "@/theme/ThemeProvider";
import { MotionPreferenceProvider } from "@/components/layout/MotionPreferenceProvider";
import "./index.css";
import App from "./App";

const queryClient=new QueryClient({defaultOptions:{queries:{retry:1,refetchOnWindowFocus:true}}});
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider><MotionPreferenceProvider><QueryClientProvider client={queryClient}><BrowserRouter><App /></BrowserRouter></QueryClientProvider></MotionPreferenceProvider></ThemeProvider>
  </StrictMode>,
)
