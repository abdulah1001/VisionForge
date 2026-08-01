import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { MotionConfig } from "framer-motion";

interface MotionCtx {
  reduced: boolean;
}

const Ctx = createContext<MotionCtx>({ reduced: false });

export function MotionPreferenceProvider({ children }: { children: ReactNode }) {
  const [reduced, setReduced] = useState(() =>
    typeof window !== "undefined"
      ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false,
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const value = useMemo(() => ({ reduced }), [reduced]);

  return (
    <Ctx.Provider value={value}>
      <MotionConfig reducedMotion="user">{children}</MotionConfig>
    </Ctx.Provider>
  );
}

export function useMotionPreference() {
  return useContext(Ctx);
}
