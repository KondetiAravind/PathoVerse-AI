import { cn } from "@/lib/utils";

export default function Card({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border border-[#1d3040] bg-[#0c1722]",
        className
      )}
    >
      {children}
    </div>
  );
}