"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Circle } from "lucide-react";

const navigation = [
  { label: "Overview", href: "/" },
  { label: "Workspace", href: "/workspace" },
  { label: "Models", href: "/models" },
  { label: "Analysis", href: "/analysis" },
  { label: "Retrieval", href: "/retrieval" },
  { label: "Benchmarks", href: "/benchmarks" },
  { label: "Analytics", href: "/analytics" },
];

export default function Topbar() {
  const pathname = usePathname();

  return (
    <header className="border-b border-[#1d3040] bg-[#09131d]">
      <div className="flex min-h-16 items-center justify-between gap-6 px-6">
        <div className="min-w-0">
          <h2 className="text-sm font-medium text-white">
            Multimodal Pathology AI Platform
          </h2>

          <p className="truncate text-xs text-[#8fa3b5]">
            Whole-slide analysis · Foundation models · Evaluation
          </p>
        </div>

        <div className="hidden items-center gap-4 xl:flex">
          <div className="flex items-center gap-1 rounded-lg border border-[#1d3040] bg-[#071018] p-1">
            {navigation.map((item) => {
              const active =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`rounded-md px-2.5 py-1.5 text-[10px] font-medium transition ${
                    active
                      ? "bg-[#37d5c3]/10 text-[#37d5c3]"
                      : "text-[#60788b] hover:bg-[#101e2b] hover:text-white"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs text-[#8fa3b5]">
              <Circle
                size={8}
                fill="currentColor"
                className="text-[#37d5c3]"
              />
              API Connected
            </div>

            <div className="flex items-center gap-2 text-xs text-[#8fa3b5]">
              <Activity size={15} />
              GPU Ready
            </div>
          </div>
        </div>
      </div>

      <div className="flex gap-1 overflow-x-auto border-t border-[#1d3040]/60 px-6 py-2 xl:hidden">
        {navigation.map((item) => {
          const active =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`whitespace-nowrap rounded-md px-2.5 py-1.5 text-[10px] font-medium ${
                active
                  ? "bg-[#37d5c3]/10 text-[#37d5c3]"
                  : "text-[#60788b]"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </div>
    </header>
  );
}
