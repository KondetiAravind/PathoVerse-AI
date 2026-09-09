"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Brain,
  Database,
  FlaskConical,
  Home,
  ImageIcon,
  Search,
  Settings2,
} from "lucide-react";

const navigation = [
  {
    label: "Overview",
    href: "/",
    icon: Home,
  },
  {
    label: "Workspace",
    href: "/workspace",
    icon: ImageIcon,
  },
  {
    label: "Models",
    href: "/models",
    icon: Brain,
  },
  {
    label: "Analysis",
    href: "/analysis",
    icon: Activity,
  },
  {
    label: "Retrieval",
    href: "/retrieval",
    icon: Search,
  },
  {
    label: "Benchmarks",
    href: "/benchmarks",
    icon: BarChart3,
  },
  {
    label: "Analytics",
    href: "/analytics",
    icon: FlaskConical,
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-64 flex-col border-r border-[#1d3040] bg-[#09131d]">
      <div className="border-b border-[#1d3040] px-5 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#37d5c3]/10">
            <Database size={21} className="text-[#37d5c3]" />
          </div>

          <div>
            <h1 className="text-base font-semibold text-white">
              PathoVerse AI
            </h1>
            <p className="text-[11px] text-[#8fa3b5]">
              Pathology Intelligence
            </p>
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-5">
        {navigation.map((item) => {
          const Icon = item.icon;

          const active =
            pathname === item.href ||
            (item.href !== "/" &&
              pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
                active
                  ? "bg-[#37d5c3]/10 text-[#37d5c3]"
                  : "text-[#8fa3b5] hover:bg-[#101e2b] hover:text-white"
              }`}
            >
              <Icon size={17} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-[#1d3040] p-3">
        <div className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-[#8fa3b5]">
          <Settings2 size={17} />
          Platform
          <span className="ml-auto text-[10px]">v0.4</span>
        </div>
      </div>
    </aside>
  );
}