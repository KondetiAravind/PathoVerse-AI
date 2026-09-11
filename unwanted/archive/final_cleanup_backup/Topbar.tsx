import { Activity, Circle } from "lucide-react";

export default function Topbar() {
  return (
    <header className="flex h-16 items-center justify-between border-b border-[#1d3040] bg-[#09131d] px-6">
      <div>
        <h2 className="text-sm font-medium text-white">
          Multimodal Pathology AI Platform
        </h2>
        <p className="text-xs text-[#8fa3b5]">
          Whole-slide analysis · Foundation models · Evaluation
        </p>
      </div>

      <div className="flex items-center gap-5">
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
    </header>
  );
}