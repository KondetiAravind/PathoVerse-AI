"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  Brain,
  Database,
  Gauge,
  Microscope,
  Search,
} from "lucide-react";

import AppShell from "@/components/layout/AppShell";
import Card from "@/components/ui/Card";
import {
  getBenchmarks,
  getHealth,
  getModels,
  getSlides,
} from "@/lib/api";

export default function Home() {
  const [health, setHealth] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [slides, setSlides] = useState<any>(null);
  const [benchmarks, setBenchmarks] = useState<any>(null);

  useEffect(() => {
    async function load() {
      try {
        const [
          healthData,
          modelData,
          slideData,
          benchmarkData,
        ] = await Promise.all([
          getHealth(),
          getModels(),
          getSlides(),
          getBenchmarks(),
        ]);

        setHealth(healthData);
        setModels(modelData);
        setSlides(slideData);
        setBenchmarks(benchmarkData);
      } catch (error) {
        console.error("Failed to load PathoVerse data:", error);
      }
    }

    load();
  }, []);

  const cards = [
    {
      label: "Foundation Models",
      value: models?.length ?? models?.models?.length ?? "—",
      icon: Brain,
      description: "Registered pathology models",
    },
    {
      label: "Whole-Slide Studies",
      value: slides?.length ?? slides?.slides?.length ?? "—",
      icon: Microscope,
      description: "Available WSI samples",
    },
    {
      label: "Benchmark Runs",
      value:
        benchmarks?.total_records ??
        benchmarks?.records?.length ??
        "—",
      icon: Gauge,
      description: "Tracked evaluation records",
    },
    {
      label: "API Status",
      value: health?.status ?? "Checking",
      icon: Activity,
      description: health?.service ?? "PathoVerse API",
    },
  ];

  return (
    <AppShell>
      <div className="mx-auto max-w-[1500px] p-6">
        <section className="mb-8">
          <p className="mb-2 text-xs uppercase tracking-[0.2em] text-[#37d5c3]">
            Pathology Intelligence
          </p>

          <h1 className="text-3xl font-semibold tracking-tight text-white">
            PathoVerse AI
          </h1>

          <p className="mt-2 max-w-3xl text-sm leading-6 text-[#8fa3b5]">
            A multimodal foundation model platform for
            whole-slide pathology analysis, retrieval,
            evaluation and explainable AI.
          </p>
        </section>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {cards.map((card) => {
            const Icon = card.icon;

            return (
              <Card key={card.label} className="p-5">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-xs text-[#8fa3b5]">
                      {card.label}
                    </p>

                    <p className="mt-3 text-2xl font-semibold text-white">
                      {String(card.value)}
                    </p>

                    <p className="mt-1 text-xs text-[#60788b]">
                      {card.description}
                    </p>
                  </div>

                  <div className="rounded-lg bg-[#37d5c3]/10 p-2.5">
                    <Icon
                      size={19}
                      className="text-[#37d5c3]"
                    />
                  </div>
                </div>
              </Card>
            );
          })}
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-3">
          <Card className="p-6 xl:col-span-2">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-base font-medium text-white">
                  Platform Architecture
                </h2>

                <p className="mt-1 text-xs text-[#8fa3b5]">
                  End-to-end multimodal pathology workflow
                </p>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-4">
              {[
                ["01", "Whole-Slide Input", "WSI / SVS"],
                ["02", "Foundation Models", "UNI · GigaPath · CONCH"],
                ["03", "AI Analysis", "MIL · Retrieval · Tasks"],
                ["04", "Evaluation", "Metrics · Benchmarks"],
              ].map(([number, title, description]) => (
                <div
                  key={number}
                  className="rounded-lg border border-[#1d3040] bg-[#09131d] p-4"
                >
                  <span className="text-xs text-[#37d5c3]">
                    {number}
                  </span>

                  <h3 className="mt-3 text-sm font-medium text-white">
                    {title}
                  </h3>

                  <p className="mt-2 text-xs leading-5 text-[#8fa3b5]">
                    {description}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-base font-medium text-white">
              Connected Services
            </h2>

            <div className="mt-5 space-y-4">
              {[
                ["FastAPI", "Backend API"],
                ["PyTorch", "Model Runtime"],
                ["OpenSlide", "WSI Engine"],
                ["FAISS", "Vector Retrieval"],
                ["MLflow", "Experiment Tracking"],
              ].map(([name, description]) => (
                <div
                  key={name}
                  className="flex items-center gap-3"
                >
                  <div className="h-2 w-2 rounded-full bg-[#37d5c3]" />

                  <div>
                    <p className="text-sm text-white">
                      {name}
                    </p>

                    <p className="text-xs text-[#60788b]">
                      {description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </section>

        <section className="mt-6 grid gap-6 md:grid-cols-2">
          <Card className="p-6">
            <div className="flex items-center gap-3">
              <Database
                size={18}
                className="text-[#37d5c3]"
              />

              <h2 className="text-base font-medium text-white">
                Dataset Infrastructure
              </h2>
            </div>

            <p className="mt-3 text-sm leading-6 text-[#8fa3b5]">
              Whole-slide images, tissue-aware tiling,
              pathology datasets and embedding stores are
              exposed through the platform API.
            </p>
          </Card>

          <Card className="p-6">
            <div className="flex items-center gap-3">
              <Search
                size={18}
                className="text-[#37d5c3]"
              />

              <h2 className="text-base font-medium text-white">
                Multimodal Retrieval
              </h2>
            </div>

            <p className="mt-3 text-sm leading-6 text-[#8fa3b5]">
              Search pathology tiles using foundation-model
              embeddings and compare retrieval quality
              across models.
            </p>
          </Card>
        </section>
      </div>
    </AppShell>
  );
}