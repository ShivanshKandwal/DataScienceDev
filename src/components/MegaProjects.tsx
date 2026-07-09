import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Cpu, ExternalLink, Layers, Terminal, Settings, CheckCircle2 } from 'lucide-react';

export function MegaProjects() {
  const [activeTab, setActiveTab] = useState<'SignalScout' | 'DevIntel'>('SignalScout');

  return (
    <div className="space-y-12">
      {/* Page Header */}
      <div className="text-left max-w-3xl">
        <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-700 text-sm font-semibold tracking-wide mb-3 select-none">
          <Layers className="w-4 h-4" /> PIPELINE-BASED END-TO-END PROJECTS
        </div>
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-slate-800">
          Data Science Mega Projects
        </h2>
        <p className="text-slate-650 text-sm md:text-base mt-2 leading-relaxed">
          Explore complete production-grade data science pipelines. These projects feature multi-agent frameworks, dense/sparse semantic retrieval, real-time hardware telemetry dashboards, and ML model evaluation layers.
        </p>
      </div>

      {/* Modern Neon Segmented Tab Switcher */}
      <div className="flex justify-center select-none">
        <div className="flex gap-2 bg-slate-200/50 p-1.5 rounded-2xl border border-slate-300/40 shadow-sm backdrop-blur-xl relative">
          <button
            onClick={() => setActiveTab('SignalScout')}
            className={`relative px-6 py-3 rounded-xl text-base font-bold transition-all z-10 flex items-center gap-2 ${
              activeTab === 'SignalScout'
                ? 'text-white'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            {activeTab === 'SignalScout' && (
              <motion.div
                layoutId="activeMegaTab"
                className="absolute inset-0 bg-gradient-to-tr from-purple-600 to-indigo-600 rounded-xl shadow-lg shadow-purple-500/10 z-[-1]"
                transition={{ type: "spring", stiffness: 350, damping: 28 }}
              />
            )}
            <Sparkles className="w-4.5 h-4.5" />
            SignalScout Agent 📡
          </button>

          <button
            onClick={() => setActiveTab('DevIntel')}
            className={`relative px-6 py-3 rounded-xl text-base font-bold transition-all z-10 flex items-center gap-2 ${
              activeTab === 'DevIntel'
                ? 'text-white'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            {activeTab === 'DevIntel' && (
              <motion.div
                layoutId="activeMegaTab"
                className="absolute inset-0 bg-gradient-to-tr from-indigo-600 to-cyan-600 rounded-xl shadow-lg shadow-indigo-500/10 z-[-1]"
                transition={{ type: "spring", stiffness: 350, damping: 28 }}
              />
            )}
            <Cpu className="w-4.5 h-4.5" />
            DevIntel Hub 🧬
          </button>
        </div>
      </div>

      {/* Content Render Pane */}
      <AnimatePresence mode="wait">
        {activeTab === 'SignalScout' ? (
          <motion.div
            key="SignalScout"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
            className="space-y-12 text-left"
          >
            {/* Ticker & Link Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white/40 border border-white/60 p-6 rounded-3xl backdrop-blur-md shadow-sm">
              <div>
                <h3 className="font-serif text-2xl font-bold text-slate-800">
                  SignalScout: Multimodal Market Intelligence Agent
                </h3>
                <p className="text-slate-550 text-sm mt-1">
                  A 5-agent LangGraph system implementing Whisper earnings call transcription, SEC filing vector search, and VLM chart captioning.
                </p>
              </div>
              <a
                href="https://signalscout-six.vercel.app/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-5 py-3 rounded-xl bg-purple-600 text-white font-bold hover:bg-purple-700 transition-all shadow-md shadow-purple-600/10 group text-sm shrink-0"
              >
                <span>Launch App</span>
                <ExternalLink className="w-4 h-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </a>
            </div>

            {/* Embedded Live Console Window with Red, Yellow, Green Window controls */}
            <div className="rounded-2xl border border-slate-200/80 overflow-hidden shadow-2xl bg-slate-900 group hover:shadow-purple-500/5 transition-all">
              <div className="flex items-center justify-between px-5 py-3 bg-slate-950/80 border-b border-slate-800">
                <div className="flex gap-2">
                  <div className="w-3.5 h-3.5 rounded-full bg-rose-500" />
                  <div className="w-3.5 h-3.5 rounded-full bg-amber-400" />
                  <div className="w-3.5 h-3.5 rounded-full bg-emerald-500" />
                </div>
                <div className="text-xs font-mono text-slate-400 select-none">
                  https://signalscout-six.vercel.app/
                </div>
                <div className="w-12" /> {/* Spacer */}
              </div>
              <div className="relative">
                <iframe
                  src="https://signalscout-six.vercel.app/"
                  className="w-full h-[760px] bg-slate-900 border-none"
                  title="SignalScout App Embed"
                />
              </div>
            </div>

            {/* Interactive SVG Flow Diagram */}
            <div className="glass-card rounded-3xl p-8 border border-black/5 bg-white/40 shadow-sm space-y-6">
              <h4 className="font-serif text-xl font-bold text-slate-800 flex items-center gap-2">
                <Layers className="w-5 h-5 text-purple-650" />
                System Pipeline Architecture
              </h4>
              
              {/* Pipeline Flow representation */}
              <div className="w-full overflow-x-auto pb-4">
                <div className="min-w-[800px] flex items-center justify-between gap-4 py-6 px-4 bg-slate-900/5 rounded-2xl border border-black/5">
                  
                  {/* Step 1: Modality Ingestion */}
                  <div className="flex flex-col gap-2.5 w-48 text-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm shrink-0">
                    <span className="text-[10px] font-mono font-bold text-purple-600 uppercase tracking-widest">01 / Ingestion</span>
                    <h5 className="font-bold text-sm text-slate-800">Multimodal Feeds</h5>
                    <div className="space-y-1 text-left text-xs text-slate-500">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-purple-500" /> SEC Filings (PDF)
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-purple-500" /> Audio (Whisper)
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-purple-500" /> Charts (Idefics3)
                      </div>
                    </div>
                  </div>

                  <span className="text-slate-400 font-bold">→</span>

                  {/* Step 2: Dense & Sparse Indexing */}
                  <div className="flex flex-col gap-2.5 w-48 text-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm shrink-0">
                    <span className="text-[10px] font-mono font-bold text-purple-600 uppercase tracking-widest">02 / Storage</span>
                    <h5 className="font-bold text-sm text-slate-800">Hybrid Databases</h5>
                    <div className="space-y-1 text-left text-xs text-slate-500">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> pgvector Embeddings
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> Sparse BM25 Indexes
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> Redis Cache Store
                      </div>
                    </div>
                  </div>

                  <span className="text-slate-400 font-bold">→</span>

                  {/* Step 3: LangGraph Multi-Agent Core */}
                  <div className="flex flex-col gap-2 bg-gradient-to-br from-purple-600 to-indigo-600 p-4 rounded-xl shadow text-white w-56 shrink-0">
                    <span className="text-[10px] font-mono font-bold text-white/80 uppercase tracking-widest">03 / Graph Agent Engine</span>
                    <h5 className="font-bold text-sm">LangGraph Core</h5>
                    <div className="space-y-1 text-left text-xs text-white/90">
                      <div>• Orchestrator Intent Parser</div>
                      <div>• Dense/Sparse Hybrid Retrieval</div>
                      <div>• Citation Verification Node</div>
                      <div>• Critique &amp; Self-Correction</div>
                    </div>
                  </div>

                  <span className="text-slate-400 font-bold">→</span>

                  {/* Step 4: Output & Evaluation */}
                  <div className="flex flex-col gap-2.5 w-48 text-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm shrink-0">
                    <span className="text-[10px] font-mono font-bold text-purple-600 uppercase tracking-widest">04 / Evaluation</span>
                    <h5 className="font-bold text-sm text-slate-800">Brief &amp; Metrics</h5>
                    <div className="space-y-1 text-left text-xs text-slate-500">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> Cited Markdown Brief
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> RAGAS Competence
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> LangSmith Telemetry
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            </div>

            {/* In-depth Specifications Grid */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
              
              {/* Left Column: Tech Stack & Checklist */}
              <div className="md:col-span-7 space-y-8 text-left">
                <div className="glass-card rounded-3xl p-6 border border-black/5 bg-white/40 shadow-sm">
                  <h4 className="font-serif text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-purple-600" /> Core Capabilities
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-slate-650">
                    <div className="flex gap-2 items-start">
                      <span className="text-purple-600 mt-0.5">✔</span>
                      <span>Whisper Earnings Call audio transcription</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-purple-600 mt-0.5">✔</span>
                      <span>Docling dense chunk extraction</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-purple-600 mt-0.5">✔</span>
                      <span>Idefics3 8B VLM visual chart captioning</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-purple-600 mt-0.5">✔</span>
                      <span>Hybrid BM25 + dense vector matching</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-purple-600 mt-0.5">✔</span>
                      <span>Self-correction guardrails (Critique Agent)</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-purple-600 mt-0.5">✔</span>
                      <span>RAGAS evaluation radar scoring</span>
                    </div>
                  </div>
                </div>

                <div className="glass-card rounded-3xl p-6 border border-black/5 bg-white/40 shadow-sm">
                  <h4 className="font-serif text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                    <Settings className="w-5 h-5 text-purple-600" /> Tech Stack &amp; Libraries
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {['LangGraph', 'PyTorch', 'docling', 'Whisper', 'Idefics3 VLM', 'pgvector', 'Redis', 'BM25', 'RAGAS', 'LangSmith', 'FastAPI', 'React'].map((tag) => (
                      <span key={tag} className="px-3 py-1.5 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-750 font-bold text-xs">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Column: Code Deployment Snippets */}
              <div className="md:col-span-5 text-left">
                <div className="glass-card rounded-3xl p-6 border border-black/5 bg-white/40 shadow-sm space-y-4">
                  <h4 className="font-serif text-lg font-bold text-slate-800 flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-purple-650" /> Local Setup Blueprint
                  </h4>
                  <p className="text-slate-650 text-xs leading-relaxed">
                    Clone the project and start the infrastructure using Docker.
                  </p>
                  
                  <div className="bg-slate-900 rounded-2xl p-4 font-mono text-[11px] text-indigo-200 space-y-2 overflow-x-auto shadow-inner">
                    <div>
                      <span className="text-slate-500"># Navigate to directory &amp; install</span>
                      <br />
                      cd c:/Software/signalscout
                      <br />
                      pip install -e ".[dev]"
                    </div>
                    <div>
                      <span className="text-slate-500"># Run local database services</span>
                      <br />
                      docker compose up -d postgres redis
                    </div>
                    <div>
                      <span className="text-slate-500"># Ingest multimodal filings</span>
                      <br />
                      python scripts/ingest_cli.py all --tickers AAPL
                    </div>
                    <div>
                      <span className="text-slate-500"># Fire API server</span>
                      <br />
                      uvicorn signalscout.api.main:app --reload
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </motion.div>
        ) : (
          <motion.div
            key="DevIntel"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
            className="space-y-12 text-left"
          >
            {/* Ticker & Link Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white/40 border border-white/60 p-6 rounded-3xl backdrop-blur-md shadow-sm">
              <div>
                <h3 className="font-serif text-2xl font-bold text-slate-800">
                  DevIntel: Developer Intelligence &amp; Telemetry Hub
                </h3>
                <p className="text-slate-550 text-sm mt-1">
                  Predictive ML dashboard modeling compensation structures, attrition risks, and semantic developer indexing.
                </p>
              </div>
              <a
                href="https://shivanshkandwal.github.io/dev_analysis_2/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-5 py-3 rounded-xl bg-indigo-600 text-white font-bold hover:bg-indigo-700 transition-all shadow-md shadow-indigo-600/10 group text-sm shrink-0"
              >
                <span>Launch App</span>
                <ExternalLink className="w-4 h-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </a>
            </div>

            {/* Embedded Live Console Window with Red, Yellow, Green Window controls */}
            <div className="rounded-2xl border border-slate-200/80 overflow-hidden shadow-2xl bg-slate-950 group hover:shadow-indigo-500/5 transition-all">
              <div className="flex items-center justify-between px-5 py-3 bg-slate-955/80 border-b border-slate-800">
                <div className="flex gap-2">
                  <div className="w-3.5 h-3.5 rounded-full bg-rose-500" />
                  <div className="w-3.5 h-3.5 rounded-full bg-amber-400" />
                  <div className="w-3.5 h-3.5 rounded-full bg-emerald-500" />
                </div>
                <div className="text-xs font-mono text-slate-400 select-none">
                  https://shivanshkandwal.github.io/dev_analysis_2/
                </div>
                <div className="w-12" /> {/* Spacer */}
              </div>
              <div className="relative">
                <iframe
                  src="https://shivanshkandwal.github.io/dev_analysis_2/"
                  className="w-full h-[760px] bg-slate-950 border-none"
                  title="DevIntel App Embed"
                />
              </div>
            </div>

            {/* Interactive SVG Flow Diagram */}
            <div className="glass-card rounded-3xl p-8 border border-black/5 bg-white/40 shadow-sm space-y-6">
              <h4 className="font-serif text-xl font-bold text-slate-800 flex items-center gap-2">
                <Layers className="w-5 h-5 text-indigo-600" />
                System Pipeline Architecture
              </h4>
              
              {/* Pipeline Flow representation */}
              <div className="w-full overflow-x-auto pb-4">
                <div className="min-w-[800px] flex items-center justify-between gap-4 py-6 px-4 bg-slate-900/5 rounded-2xl border border-black/5">
                  
                  {/* Step 1: Raw Datasets Ingestion */}
                  <div className="flex flex-col gap-2.5 w-48 text-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm shrink-0">
                    <span className="text-[10px] font-mono font-bold text-indigo-600 uppercase tracking-widest">01 / Data Ingestion</span>
                    <h5 className="font-bold text-sm text-slate-800">Survey Repositories</h5>
                    <div className="space-y-1 text-left text-xs text-slate-500">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> Stack Overflow Datasets
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> GitHub Profiles
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> High-dim matrices
                      </div>
                    </div>
                  </div>

                  <span className="text-slate-400 font-bold">→</span>

                  {/* Step 2: Processing & Encoding */}
                  <div className="flex flex-col gap-2.5 w-48 text-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm shrink-0">
                    <span className="text-[10px] font-mono font-bold text-indigo-600 uppercase tracking-widest">02 / Processing</span>
                    <h5 className="font-bold text-sm text-slate-800">Feature Engineering</h5>
                    <div className="space-y-1 text-left text-xs text-slate-500">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> One-hot target encoders
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> UMAP dimensionality maps
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" /> FAISS dense embeddings
                      </div>
                    </div>
                  </div>

                  <span className="text-slate-400 font-bold">→</span>

                  {/* Step 3: Estimators & Predictive Core */}
                  <div className="flex flex-col gap-2 bg-gradient-to-br from-indigo-600 to-cyan-600 p-4 rounded-xl shadow text-white w-56 shrink-0">
                    <span className="text-[10px] font-mono font-bold text-white/80 uppercase tracking-widest">03 / Model Inference</span>
                    <h5 className="font-bold text-sm">Gradio Predictive app.py</h5>
                    <div className="space-y-1 text-left text-xs text-white/90">
                      <div>• XGBoost Salary Regressor</div>
                      <div>• LightGBM Attrition Risk Classification</div>
                      <div>• Uplift Causal satisfied analysis</div>
                      <div>• FAISS semantic vector matcher</div>
                    </div>
                  </div>

                  <span className="text-slate-400 font-bold">→</span>

                  {/* Step 4: Client rendering */}
                  <div className="flex flex-col gap-2.5 w-48 text-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm shrink-0">
                    <span className="text-[10px] font-mono font-bold text-indigo-600 uppercase tracking-widest">04 / UI Client</span>
                    <h5 className="font-bold text-sm text-slate-800">React Telemetry dashboard</h5>
                    <div className="space-y-1 text-left text-xs text-slate-500">
                      <div>• Neon gradient metrics</div>
                      <div>• Interactive A/B tests</div>
                      <div>• FAISS semantic prompts</div>
                      <div>• Multi-year forecasts</div>
                    </div>
                  </div>

                </div>
              </div>
            </div>

            {/* In-depth Specifications Grid */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
              
              {/* Left Column: Tech Stack & Checklist */}
              <div className="md:col-span-7 space-y-8 text-left">
                <div className="glass-card rounded-3xl p-6 border border-black/5 bg-white/40 shadow-sm">
                  <h4 className="font-serif text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-indigo-600" /> Core Capabilities
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-slate-650">
                    <div className="flex gap-2 items-start">
                      <span className="text-indigo-600 mt-0.5">✔</span>
                      <span>Compensation regression via XGBoost/LightGBM</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-indigo-600 mt-0.5">✔</span>
                      <span>FAISS semantic search indexing 15,000 developers</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-indigo-600 mt-0.5">✔</span>
                      <span>Uplift modeling for job satisfaction (Causal AB)</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-indigo-600 mt-0.5">✔</span>
                      <span>UMAP dimensionality mapping skill profiles</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-indigo-600 mt-0.5">✔</span>
                      <span>Gradio backend inference engine api endpoints</span>
                    </div>
                    <div className="flex gap-2 items-start">
                      <span className="text-indigo-600 mt-0.5">✔</span>
                      <span>Interactive client telemetry graphs</span>
                    </div>
                  </div>
                </div>

                <div className="glass-card rounded-3xl p-6 border border-black/5 bg-white/40 shadow-sm">
                  <h4 className="font-serif text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                    <Settings className="w-5 h-5 text-indigo-600" /> Tech Stack &amp; Libraries
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {['XGBoost', 'LightGBM', 'Random Forest', 'SentenceTransformers', 'FAISS', 'UMAP', 'scikit-learn', 'Gradio', 'FastAPI', 'Pandas', 'NumPy', 'React'].map((tag) => (
                      <span key={tag} className="px-3 py-1.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-750 font-bold text-xs">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Column: Code Deployment Snippets */}
              <div className="md:col-span-5 text-left">
                <div className="glass-card rounded-3xl p-6 border border-black/5 bg-white/40 shadow-sm space-y-4">
                  <h4 className="font-serif text-lg font-bold text-slate-800 flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-indigo-600" /> Local Setup Blueprint
                  </h4>
                  <p className="text-slate-650 text-xs leading-relaxed">
                    Set up python backend and run react frontend locally.
                  </p>
                  
                  <div className="bg-slate-900 rounded-2xl p-4 font-mono text-[11px] text-indigo-200 space-y-2 overflow-x-auto shadow-inner">
                    <div>
                      <span className="text-slate-500"># Navigate to directory &amp; install</span>
                      <br />
                      cd c:/Software/dev_analysis_2
                      <br />
                      pip install -r requirements.txt
                    </div>
                    <div>
                      <span className="text-slate-500"># Run Gradio backend app.py</span>
                      <br />
                      python app.py
                    </div>
                    <div>
                      <span className="text-slate-500"># Start frontend server</span>
                      <br />
                      cd frontend
                      <br />
                      npm install
                      <br />
                      npm run dev
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
