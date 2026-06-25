import { useState, useEffect } from 'react';
import type { SVGProps } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Sparkles, ChevronRight, BookOpen, Binary, BarChart2 } from 'lucide-react';
import * as echarts from 'echarts';
import { NeuralBackground } from './components/NeuralBackground';
import { EChartsCard } from './components/EChartsCard';
import { D3Network } from './components/D3Network';
import { ProjectCard } from './components/ProjectCard';
import { NotebookViewer } from './components/NotebookViewer';
import { projectsData } from './data/projects';
import type { ProjectData } from './data/projects';

// Custom SVG Brand Icons
const GithubIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

const LinkedinIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect width="4" height="12" x="2" y="9" />
    <circle cx="4" cy="4" r="2" />
  </svg>
);

export default function App() {
  const [currentPage, setCurrentPage] = useState<'Home' | 'Visuals' | 'ML' | 'AI'>('Home');
  const [selectedProject, setSelectedProject] = useState<ProjectData | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Update mouse position for spotlight cursor glow
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Spotlight color shifts based on selected tab (Light Theme colors)
  const getGlowColor = () => {
    switch (currentPage) {
      case 'Visuals': return 'radial-gradient(circle, rgba(16, 185, 129, 0.12) 0%, rgba(16, 185, 129, 0) 70%)';
      case 'ML': return 'radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, rgba(99, 102, 241, 0) 70%)';
      case 'AI': return 'radial-gradient(circle, rgba(244, 63, 94, 0.12) 0%, rgba(244, 63, 94, 0) 70%)';
      default: return 'radial-gradient(circle, rgba(168, 85, 247, 0.12) 0%, rgba(168, 85, 247, 0) 70%)';
    }
  };

  // ECharts training configuration
  const trainingChartOption: echarts.EChartsOption = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      textStyle: { color: '#1E293B' }
    },
    legend: {
      data: ['Training Loss', 'Validation Accuracy'],
      textStyle: { color: '#64748B', fontFamily: 'Outfit, sans-serif' },
      bottom: '0%'
    },
    grid: { left: '3%', right: '4%', top: '10%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category' as const,
      boundaryGap: false,
      data: ['Epoch 20', 'Epoch 40', 'Epoch 60', 'Epoch 80', 'Epoch 100', 'Epoch 120'],
      axisLabel: { color: '#64748B' },
      axisLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.06)' } }
    },
    yAxis: [
      {
        type: 'value' as const,
        name: 'Loss',
        position: 'left',
        splitLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.03)' } },
        axisLabel: { color: '#64748B' }
      },
      {
        type: 'value' as const,
        name: 'Accuracy',
        position: 'right',
        max: 100,
        splitLine: { show: false },
        axisLabel: { color: '#64748B' }
      }
    ],
    series: [
      {
        name: 'Training Loss',
        type: 'line' as const,
        smooth: true,
        yAxisIndex: 0,
        data: [0.74, 0.42, 0.25, 0.14, 0.08, 0.04],
        itemStyle: { color: '#F43F5E' },
        lineStyle: { width: 3 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(244, 63, 94, 0.08)' },
            { offset: 1, color: 'rgba(244, 63, 94, 0)' }
          ])
        }
      },
      {
        name: 'Validation Accuracy',
        type: 'line' as const,
        smooth: true,
        yAxisIndex: 1,
        data: [72, 86, 92, 95.8, 96.8, 97.4],
        itemStyle: { color: '#10B981' },
        lineStyle: { width: 3 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(16, 185, 129, 0.08)' },
            { offset: 1, color: 'rgba(16, 185, 129, 0)' }
          ])
        }
      }
    ]
  };

  return (
    <div className="relative min-h-screen overflow-hidden selection:bg-purple-500/20 selection:text-slate-900 text-slate-700 pb-20">
      
      {/* Interactive Cursor Spotlight Glow */}
      <div 
        className="cursor-glow"
        style={{
          left: mousePos.x,
          top: mousePos.y,
          background: getGlowColor()
        }}
      />

      {/* Dynamic Animated Bright Gradient Blobs */}
      <div className="absolute top-[-10%] left-[-5%] w-[600px] h-[600px] blob blob-purple opacity-40 z-0" />
      <div className="absolute bottom-[20%] right-[-10%] w-[700px] h-[700px] blob blob-indigo opacity-35 z-0" />
      <div className="absolute top-[40%] left-[30%] w-[600px] h-[600px] blob blob-pink opacity-30 z-0" />
      <div className="absolute bottom-[-10%] left-[-10%] w-[650px] h-[650px] blob blob-cyan opacity-25 z-0" />

      {/* Canvas Neural Background */}
      <NeuralBackground />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        
        {/* Navigation / Header */}
        <header className="flex flex-col md:flex-row justify-between items-center py-5 border-b border-black/5 mb-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-purple-500 to-pink-500 flex items-center justify-center shadow-md shadow-purple-500/10">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div className="text-left">
              <h1 className="font-serif text-2xl md:text-3xl font-bold tracking-tight text-slate-800 m-0">
                Shivansh Kandwal
              </h1>
              <p className="text-xs md:text-sm text-slate-500 font-mono tracking-wide mt-0.5">
                AI &amp; Data Science Research Sandbox
              </p>
            </div>
          </div>

          {/* Social Links */}
          <div className="flex gap-4 mt-4 md:mt-0">
            <a 
              href="https://github.com/ShivanshKandwal" 
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2.5 rounded-xl bg-white/60 border border-slate-200/60 hover:border-slate-350 text-slate-500 hover:text-slate-800 hover:bg-white/90 transition-all"
            >
              <GithubIcon className="w-4.5 h-4.5" />
            </a>
            <a 
              href="https://linkedin.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2.5 rounded-xl bg-white/60 border border-slate-200/60 hover:border-slate-350 text-slate-500 hover:text-slate-800 hover:bg-white/90 transition-all"
            >
              <LinkedinIcon className="w-4.5 h-4.5" />
            </a>
            <a 
              href="mailto:contact@shivanshkandwal.com" 
              className="p-2.5 rounded-xl bg-white/60 border border-slate-200/60 hover:border-slate-350 text-slate-500 hover:text-slate-800 hover:bg-white/90 transition-all"
            >
              <Mail className="w-4.5 h-4.5" />
            </a>
          </div>
        </header>

        {/* Sticky top Navigation Links (Decoupled router) */}
        <div className="flex items-center justify-center mb-12">
          <div className="flex gap-1.5 bg-white/40 p-1.5 rounded-2xl border border-black/5 shadow-sm backdrop-blur">
            {([
              { key: 'Home', label: 'Overview' },
              { key: 'Visuals', label: 'Data Visuals & Dashboards' },
              { key: 'ML', label: 'Machine Learning' },
              { key: 'AI', label: 'Deep Learning / AI' }
            ] as const).map((tab) => {
              const isActive = currentPage === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setCurrentPage(tab.key)}
                  className={`relative px-4 py-2.5 rounded-xl text-xs md:text-sm font-semibold transition-all ${
                    isActive 
                      ? 'bg-gradient-to-tr from-purple-500 to-pink-500 text-white shadow shadow-purple-500/20' 
                      : 'text-slate-500 hover:text-slate-800 hover:bg-white/40'
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Page Switcher */}
        <div className="min-h-[50vh]">
          <AnimatePresence mode="wait">
            
            {/* 1. LANDING PAGE: OVERVIEW & STRUCTURE GUIDE */}
            {currentPage === 'Home' && (
              <motion.div
                key="home"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-12 max-w-4xl mx-auto"
              >
                {/* Hero Introduction */}
                <div className="text-center space-y-4">
                  <h2 className="font-serif text-3xl md:text-5xl font-bold tracking-tight text-slate-800">
                    AI &amp; Data Science Research Sandbox
                  </h2>
                  <p className="text-slate-600 max-w-2xl mx-auto text-base md:text-lg leading-relaxed">
                    Welcome to my active project index. This website acts as a visual interface to explore notebook outputs, Matplotlib plots, live web metrics, and interactive algorithms.
                  </p>
                </div>

                {/* Walkthrough Layout Guide */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-6">
                  
                  {/* Card 1: Visuals */}
                  <div 
                    onClick={() => setCurrentPage('Visuals')}
                    className="glass-card p-6 md:p-8 rounded-3xl border border-emerald-500/20 hover:border-emerald-500/40 cursor-pointer shadow-sm text-left group hover:scale-[1.01] transition-all"
                  >
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-4 text-emerald-600">
                      <BarChart2 className="w-5 h-5" />
                    </div>
                    <h3 className="font-serif text-xl font-bold text-slate-800 mb-2 group-hover:text-emerald-600 transition-colors">
                      Data Visuals &amp; Dashboards
                    </h3>
                    <p className="text-slate-600 text-sm leading-relaxed mb-4">
                      Exploratory findings, correlation matrices, and custom data network nodes. Includes fully interactive ECharts distributions and D3 networks.
                    </p>
                    <div className="flex items-center gap-1 text-xs font-semibold text-emerald-600 mt-auto">
                      <span>Explore Visuals</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </div>
                  </div>

                  {/* Card 2: ML */}
                  <div 
                    onClick={() => setCurrentPage('ML')}
                    className="glass-card p-6 md:p-8 rounded-3xl border border-indigo-500/20 hover:border-indigo-500/40 cursor-pointer shadow-sm text-left group hover:scale-[1.01] transition-all"
                  >
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center mb-4 text-indigo-600">
                      <Binary className="w-5 h-5" />
                    </div>
                    <h3 className="font-serif text-xl font-bold text-slate-800 mb-2 group-hover:text-indigo-600 transition-colors">
                      Machine Learning
                    </h3>
                    <p className="text-slate-600 text-sm leading-relaxed mb-4">
                      Predictive regression modeling, feature imputations, and parameter searches. Linked to live dashboard applications.
                    </p>
                    <div className="flex items-center gap-1 text-xs font-semibold text-indigo-600 mt-auto">
                      <span>Explore ML Models</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </div>
                  </div>

                  {/* Card 3: Deep Learning / AI */}
                  <div 
                    onClick={() => setCurrentPage('AI')}
                    className="glass-card p-6 md:p-8 rounded-3xl border border-rose-500/20 hover:border-rose-500/40 cursor-pointer shadow-sm text-left group hover:scale-[1.01] transition-all"
                  >
                    <div className="w-10 h-10 rounded-xl bg-rose-500/10 flex items-center justify-center mb-4 text-rose-600">
                      <BookOpen className="w-5 h-5" />
                    </div>
                    <h3 className="font-serif text-xl font-bold text-slate-800 mb-2 group-hover:text-rose-600 transition-colors">
                      Deep Learning / AI
                    </h3>
                    <p className="text-slate-600 text-sm leading-relaxed mb-4">
                      Neural network training loops, convolutional segmentation layers, and MRI scan prediction mask notebooks built in PyTorch.
                    </p>
                    <div className="flex items-center gap-1 text-xs font-semibold text-rose-600 mt-auto">
                      <span>Explore Neural Nets</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </div>
                  </div>

                </div>

                {/* Visual Overview Details Banner */}
                <div className="glass-card rounded-3xl p-6 md:p-8 border border-black/5 bg-white/40 flex flex-col md:flex-row items-center gap-6 text-left">
                  <div className="w-12 h-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-600 shrink-0">
                    <Sparkles className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="font-serif text-lg font-bold text-slate-800 mb-1">
                      Interactive Code Integration
                    </h4>
                    <p className="text-slate-600 text-sm leading-relaxed">
                      Every project features an integrated Jupyter Notebook viewer. You can click on the projects inside the sections to see python code alongside parsed output tables, matplotlib figures, and zoomable Plotly JSON charts.
                    </p>
                  </div>
                </div>
              </motion.div>
            )}

            {/* 2. DATA VISUALS & DASHBOARDS PAGE */}
            {currentPage === 'Visuals' && (
              <motion.div
                key="visuals"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-8"
              >
                {/* Visuals Intro Header */}
                <div className="text-left max-w-2xl mb-8">
                  <h2 className="font-serif text-2xl md:text-3xl font-bold text-slate-800">
                    Data Visuals &amp; Dashboards
                  </h2>
                  <p className="text-slate-600 text-sm md:text-base mt-1">
                    Exploratory analysis sandbox. Use the interactive D3 nodes to explore data connections, and click on projects below to study Jupyter findings.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  {/* Left Column: Interactive D3 Network Node (Grid Span 5) */}
                  <div className="lg:col-span-5">
                    <D3Network />
                  </div>
                  {/* Right Column: Visuals Catalog Ledger (Grid Span 7) */}
                  <div className="lg:col-span-7 grid grid-cols-1 md:grid-cols-2 gap-6">
                    {projectsData
                      .filter((p) => p.category === 'EDA')
                      .map((project) => (
                        <ProjectCard 
                          key={project.id} 
                          project={project} 
                          onOpenNotebook={() => setSelectedProject(project)}
                        />
                      ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* 3. MACHINE LEARNING PAGE */}
            {currentPage === 'ML' && (
              <motion.div
                key="ml"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-8"
              >
                {/* ML Intro Header */}
                <div className="text-left max-w-2xl mb-8">
                  <h2 className="font-serif text-2xl md:text-3xl font-bold text-slate-800">
                    Machine Learning Models
                  </h2>
                  <p className="text-slate-600 text-sm md:text-base mt-1">
                    Regression and classification estimators. Explore xgboost regression curves, and launch streamlit dashboards to deploy predicting inputs.
                  </p>
                </div>

                {/* ML Grid Catalog */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {projectsData
                    .filter((p) => p.category === 'ML')
                    .map((project) => (
                      <ProjectCard 
                        key={project.id} 
                        project={project} 
                        onOpenNotebook={() => setSelectedProject(project)}
                      />
                    ))}
                </div>
              </motion.div>
            )}

            {/* 4. DEEP LEARNING / AI PAGE */}
            {currentPage === 'AI' && (
              <motion.div
                key="ai"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-8"
              >
                {/* AI Intro Header */}
                <div className="text-left max-w-2xl mb-8">
                  <h2 className="font-serif text-2xl md:text-3xl font-bold text-slate-800">
                    Deep Learning &amp; Cognitive AI
                  </h2>
                  <p className="text-slate-600 text-sm md:text-base mt-1">
                    Neural network weights training logs. Examine PyTorch CNN configurations and inspect model convergence losses.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  {/* Left Column: Live CNN convergence ECharts (Grid Span 5) */}
                  <div className="lg:col-span-5">
                    <EChartsCard 
                      title="CNN Model Loss & Accuracy (Live Train Curve)" 
                      option={trainingChartOption} 
                      height="260px"
                    />
                  </div>
                  {/* Right Column: AI Catalog Ledger (Grid Span 7) */}
                  <div className="lg:col-span-7 grid grid-cols-1 md:grid-cols-2 gap-6">
                    {projectsData
                      .filter((p) => p.category === 'AI')
                      .map((project) => (
                        <ProjectCard 
                          key={project.id} 
                          project={project} 
                          onOpenNotebook={() => setSelectedProject(project)}
                        />
                      ))}
                  </div>
                </div>
              </motion.div>
            )}

          </AnimatePresence>
        </div>

      </div>

      {/* Jupyter Notebook Interactive Details Modal Overlay */}
      <AnimatePresence>
        {selectedProject && (
          <NotebookViewer
            notebookUrl={selectedProject.notebookUrl}
            title={selectedProject.title}
            category={selectedProject.category}
            onClose={() => setSelectedProject(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
