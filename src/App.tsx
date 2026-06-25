import { useState, useEffect } from 'react';
import type { SVGProps } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Sparkles } from 'lucide-react';
import * as echarts from 'echarts';
import { NeuralBackground } from './components/NeuralBackground';
import { TerminalDashboard } from './components/TerminalDashboard';
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


// GitHub Contribution Calendar Simulation Data
const generateContributionGrid = () => {
  const grid = [];
  const levels = [0, 1, 2, 3, 4];
  // 52 columns, 7 rows for a standard calendar representation
  for (let col = 0; col < 32; col++) {
    const column = [];
    for (let row = 0; row < 7; row++) {
      // Weight higher numbers for recent columns to show active daily contribution
      const weight = col > 25 ? 0.8 : 0.45;
      const level = Math.random() < weight ? levels[Math.floor(Math.random() * 4) + 1] : 0;
      column.push(level);
    }
    grid.push(column);
  }
  return grid;
};

const contributionGrid = generateContributionGrid();

export default function App() {
  const [activeSection, setActiveSection] = useState<'All' | 'EDA' | 'ML' | 'AI'>('All');
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

  // Filter projects by category
  const filteredProjects = projectsData.filter((project) => {
    if (activeSection === 'All') return true;
    return project.category === activeSection;
  });

  // Spotlight color shifts based on selected tab
  const getGlowColor = () => {
    switch (activeSection) {
      case 'EDA': return 'radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(16, 185, 129, 0) 70%)';
      case 'ML': return 'radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0) 70%)';
      case 'AI': return 'radial-gradient(circle, rgba(244, 63, 94, 0.25) 0%, rgba(244, 63, 94, 0) 70%)';
      default: return 'radial-gradient(circle, rgba(168, 85, 247, 0.25) 0%, rgba(168, 85, 247, 0) 70%)';
    }
  };

  // ECharts training configuration
  const trainingChartOption: echarts.EChartsOption = {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(11, 8, 22, 0.9)',
      borderColor: 'rgba(255, 255, 255, 0.08)',
      textStyle: { color: '#F8FAFC' }
    },
    legend: {
      data: ['Training Loss', 'Validation Accuracy'],
      textStyle: { color: '#94A3B8', fontFamily: 'Outfit, sans-serif' },
      bottom: '0%'
    },
    grid: { left: '3%', right: '4%', top: '10%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category' as const,
      boundaryGap: false,
      data: ['Epoch 20', 'Epoch 40', 'Epoch 60', 'Epoch 80', 'Epoch 100', 'Epoch 120'],
      axisLabel: { color: '#94A3B8' },
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } }
    },
    yAxis: [
      {
        type: 'value' as const,
        name: 'Loss',
        position: 'left',
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
        axisLabel: { color: '#94A3B8' }
      },
      {
        type: 'value' as const,
        name: 'Accuracy',
        position: 'right',
        max: 100,
        splitLine: { show: false },
        axisLabel: { color: '#94A3B8' }
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
            { offset: 0, color: 'rgba(244, 63, 94, 0.2)' },
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
            { offset: 0, color: 'rgba(16, 185, 129, 0.2)' },
            { offset: 1, color: 'rgba(16, 185, 129, 0)' }
          ])
        }
      }
    ]
  };


  return (
    <div className="relative min-h-screen overflow-hidden selection:bg-purple-500/30 selection:text-white font-sans text-slate-200 pb-20">
      
      {/* Interactive Cursor Spotlight Glow */}
      <div 
        className="cursor-glow"
        style={{
          left: mousePos.x,
          top: mousePos.y,
          background: getGlowColor()
        }}
      />

      {/* Dynamic Animated Gradient Blobs */}
      <div className="absolute top-[-10%] left-[-5%] w-[600px] h-[600px] blob blob-purple opacity-40 z-0" />
      <div className="absolute bottom-[20%] right-[-10%] w-[700px] h-[700px] blob blob-indigo opacity-35 z-0" />
      <div className="absolute top-[40%] left-[30%] w-[600px] h-[600px] blob blob-pink opacity-30 z-0" />
      <div className="absolute bottom-[-10%] left-[-10%] w-[650px] h-[650px] blob blob-cyan opacity-25 z-0" />

      {/* Canvas Neural Background */}
      <NeuralBackground />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        
        {/* Navigation / Header */}
        <header className="flex flex-col md:flex-row justify-between items-center py-6 border-b border-white/5 mb-12">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-serif text-2xl md:text-3xl font-bold tracking-tight text-white m-0">
                Shivansh Kandwal
              </h1>
              <p className="text-xs md:text-sm text-slate-400 font-mono tracking-wide mt-0.5">
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
              className="p-2.5 rounded-xl bg-white/5 border border-white/5 hover:border-white/15 text-slate-400 hover:text-white hover:bg-white/10 transition-all"
            >
              <GithubIcon className="w-4.5 h-4.5" />
            </a>
            <a 
              href="https://linkedin.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2.5 rounded-xl bg-white/5 border border-white/5 hover:border-white/15 text-slate-400 hover:text-white hover:bg-white/10 transition-all"
            >
              <LinkedinIcon className="w-4.5 h-4.5" />
            </a>
            <a 
              href="mailto:contact@shivanshkandwal.com" 
              className="p-2.5 rounded-xl bg-white/5 border border-white/5 hover:border-white/15 text-slate-400 hover:text-white hover:bg-white/10 transition-all"
            >
              <Mail className="w-4.5 h-4.5" />
            </a>
          </div>
        </header>

        {/* Dashboard Grid Layout */}
        <main className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT SIDEBAR: Visual Models & Terminal Control (Grid Span 5) */}
          <section className="lg:col-span-5 space-y-8">
            
            {/* Live Model Training Metrics (ECharts) */}
            <EChartsCard 
              title="CNN Model Loss & Accuracy (Live Train Curve)" 
              option={trainingChartOption} 
              height="240px"
            />

            {/* Neural Net Graph Layout (D3) */}
            <D3Network />

            {/* Simulated Data CLI terminal */}
            <TerminalDashboard />

          </section>

          {/* RIGHT SHOWCASE: Projects Ledger, Calendar, Activity (Grid Span 7) */}
          <section className="lg:col-span-7 space-y-8">

            {/* GitHub Contributions Sandbox */}
            <div className="glass-card rounded-3xl p-6 border border-white/5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-white/5 mb-6">
                <div>
                  <h3 className="font-serif text-lg font-medium text-slate-100">
                    Daily Contributions Sandbox
                  </h3>
                  <p className="text-xs text-slate-400 font-mono tracking-wide mt-0.5">
                    Commit activity on active GitHub Pages deployment
                  </p>
                </div>
                <div className="flex items-center gap-6 font-mono text-center">
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase block">Streak</span>
                    <span className="text-sm font-bold text-emerald-400">12 Days</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase block">Total Commits</span>
                    <span className="text-sm font-bold text-purple-400">482</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase block">Active rate</span>
                    <span className="text-sm font-bold text-pink-400">98%</span>
                  </div>
                </div>
              </div>

              {/* Grid Calendar Visual */}
              <div className="overflow-x-auto no-scrollbar">
                <div className="flex gap-1 min-w-[500px]">
                  {contributionGrid.map((column, colIdx) => (
                    <div key={colIdx} className="flex flex-col gap-1 flex-1">
                      {column.map((level, rowIdx) => {
                        const bgClass = {
                          0: 'bg-white/5',
                          1: 'bg-emerald-500/20',
                          2: 'bg-emerald-500/40',
                          3: 'bg-emerald-500/70',
                          4: 'bg-emerald-500'
                        }[level as 0 | 1 | 2 | 3 | 4];

                        return (
                          <div 
                            key={rowIdx} 
                            className={`aspect-square rounded-[2px] transition-colors duration-300 hover:scale-125 hover:z-10 cursor-pointer ${bgClass}`}
                            title={`Commits Level: ${level}`}
                          />
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-end gap-1.5 mt-3 text-[10px] font-mono text-slate-500">
                <span>Less</span>
                <span className="w-2.5 h-2.5 rounded-[1px] bg-white/5" />
                <span className="w-2.5 h-2.5 rounded-[1px] bg-emerald-500/20" />
                <span className="w-2.5 h-2.5 rounded-[1px] bg-emerald-500/40" />
                <span className="w-2.5 h-2.5 rounded-[1px] bg-emerald-500/70" />
                <span className="w-2.5 h-2.5 rounded-[1px] bg-emerald-500" />
                <span>More</span>
              </div>
            </div>

            {/* Section Controls (Filters) */}
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
              <div className="flex gap-1 bg-white/3 p-1 rounded-2xl border border-white/5 relative">
                {(['All', 'EDA', 'ML', 'AI'] as const).map((section) => {
                  const isActive = activeSection === section;
                  const label = {
                    All: 'All Sandbox',
                    EDA: 'EDA',
                    ML: 'Machine Learning',
                    AI: 'Deep Learning'
                  }[section];

                  return (
                    <button
                      key={section}
                      onClick={() => setActiveSection(section)}
                      className={`relative px-4 py-2 rounded-xl text-xs md:text-sm font-medium transition-all ${
                        isActive 
                          ? 'bg-gradient-to-tr from-purple-500/80 to-pink-500/80 text-white shadow shadow-purple-500/20' 
                          : 'text-slate-400 hover:text-slate-200 hover:bg-white/2'
                      }`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
              <span className="text-xs text-slate-500 font-mono hidden sm:inline">
                Showing {filteredProjects.length} projects
              </span>
            </div>

            {/* Projects Showcase Ledger Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <AnimatePresence mode="popLayout">
                {filteredProjects.map((project) => (
                  <motion.div
                    key={project.id}
                    layout
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                  >
                    <ProjectCard 
                      project={project} 
                      onOpenNotebook={() => setSelectedProject(project)}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

          </section>

        </main>
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
