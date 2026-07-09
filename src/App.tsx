import { useState, useEffect, useMemo } from 'react';
import type { SVGProps } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Sparkles, ChevronRight, BookOpen, Binary, BarChart2, Cpu } from 'lucide-react';
import * as echarts from 'echarts';
import { NeuralBackground } from './components/NeuralBackground';
import { EChartsCard } from './components/EChartsCard';
import { D3Network } from './components/D3Network';
import { ProjectCard } from './components/ProjectCard';
import { NotebookViewer } from './components/NotebookViewer';
import { GradientDescent } from './components/GradientDescent';
import { TelemetryDashboard } from './components/TelemetryDashboard';
import { MegaProjects } from './components/MegaProjects';
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
  const [currentPage, setCurrentPage] = useState<'Home' | 'Visuals' | 'ML' | 'AI' | 'MegaProjects'>('Home');
  const [selectedProject, setSelectedProject] = useState<ProjectData | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [currentSlide, setCurrentSlide] = useState(0);

  // Sort projects by date (descending)
  const sortedProjects = useMemo(() => {
    return [...projectsData].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  }, []);

  // Update mouse position for spotlight cursor glow
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Spotlight color shifts based on selected tab (Light Theme colors - Bolder)
  const getGlowColor = () => {
    switch (currentPage) {
      case 'Visuals': return 'radial-gradient(circle, rgba(16, 185, 129, 0.32) 0%, rgba(16, 185, 129, 0) 70%)';
      case 'ML': return 'radial-gradient(circle, rgba(99, 102, 241, 0.32) 0%, rgba(99, 102, 241, 0) 70%)';
      case 'AI': return 'radial-gradient(circle, rgba(244, 63, 94, 0.32) 0%, rgba(244, 63, 94, 0) 70%)';
      case 'MegaProjects': return 'radial-gradient(circle, rgba(168, 85, 247, 0.32) 0%, rgba(168, 85, 247, 0) 70%)';
      default: return 'radial-gradient(circle, rgba(168, 85, 247, 0.32) 0%, rgba(168, 85, 247, 0) 70%)';
    }
  };

  // ECharts training configuration (memoized to avoid re-creation on mouse-move re-renders)
  const trainingChartOption: echarts.EChartsOption = useMemo(() => ({
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
  }), []);

  // ECharts radar competence skill matrix configuration (memoized)
  const skillsChartOption: echarts.EChartsOption = useMemo(() => ({
    tooltip: {
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      textStyle: { color: '#1E293B' }
    },
    radar: {
      indicator: [
        { name: 'Deep Learning (PyTorch)', max: 100 },
        { name: 'ML Modeling (Scikit-Learn)', max: 100 },
        { name: 'Data Visuals (D3/ECharts)', max: 100 },
        { name: 'Math & Stats (Pandas/NumPy)', max: 100 },
        { name: 'MLOps (Git/CI-CD)', max: 100 },
        { name: 'Python Core Architecture', max: 100 }
      ],
      axisName: {
        color: '#475569',
        fontFamily: 'Outfit, sans-serif',
        fontSize: 10,
        fontWeight: 'bold'
      },
      splitArea: {
        areaStyle: {
          color: [
            'rgba(168, 85, 247, 0.01)',
            'rgba(99, 102, 241, 0.03)',
            'rgba(6, 182, 212, 0.01)',
            'rgba(16, 185, 129, 0.03)',
            'rgba(244, 63, 94, 0.01)'
          ]
        }
      },
      splitLine: {
        lineStyle: { color: 'rgba(0, 0, 0, 0.05)' }
      },
      axisLine: {
        lineStyle: { color: 'rgba(0, 0, 0, 0.05)' }
      }
    },
    series: [
      {
        name: 'Skill Competence',
        type: 'radar',
        data: [
          {
            value: [95, 90, 88, 92, 85, 94],
            name: 'Competence Rate (%)',
            itemStyle: { color: '#A855F7' },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 1, 1, [
                { offset: 0, color: 'rgba(168, 85, 247, 0.4)' },
                { offset: 1, color: 'rgba(244, 63, 94, 0.4)' }
              ])
            },
            lineStyle: { width: 2 }
          }
        ]
      }
    ]
  }), []);

  // ECharts model performance benchmark configuration (memoized)
  const modelPerformanceChartOption: echarts.EChartsOption = useMemo(() => ({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      textStyle: { color: '#1E293B' }
    },
    legend: {
      data: ['Accuracy (%)', 'Latency (ms)'],
      textStyle: { color: '#64748B', fontFamily: 'Outfit, sans-serif' },
      bottom: '0%'
    },
    grid: { left: '3%', right: '4%', top: '10%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'value' as const,
      boundaryGap: [0, 0.01],
      axisLabel: { color: '#64748B' },
      splitLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.03)' } }
    },
    yAxis: {
      type: 'category' as const,
      data: ['Random Forest', 'XGBoost', 'LightGBM', 'CNN', 'Transformer'],
      axisLabel: { color: '#64748B', fontWeight: 'bold' },
      axisLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.06)' } }
    },
    series: [
      {
        name: 'Accuracy (%)',
        type: 'bar' as const,
        data: [82.4, 89.1, 91.2, 94.6, 98.2],
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: 'rgba(168, 85, 247, 0.6)' },
            { offset: 1, color: '#A855F7' }
          ]),
          borderRadius: [0, 4, 4, 0]
        }
      },
      {
        name: 'Latency (ms)',
        type: 'bar' as const,
        data: [1.2, 2.5, 1.8, 12.4, 45.0],
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: 'rgba(244, 63, 94, 0.6)' },
            { offset: 1, color: '#F43F5E' }
          ]),
          borderRadius: [0, 4, 4, 0]
        }
      }
    ]
  }), []);

  // ECharts feature correlation matrix configuration (memoized)
  const correlationMatrixOption: echarts.EChartsOption = useMemo(() => ({
    tooltip: {
      position: 'top',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: 'rgba(0, 0, 0, 0.08)',
      textStyle: { color: '#1E293B' }
    },
    grid: { height: '65%', top: '10%', bottom: '20%' },
    xAxis: {
      type: 'category' as const,
      data: ['Age', 'Income', 'Score', 'Rating', 'Sales'],
      splitArea: { show: true },
      axisLabel: { color: '#64748B', fontWeight: 'bold' },
      axisLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.06)' } }
    },
    yAxis: {
      type: 'category' as const,
      data: ['Age', 'Income', 'Score', 'Rating', 'Sales'],
      splitArea: { show: true },
      axisLabel: { color: '#64748B', fontWeight: 'bold' },
      axisLine: { lineStyle: { color: 'rgba(0, 0, 0, 0.06)' } }
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '0%',
      inRange: {
        color: ['#F43F5E', '#FFFFFF', '#10B981']
      },
      textStyle: { color: '#64748B' }
    },
    series: [
      {
        name: 'Correlation',
        type: 'heatmap' as const,
        data: [
          [0, 0, 1.0], [0, 1, 0.42], [0, 2, -0.15], [0, 3, 0.08], [0, 4, 0.31],
          [1, 0, 0.42], [1, 1, 1.0], [1, 2, 0.65], [1, 3, 0.23], [1, 4, 0.72],
          [2, 0, -0.15], [2, 1, 0.65], [2, 2, 1.0], [2, 3, 0.54], [2, 4, 0.48],
          [3, 0, 0.08], [3, 1, 0.23], [3, 2, 0.54], [3, 3, 1.0], [3, 4, 0.15],
          [4, 0, 0.31], [4, 1, 0.72], [4, 2, 0.48], [4, 3, 0.15], [4, 4, 1.0]
        ],
        label: {
          show: true,
          formatter: (params: any) => {
            return params.data[2].toFixed(2);
          },
          color: '#1E293B'
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.15)'
          }
        }
      }
    ]
  }), []);

  const carouselSlides = [
    {
      title: "Core Skill Matrix",
      description: "Live ECharts radar showing competence stats across deep learning, PyTorch, Pandas, MLOps, and architecture.",
      element: <EChartsCard title="Core Skill Matrix (Radar Dist)" option={skillsChartOption} height="350px" />
    },
    {
      title: "GPU Training Telemetry",
      description: "Real-time hardware allocation, temperature metrics, and live-streaming neural network epoch loss updates.",
      element: <TelemetryDashboard />
    },
    {
      title: "Gradient Descent Optimizer",
      description: "Interactive visual simulator demonstrating parameter updates and convergence on mathematical error gradients.",
      element: <GradientDescent />
    },
    {
      title: "Feature Correlation Matrix",
      description: "ECharts 2D correlation matrix showing linear relationship coefficients between dataset features.",
      element: <EChartsCard title="Feature Correlation Heatmap" option={correlationMatrixOption} height="350px" />
    },
    {
      title: "Model Performance Benchmarks",
      description: "Performance comparative bar chart mapping accuracy scores against execution latency for ML models.",
      element: <EChartsCard title="Model Performance Benchmarks" option={modelPerformanceChartOption} height="350px" />
    }
  ];

  return (
    <div className="relative min-h-screen selection:bg-purple-500/20 selection:text-slate-900 text-slate-700 pb-6">
      
      {/* Interactive Cursor Spotlight Glow */}
      <div 
        className="cursor-glow"
        style={{
          left: mousePos.x,
          top: mousePos.y,
          background: getGlowColor()
        }}
      />

      {/* Background Graphic elements wrapper to prevent empty footer/scroll overflow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
        {/* Layered background patterns (separated to prevent CSS background-image override cascade) */}
        <div className="absolute inset-0 bg-blueprint" />
        <div className="absolute inset-0 bg-grid-lines" />
        <div className="absolute inset-0 bg-topo" />
        <div className="absolute inset-0 bg-crosshatch" />

        {/* Dynamic Animated Bright Gradient Blobs */}
        <div className="absolute top-[-10%] left-[-5%] w-[650px] h-[650px] blob blob-purple opacity-55" />
        <div className="absolute bottom-[20%] right-[-10%] w-[750px] h-[750px] blob blob-indigo opacity-50" />
        <div className="absolute top-[40%] left-[30%] w-[650px] h-[650px] blob blob-pink opacity-45" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[700px] h-[700px] blob blob-cyan opacity-45" />
        <div className="absolute top-[15%] right-[20%] w-[500px] h-[500px] blob blob-purple opacity-35" />
        <div className="absolute top-[65%] right-[5%] w-[550px] h-[550px] blob blob-pink opacity-30" />

        {/* Decorative Floating Blueprint Geometry */}
        <svg className="absolute w-[500px] h-[500px] text-indigo-500/15 animate-spin-slow top-[5%] right-[-8%]" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="45" stroke="currentColor" strokeWidth="0.4" fill="none" strokeDasharray="3 3" />
          <circle cx="50" cy="50" r="30" stroke="currentColor" strokeWidth="0.4" fill="none" />
          <circle cx="50" cy="50" r="15" stroke="currentColor" strokeWidth="0.3" fill="none" />
          <path d="M 5 50 L 95 50 M 50 5 L 50 95" stroke="currentColor" strokeWidth="0.2" />
        </svg>
        
        <svg className="absolute w-[400px] h-[400px] text-pink-500/12 animate-float-slow bottom-[8%] left-[-5%]" viewBox="0 0 100 100">
          <polygon points="50,5 95,95 5,95" stroke="currentColor" strokeWidth="0.4" fill="none" strokeDasharray="2 2" />
          <circle cx="50" cy="65" r="18" stroke="currentColor" strokeWidth="0.4" fill="none" />
          <line x1="50" y1="5" x2="50" y2="95" stroke="currentColor" strokeWidth="0.2" />
        </svg>

        {/* Additional geometric decorations for mid-page visual interest */}
        <svg className="absolute w-[350px] h-[350px] text-emerald-500/10 animate-spin-slow top-[50%] left-[-6%]" style={{animationDirection: 'reverse', animationDuration: '45s'}} viewBox="0 0 100 100">
          <rect x="15" y="15" width="70" height="70" stroke="currentColor" strokeWidth="0.35" fill="none" strokeDasharray="4 3" />
          <rect x="25" y="25" width="50" height="50" stroke="currentColor" strokeWidth="0.3" fill="none" />
          <circle cx="50" cy="50" r="20" stroke="currentColor" strokeWidth="0.25" fill="none" />
        </svg>

        <svg className="absolute w-[300px] h-[300px] text-purple-500/10 animate-float-slow top-[30%] right-[3%]" viewBox="0 0 100 100">
          <polygon points="50,10 90,35 90,75 50,95 10,75 10,35" stroke="currentColor" strokeWidth="0.35" fill="none" strokeDasharray="3 2" />
          <circle cx="50" cy="52" r="22" stroke="currentColor" strokeWidth="0.3" fill="none" />
        </svg>
      </div>

      {/* Canvas Neural Background */}
      <NeuralBackground />

      {/* Sticky top Navigation Links — sits OUTSIDE the content container so it sticks to the full viewport */}
      <div className="sticky top-0 z-50 w-full select-none pointer-events-none py-3">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-center">
          <div className="flex gap-1.5 bg-white/45 p-1.5 rounded-2xl border border-white/70 shadow-lg backdrop-blur-xl relative pointer-events-auto">
            {([
              { key: 'Home', label: 'Overview' },
              { key: 'Visuals', label: 'Data Visuals & Dashboards' },
              { key: 'ML', label: 'Machine Learning' },
              { key: 'AI', label: 'Deep Learning / AI' },
              { key: 'MegaProjects', label: 'Mega Projects' }
            ] as const).map((tab) => {
              const isActive = currentPage === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setCurrentPage(tab.key)}
                  className={`relative px-4 py-2.5 rounded-xl text-sm md:text-base font-bold transition-all select-none z-10 ${
                    isActive 
                      ? 'text-white' 
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {isActive && (
                    <motion.div 
                      layoutId="activeNavbarTab"
                      className="absolute inset-0 bg-gradient-to-tr from-purple-500 to-pink-500 rounded-xl shadow shadow-purple-500/15 z-[-1]"
                      transition={{ type: "spring", stiffness: 350, damping: 28 }}
                    />
                  )}
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4">
        
        {/* Navigation / Header */}
        <header className="flex flex-col md:flex-row justify-between items-center px-8 py-4 bg-white/40 border border-white/60 backdrop-blur-md shadow-sm rounded-3xl mb-8">
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
                className="space-y-20 w-full text-left"
              >
                {/* Stylized Profile Hero (Creative, Free-Flow) */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center pt-4 pb-6 border-b border-slate-200/40">
                  <div className="lg:col-span-4 space-y-4">
                    <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 text-sm font-semibold tracking-wide select-none">
                      <Cpu className="w-4 h-4" /> DATA SCIENTIST &amp; AI SYSTEMS
                    </div>

                    <div className="space-y-1">
                      <h2 className="font-serif text-5xl md:text-6xl text-slate-800 font-normal tracking-tight leading-none">Shivansh</h2>
                      <h2 className="font-serif text-5xl md:text-6xl italic font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-700 via-purple-750 to-rose-600 pb-1">
                        Kandwal<span className="text-slate-900 font-normal">.</span>
                      </h2>
                    </div>
                  </div>

                  <div className="lg:col-span-8 lg:border-l lg:border-slate-200/60 lg:pl-8">
                    <p className="text-slate-650 text-base md:text-lg leading-relaxed font-medium">
                      Welcome to my active project index. This website acts as a visual computational sandbox where you can explore Python notebook outputs, interactive algorithms, and live cluster telemetry. Use the navigation links above to browse through specific modules.
                    </p>
                  </div>
                </div>

                {/* Center Stage: Creative Information Portal (Free-Flow, borderless) */}
                <div className="w-full my-16 text-center space-y-8 select-none">
                  <div className="inline-flex items-center gap-2 text-purple-600 text-xs font-bold uppercase tracking-widest bg-purple-500/10 px-3 py-1 rounded-full border border-purple-500/20">
                    <Sparkles className="w-3.5 h-3.5" />
                    Computational Sandbox Overview
                  </div>
                  
                  <h3 className="font-serif text-4xl md:text-5xl font-normal text-slate-800 leading-tight">
                    Explore the computational layers of <span className="italic font-semibold text-purple-650">data science</span>.
                  </h3>
                  
                  <p className="text-slate-500 text-base md:text-lg max-w-3xl mx-auto leading-relaxed">
                    This portfolio operates as an active research index. Instead of static mockups, the pages render live simulations, GPU cluster metrics, and actual Jupyter outputs.
                  </p>

                  <div className="max-w-4xl mx-auto flex flex-col gap-4 pt-10 border-t border-slate-200/60 text-left">
                    <div className="flex gap-4 items-start p-5 bg-white/20 hover:bg-white/40 border border-black/5 rounded-2xl transition-all shadow-sm">
                      <div className="w-10 h-10 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 flex items-center justify-center font-mono font-bold shrink-0">
                        01
                      </div>
                      <div className="space-y-1">
                        <span className="text-xs font-mono font-bold text-emerald-600 tracking-wider uppercase block">Live Simulation</span>
                        <h4 className="font-serif text-lg font-bold text-slate-800">Dynamic Sandbox</h4>
                        <p className="text-sm text-slate-550 leading-relaxed">
                          Drag feedforward nodes in the centered neural network or tweak learning rates below to observe weights descent optimization.
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-4 items-start p-5 bg-white/20 hover:bg-white/40 border border-black/5 rounded-2xl transition-all shadow-sm">
                      <div className="w-10 h-10 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-700 flex items-center justify-center font-mono font-bold shrink-0">
                        02
                      </div>
                      <div className="space-y-1">
                        <span className="text-xs font-mono font-bold text-indigo-600 tracking-wider uppercase block">Code Integration</span>
                        <h4 className="font-serif text-lg font-bold text-slate-800">Notebook Pipeline</h4>
                        <p className="text-sm text-slate-550 leading-relaxed">
                          Click any project card to view actual Python notebook source files integrated with Plotly interactive charts and parsed dataframes.
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-4 items-start p-5 bg-white/20 hover:bg-white/40 border border-black/5 rounded-2xl transition-all shadow-sm">
                      <div className="w-10 h-10 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-700 flex items-center justify-center font-mono font-bold shrink-0">
                        03
                      </div>
                      <div className="space-y-1">
                        <span className="text-xs font-mono font-bold text-rose-600 tracking-wider uppercase block">Performance Benchmarks</span>
                        <h4 className="font-serif text-lg font-bold text-slate-800">Active Telemetry</h4>
                        <p className="text-sm text-slate-555 leading-relaxed">
                          Observe live cluster telemetry simulating VRAM, Core Temperature, and epoch training steps to understand models benchmarks.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Center Stage: Grand Interactive Neural Network */}
                <div className="w-full mx-auto flex flex-col items-center justify-center my-6">
                  <D3Network />
                </div>

                {/* Lab Dashboard Slides Carousel */}
                <div id="carousel-section" className="w-full my-12 relative">
                  <div className="relative glass-card rounded-3xl p-8 border border-black/5 bg-white/40 shadow-sm">
                    {/* Carousel Title & Navigation Header */}
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6 border-b border-slate-200/40 pb-5">
                      <div className="text-left">
                        <span className="text-xs font-bold uppercase tracking-wider text-purple-650 bg-purple-500/10 px-3 py-1 rounded-full select-none">
                          Lab Widget {currentSlide + 1} of {carouselSlides.length}
                        </span>
                        <h3 className="font-serif text-2xl font-bold text-slate-800 mt-2">
                          {carouselSlides[currentSlide].title}
                        </h3>
                        <p className="text-slate-550 text-sm mt-1 max-w-2xl leading-relaxed">
                          {carouselSlides[currentSlide].description}
                        </p>
                      </div>
                      
                      {/* Arrow Buttons */}
                      <div className="flex gap-2 shrink-0 select-none">
                        <button
                          onClick={() => setCurrentSlide(prev => (prev === 0 ? carouselSlides.length - 1 : prev - 1))}
                          className="w-10 h-10 rounded-xl bg-white/60 hover:bg-white border border-slate-200 hover:border-slate-300 text-slate-600 flex items-center justify-center transition-all cursor-pointer shadow-sm text-sm"
                          title="Previous widget"
                        >
                          ←
                        </button>
                        <button
                          onClick={() => setCurrentSlide(prev => (prev === carouselSlides.length - 1 ? 0 : prev + 1))}
                          className="w-10 h-10 rounded-xl bg-white/60 hover:bg-white border border-slate-200 hover:border-slate-300 text-slate-650 flex items-center justify-center transition-all cursor-pointer shadow-sm text-sm"
                          title="Next widget"
                        >
                          →
                        </button>
                      </div>
                    </div>

                    {/* Active Slide Wrapper with sliding motion */}
                    <div className="min-h-[400px] relative overflow-hidden flex items-center justify-center">
                      <AnimatePresence mode="wait">
                        <motion.div
                          key={currentSlide}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          transition={{ duration: 0.35, ease: "easeInOut" }}
                          className="w-full h-full"
                        >
                          {carouselSlides[currentSlide].element}
                        </motion.div>
                      </AnimatePresence>
                    </div>

                    {/* Indicator Dots */}
                    <div className="flex justify-center gap-2.5 mt-6 select-none">
                      {carouselSlides.map((_, idx) => (
                        <button
                          key={idx}
                          onClick={() => setCurrentSlide(idx)}
                          className={`w-3 h-3 rounded-full transition-all cursor-pointer ${
                            idx === currentSlide 
                              ? 'bg-purple-600 w-8' 
                              : 'bg-slate-300 hover:bg-slate-400'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Walkthrough Layout Guide with Color Segmentation & Micro-Animations */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 pt-6">
                  
                  {/* Card 1: Visuals */}
                  <motion.div 
                    onClick={() => setCurrentPage('Visuals')}
                    whileHover={{ y: -10, scale: 1.03 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    className="glass-eda p-8 rounded-3xl cursor-pointer shadow-sm text-left group flex flex-col justify-between"
                  >
                    <div>
                      <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-6 text-emerald-600 border border-emerald-500/20 shadow-sm shadow-emerald-500/5">
                        <BarChart2 className="w-6 h-6" />
                      </div>
                      <h3 className="font-serif text-2xl font-bold text-slate-800 mb-3 group-hover:text-emerald-700 transition-colors">
                        Data Visuals &amp; Dashboards
                      </h3>
                      <p className="text-slate-650 text-sm md:text-base leading-relaxed mb-8">
                        Exploratory findings, correlation matrices, and custom data network nodes. Includes fully interactive ECharts distributions and D3 networks.
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 text-sm font-semibold text-emerald-650">
                      <span>Explore Visuals</span>
                      <ChevronRight className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-1" />
                    </div>
                  </motion.div>

                  {/* Card 2: ML */}
                  <motion.div 
                    onClick={() => setCurrentPage('ML')}
                    whileHover={{ y: -10, scale: 1.03 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    className="glass-ml p-8 rounded-3xl cursor-pointer shadow-sm text-left group flex flex-col justify-between"
                  >
                    <div>
                      <div className="w-12 h-12 rounded-xl bg-indigo-500/10 flex items-center justify-center mb-6 text-indigo-600 border border-indigo-500/20 shadow-sm shadow-indigo-500/5">
                        <Binary className="w-6 h-6" />
                      </div>
                      <h3 className="font-serif text-2xl font-bold text-slate-800 mb-3 group-hover:text-indigo-700 transition-colors">
                        Machine Learning
                      </h3>
                      <p className="text-slate-650 text-sm md:text-base leading-relaxed mb-8">
                        Predictive regression modeling, feature imputations, and parameter searches. Linked to live dashboard applications.
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 text-sm font-semibold text-indigo-650">
                      <span>Explore ML Models</span>
                      <ChevronRight className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-1" />
                    </div>
                  </motion.div>

                  {/* Card 3: Deep Learning / AI */}
                  <motion.div 
                    onClick={() => setCurrentPage('AI')}
                    whileHover={{ y: -10, scale: 1.03 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    className="glass-ai p-8 rounded-3xl cursor-pointer shadow-sm text-left group flex flex-col justify-between"
                  >
                    <div>
                      <div className="w-12 h-12 rounded-xl bg-rose-500/10 flex items-center justify-center mb-6 text-rose-600 border border-rose-500/20 shadow-sm shadow-rose-500/5">
                        <BookOpen className="w-6 h-6" />
                      </div>
                      <h3 className="font-serif text-2xl font-bold text-slate-800 mb-3 group-hover:text-rose-700 transition-colors">
                        Deep Learning / AI
                      </h3>
                      <p className="text-slate-650 text-sm md:text-base leading-relaxed mb-8">
                        Neural network training loops, convolutional segmentation layers, and MRI scan prediction mask notebooks built in PyTorch.
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 text-sm font-semibold text-rose-650">
                      <span>Explore Neural Nets</span>
                      <ChevronRight className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-1" />
                    </div>
                  </motion.div>

                </div>

                {/* Visual Overview Details Banner */}
                <div className="glass-card rounded-3xl p-8 border border-black/5 bg-white/40 flex flex-col md:flex-row items-center gap-6 text-left">
                  <div className="w-12 h-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-650 shrink-0">
                    <Sparkles className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="font-serif text-xl font-bold text-slate-800 mb-1">
                      Interactive Code Integration
                    </h4>
                    <p className="text-slate-650 text-base leading-relaxed">
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
                className="space-y-12"
              >
                {/* Visuals Intro Header */}
                <div className="text-left max-w-2xl">
                  <h2 className="font-serif text-2xl md:text-3xl font-bold text-slate-800">
                    Data Visuals &amp; Dashboards
                  </h2>
                  <p className="text-slate-600 text-sm md:text-base mt-1">
                    Exploratory analysis sandbox. Use the interactive correlation heatmap and browse projects below to study Jupyter findings.
                  </p>
                </div>

                {/* Top Row: Chart & Description */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  <div className="lg:col-span-8">
                    <EChartsCard title="Feature Correlation Heatmap" option={correlationMatrixOption} height="360px" />
                  </div>
                  <div className="lg:col-span-4 glass-card rounded-3xl p-6 border border-black/5 bg-white/40 flex flex-col justify-center text-left">
                    <h3 className="font-serif text-xl font-bold text-slate-800 mb-3">Dimensionality &amp; Associations</h3>
                    <p className="text-slate-650 text-sm leading-relaxed mb-4">
                      Feature engineering requires finding relationships between variables. This heatmap displays Pearson correlation coefficients across standard dataset attributes.
                    </p>
                    <div className="text-xs font-mono text-slate-400">
                      * Colored bounds highlight positive/negative associations.
                    </div>
                  </div>
                </div>

                {/* Bottom Row: Full Width Projects Grid */}
                <div className="space-y-6">
                  <div className="border-b border-slate-200/60 pb-3 text-left">
                    <h3 className="font-serif text-2xl font-bold text-slate-800">Project Catalog</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {sortedProjects
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
                className="space-y-12"
              >
                {/* ML Intro Header */}
                <div className="text-left max-w-2xl">
                  <h2 className="font-serif text-2xl md:text-3xl font-bold text-slate-800">
                    Machine Learning Models
                  </h2>
                  <p className="text-slate-600 text-sm md:text-base mt-1">
                    Predictive regression modeling, feature imputations, and parameter searches. Study model benchmarks and browse projects below to view Jupyter outputs.
                  </p>
                </div>

                {/* Top Row: Chart & Description */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  <div className="lg:col-span-8">
                    <EChartsCard title="Model Performance Benchmarks" option={modelPerformanceChartOption} height="360px" />
                  </div>
                  <div className="lg:col-span-4 glass-card rounded-3xl p-6 border border-black/5 bg-white/40 flex flex-col justify-center text-left">
                    <h3 className="font-serif text-xl font-bold text-slate-800 mb-3">Benchmark Metrics</h3>
                    <p className="text-slate-650 text-sm leading-relaxed mb-4">
                      Comparing latency against overall precision scores across estimators. Trees models show high accuracy while maintaining quick inference speed.
                    </p>
                    <div className="text-xs font-mono text-slate-400">
                      * Values represent cross-validated telemetry averages.
                    </div>
                  </div>
                </div>

                {/* Bottom Row: Full Width Projects Grid */}
                <div className="space-y-6">
                  <div className="border-b border-slate-200/60 pb-3 text-left">
                    <h3 className="font-serif text-2xl font-bold text-slate-800">Project Catalog</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {sortedProjects
                      .filter((p) => p.category === 'ML')
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

            {/* 4. DEEP LEARNING / AI PAGE */}
            {currentPage === 'AI' && (
              <motion.div
                key="ai"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-12"
              >
                {/* AI Intro Header */}
                <div className="text-left max-w-2xl">
                  <h2 className="font-serif text-2xl md:text-3xl font-bold text-slate-800">
                    Deep Learning &amp; Cognitive AI
                  </h2>
                  <p className="text-slate-600 text-sm md:text-base mt-1">
                    Neural network weights training logs. Examine PyTorch CNN configurations and inspect model convergence losses.
                  </p>
                </div>

                {/* Top Row: Chart & Description */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  <div className="lg:col-span-8">
                    <EChartsCard 
                      title="CNN Model Loss & Accuracy (Live Train Curve)" 
                      option={trainingChartOption} 
                      height="300px"
                    />
                  </div>
                  <div className="lg:col-span-4 glass-card rounded-3xl p-6 border border-black/5 bg-white/40 flex flex-col justify-center text-left">
                    <h3 className="font-serif text-xl font-bold text-slate-800 mb-3">Model Convergence</h3>
                    <p className="text-slate-650 text-sm leading-relaxed mb-4">
                      Real-time training diagnostics showing categorical cross-entropy loss decaying while validation metrics scale up towards convergence.
                    </p>
                    <div className="text-xs font-mono text-slate-400">
                      * Tracks dynamic feedforward parameter weights.
                    </div>
                  </div>
                </div>

                {/* Bottom Row: Full Width Projects Grid */}
                <div className="space-y-6">
                  <div className="border-b border-slate-200/60 pb-3 text-left">
                    <h3 className="font-serif text-2xl font-bold text-slate-800">Project Catalog</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {sortedProjects
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

            {/* 5. MEGA PROJECTS PAGE */}
            {currentPage === 'MegaProjects' && (
              <motion.div
                key="megaprojects"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              >
                <MegaProjects />
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
