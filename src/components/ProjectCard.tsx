import React from 'react';
import { FileCode, ExternalLink, Calendar } from 'lucide-react';
import type { ProjectData } from '../data/projects';
import { GlassCard } from './GlassCard';


interface ProjectCardProps {
  project: ProjectData;
  onOpenNotebook: () => void;
}

export const ProjectCard: React.FC<ProjectCardProps> = ({ project, onOpenNotebook }) => {
  const { title, description, category, tags, date, metrics, dashboardUrl, dashboardImage } = project;

  // Theme config based on category
  const theme = {
    EDA: {
      accent: 'text-emerald-400',
      bg: 'bg-emerald-500/10 border-emerald-500/20',
      glow: 'rgba(16, 185, 129, 0.25)',
      cardGlow: 'glass-eda'
    },
    ML: {
      accent: 'text-indigo-400',
      bg: 'bg-indigo-500/10 border-indigo-500/20',
      glow: 'rgba(99, 102, 241, 0.25)',
      cardGlow: 'glass-ml'
    },
    AI: {
      accent: 'text-rose-400',
      bg: 'bg-rose-500/10 border-rose-500/20',
      glow: 'rgba(244, 63, 94, 0.25)',
      cardGlow: 'glass-ai'
    }
  }[category];

  return (
    <GlassCard 
      className={`flex flex-col justify-between h-full text-left ${theme.cardGlow}`}
      shadowColor={theme.glow}
    >
      <div className="space-y-4">
        {/* Category & Date */}
        <div className="flex items-center justify-between text-xs">
          <span className={`px-2.5 py-1 rounded-full font-mono font-semibold tracking-wider uppercase border ${theme.accent} ${theme.bg}`}>
            {category === 'AI' ? 'Deep Learning' : category}
          </span>
          <div className="flex items-center gap-1.5 text-slate-400 font-mono">
            <Calendar className="w-3.5 h-3.5" />
            <span>{date}</span>
          </div>
        </div>

        {/* Dashboard Image Preview */}
        {dashboardImage && (
          <div className="relative group/img rounded-2xl overflow-hidden aspect-[16/9] border border-white/5 bg-slate-950/40">
            <img 
              src={dashboardImage} 
              alt={`${title} Preview`} 
              className="w-full h-full object-cover transition-transform duration-500 group-hover/img:scale-105"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent flex items-end p-4">
              <span className="text-[10px] font-mono text-slate-300 bg-black/60 px-2 py-0.5 rounded backdrop-blur">
                Click to explore analysis
              </span>
            </div>
          </div>
        )}

        {/* Title & Description */}
        <div className="space-y-2">
          <h3 className="font-serif text-xl font-semibold text-slate-100 group-hover:text-white leading-tight">
            {title}
          </h3>
          <p className="text-slate-400 text-sm leading-relaxed font-sans line-clamp-3">
            {description}
          </p>
        </div>

        {/* Key Metrics Dashboard Summary */}
        <div className="grid grid-cols-3 gap-2 bg-white/2 p-3 rounded-2xl border border-white/5">
          {Object.entries(metrics).map(([key, val]) => (
            <div key={key} className="flex flex-col text-center justify-center">
              <span className="text-[9px] font-mono tracking-wider text-slate-500 uppercase">{key}</span>
              <span className="text-xs font-mono font-bold text-slate-200 mt-0.5">{val}</span>
            </div>
          ))}
        </div>

        {/* Technology Badges */}
        <div className="flex flex-wrap gap-1.5 pt-2">
          {tags.map((tag, i) => (
            <span 
              key={i} 
              className="text-[10px] font-mono bg-white/5 text-slate-400 px-2 py-0.5 rounded border border-white/5"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 pt-6 mt-auto">
        <button
          onClick={onOpenNotebook}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-slate-200 hover:text-white border border-white/5 hover:border-white/15 transition-all text-xs font-semibold"
        >
          <FileCode className={`w-4 h-4 ${theme.accent}`} />
          <span>Jupyter Notebook</span>
        </button>

        {dashboardUrl && (
          <a
            href={dashboardUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 hover:text-purple-300 border border-purple-500/20 hover:border-purple-500/35 transition-all text-xs font-semibold"
            title="Launch Interactive Dashboard"
          >
            <ExternalLink className="w-4 h-4" />
            <span className="hidden sm:inline">Launch App</span>
          </a>
        )}
      </div>
    </GlassCard>
  );
};
