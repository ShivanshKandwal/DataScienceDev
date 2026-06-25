import React, { useState, useEffect } from 'react';
import { Cpu, Thermometer, Database, Activity, Terminal } from 'lucide-react';

export const TelemetryDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState({
    gpuLoad: 88,
    vram: 11.2,
    temp: 74,
    speed: 342,
    loss: 0.158,
    accuracy: 94.2
  });

  const [logs, setLogs] = useState<string[]>([
    "Initializing training session on CUDA device:0 (RTX 4090)...",
    "Loaded dataset with 48,200 training samples.",
    "Epoch 1/100 | loss: 0.892 | acc: 62.4%",
    "Epoch 10/100 | loss: 0.451 | acc: 81.9%",
    "Epoch 20/100 | loss: 0.284 | acc: 89.2%",
  ]);

  useEffect(() => {
    const metricsInterval = setInterval(() => {
      setMetrics(prev => {
        const deltaLoad = (Math.random() - 0.5) * 4;
        const newLoad = Math.min(Math.max(prev.gpuLoad + deltaLoad, 80), 98);
        
        const deltaVram = (Math.random() - 0.5) * 0.2;
        const newVram = Math.min(Math.max(prev.vram + deltaVram, 10.8), 12.0);

        const deltaTemp = (Math.random() - 0.5) * 2;
        const newTemp = Math.min(Math.max(prev.temp + deltaTemp, 70), 79);

        const deltaSpeed = (Math.random() - 0.5) * 15;
        const newSpeed = Math.min(Math.max(prev.speed + deltaSpeed, 320), 365);

        // Gradually decay loss, increase accuracy
        const newLoss = Math.max(prev.loss - 0.001 * Math.random(), 0.015);
        const newAccuracy = Math.min(prev.accuracy + 0.02 * Math.random(), 99.8);

        return {
          gpuLoad: parseFloat(newLoad.toFixed(1)),
          vram: parseFloat(newVram.toFixed(2)),
          temp: Math.round(newTemp),
          speed: Math.round(newSpeed),
          loss: parseFloat(newLoss.toFixed(4)),
          accuracy: parseFloat(newAccuracy.toFixed(2))
        };
      });
    }, 1000);

    const logInterval = setInterval(() => {
      setMetrics(current => {
        setLogs(prev => {
          const nextEpoch = prev.length - 4 + 21;
          const newLog = `Epoch ${nextEpoch}/100 | loss: ${current.loss.toFixed(3)} | acc: ${current.accuracy.toFixed(1)}% | speed: ${current.speed} it/s`;
          const updated = [...prev, newLog];
          if (updated.length > 5) updated.shift();
          return updated;
        });
        return current;
      });
    }, 4000);

    return () => {
      clearInterval(metricsInterval);
      clearInterval(logInterval);
    };
  }, []);

  return (
    <div className="glass-card rounded-3xl p-6 relative flex flex-col justify-between w-full h-[400px]">
      <div className="text-left">
        <h4 className="font-sans text-sm font-bold tracking-wider text-slate-500 mb-1 uppercase flex items-center gap-1.5">
          <Activity className="w-4.5 h-4.5 text-rose-500 animate-pulse" />
          Model Training &amp; GPU Telemetry (RTX 4090)
        </h4>
        <p className="text-sm text-slate-500 mb-3 font-medium">
          Live cluster resource utilization monitoring. Neural network weights optimization telemetry.
        </p>
      </div>

      {/* Grid of stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4 select-none">
        
        {/* GPU Load */}
        <div className="bg-white/30 rounded-2xl border border-black/5 p-3 flex flex-col items-center justify-center">
          <Cpu className="w-5 h-5 text-indigo-500 mb-1" />
          <span className="text-xs text-slate-400 font-semibold">GPU Load</span>
          <span className="text-base font-bold text-slate-800 font-mono">{metrics.gpuLoad}%</span>
        </div>

        {/* VRAM */}
        <div className="bg-white/30 rounded-2xl border border-black/5 p-3 flex flex-col items-center justify-center">
          <Database className="w-5 h-5 text-purple-500 mb-1" />
          <span className="text-xs text-slate-400 font-semibold">VRAM Alloc</span>
          <span className="text-base font-bold text-slate-800 font-mono">{metrics.vram} GB</span>
        </div>

        {/* Temp */}
        <div className="bg-white/30 rounded-2xl border border-black/5 p-3 flex flex-col items-center justify-center">
          <Thermometer className="w-5 h-5 text-rose-500 mb-1" />
          <span className="text-xs text-slate-400 font-semibold">Core Temp</span>
          <span className="text-base font-bold text-slate-800 font-mono">{metrics.temp}°C</span>
        </div>

        {/* Speed */}
        <div className="bg-white/30 rounded-2xl border border-black/5 p-3 flex flex-col items-center justify-center">
          <Activity className="w-5 h-5 text-emerald-500 mb-1" />
          <span className="text-xs text-slate-400 font-semibold">Step Speed</span>
          <span className="text-base font-bold text-slate-800 font-mono">{metrics.speed} it/s</span>
        </div>

      </div>

      {/* Mini terminal */}
      <div className="flex-1 w-full bg-slate-900/95 rounded-2xl p-3 border border-black/10 flex flex-col justify-between font-mono text-xs">
        <div className="flex items-center gap-1.5 text-emerald-450 mb-1.5 border-b border-white/5 pb-1 select-none">
          <Terminal className="w-4 h-4" />
          <span className="text-xs uppercase tracking-wider font-semibold">Console Outputs (Live Stream)</span>
        </div>
        <div className="flex-1 flex flex-col justify-end gap-1 text-slate-300">
          {logs.map((log, idx) => (
            <div key={idx} className="whitespace-nowrap overflow-hidden text-ellipsis text-left">
              <span className="text-slate-500">[{new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}]</span> {log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
