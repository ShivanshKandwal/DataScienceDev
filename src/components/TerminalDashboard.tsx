import React, { useState, useRef, useEffect } from 'react';
import { Terminal, CornerDownLeft } from 'lucide-react';

interface TerminalLine {
  text: string;
  type: 'system' | 'input' | 'output' | 'error' | 'success' | 'info';
}

export const TerminalDashboard: React.FC = () => {
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([
    { text: "Welcome to Shivansh Kandwal Data Science CLI v1.2.0", type: 'system' },
    { text: "Hardware active: NVIDIA RTX 4090 (CUDA 12.1). System: OK.", type: 'success' },
    { text: "Type 'help' to see list of available commands.", type: 'info' },
    { text: "", type: 'system' }
  ]);
  const [terminalInput, setTerminalInput] = useState('');
  const terminalEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    // Scroll terminal to bottom when content changes
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [terminalLines]);

  const handleTerminalSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cmd = terminalInput.trim().toLowerCase();
    if (!cmd) return;

    // Add user input line
    const newLines = [...terminalLines, { text: `> ${terminalInput}`, type: 'input' as const }];
    
    // Command Router
    switch (cmd) {
      case 'help':
        newLines.push(
          { text: "Available commands:", type: 'info' },
          { text: "  help        - Display list of commands", type: 'system' },
          { text: "  list        - List all cataloged Data Science projects", type: 'system' },
          { text: "  status      - Display server and GPU diagnostics", type: 'system' },
          { text: "  accuracy    - Display tabular model metrics", type: 'system' },
          { text: "  run titanic - Execute Titanic EDA data-mining mock script", type: 'system' },
          { text: "  run dl-net  - Initialize Deep Neural Network training simulation", type: 'system' },
          { text: "  clear       - Clear screen history", type: 'system' }
        );
        break;

      case 'list':
        newLines.push(
          { text: "Cataloged Projects:", type: 'info' },
          { text: "  [EDA] titanic_eda.ipynb      - Exploratory Analysis on Titanic Survival", type: 'system' },
          { text: "  [ML]  housing_regression.ipynb - Predict pricing utilizing Random Forest", type: 'system' },
          { text: "  [DL]  brain_tumor_segment.ipynb - CNN segmentation using PyTorch", type: 'system' }
        );
        break;

      case 'status':
        newLines.push(
          { text: "System Diagnostics:", type: 'info' },
          { text: "  GPU Model: NVIDIA GeForce RTX 4090 (24GB VRAM)", type: 'system' },
          { text: "  CUDA Version: 12.1 | Driver Version: 531.14", type: 'system' },
          { text: "  Memory usage: 14.8 GB / 64 GB RAM (Active)", type: 'system' },
          { text: "  Active Kernels: 3 (Python 3.11.9, PyTorch 2.1.2)", type: 'system' },
          { text: "  Status: Idle & Operational", type: 'success' }
        );
        break;

      case 'accuracy':
        newLines.push(
          { text: "Model Leaderboard:", type: 'info' },
          { text: "  Project Name           | Model Type          | Accuracy | F1-Score", type: 'success' },
          { text: "  -----------------------+---------------------+----------+---------", type: 'system' },
          { text: "  titanic_eda            | Random Forest       |  83.2%   |  0.81   ", type: 'system' },
          { text: "  housing_regression     | XGBoost Regressor   |  91.5%   |  0.90   ", type: 'system' },
          { text: "  brain_tumor_segment    | ResNet50 Unet (DL)  |  96.8%   |  0.95   ", type: 'system' }
        );
        break;

      case 'run titanic':
        newLines.push(
          { text: "Initializing Titanic Data Miner...", type: 'info' },
          { text: "[1/3] Loading passenger manifest dataset...", type: 'system' },
          { text: "[2/3] Cleaning missing age values using median imputation...", type: 'system' },
          { text: "[3/3] Engineering 'FamilySize' and 'IsAlone' features...", type: 'system' },
          { text: "Classification report:", type: 'info' },
          { text: "                 precision    recall  f1-score   support", type: 'system' },
          { text: "       Died       0.85      0.88      0.86       145", type: 'system' },
          { text: "       Survived   0.81      0.76      0.78        95", type: 'system' },
          { text: "       accuracy                       0.83       240", type: 'success' },
          { text: "Mining simulation complete. Output saved to: /exports/findings.csv", type: 'success' }
        );
        break;

      case 'run dl-net':
        newLines.push(
          { text: "Configuring PyTorch CNN training loop...", type: 'info' },
          { text: "Epoch 1/5 - Loss: 0.842 - Acc: 56.4% - Val Loss: 0.695 - Val Acc: 68.2%", type: 'system' },
          { text: "Epoch 2/5 - Loss: 0.512 - Acc: 78.9% - Val Loss: 0.421 - Val Acc: 84.5%", type: 'system' },
          { text: "Epoch 3/5 - Loss: 0.301 - Acc: 89.2% - Val Loss: 0.288 - Val Acc: 91.0%", type: 'system' },
          { text: "Epoch 4/5 - Loss: 0.187 - Acc: 94.5% - Val Loss: 0.201 - Val Acc: 95.2%", type: 'system' },
          { text: "Epoch 5/5 - Loss: 0.104 - Acc: 97.8% - Val Loss: 0.115 - Val Acc: 96.8%", type: 'success' },
          { text: "Model saved: /weights/cnn_best_weights.pt. Training complete!", type: 'success' }
        );
        break;

      case 'clear':
        setTerminalLines([]);
        setTerminalInput('');
        return;

      default:
        newLines.push({
          text: `Command not found: '${cmd}'. Type 'help' to see valid utilities.`,
          type: 'error'
        });
    }

    setTerminalLines(newLines);
    setTerminalInput('');
  };

  const focusInput = () => {
    inputRef.current?.focus();
  };

  return (
    <div 
      onClick={focusInput}
      className="glass-card rounded-3xl p-6 relative flex flex-col justify-between w-full h-[350px] bg-slate-950/70 border border-white/5 shadow-2xl cursor-text text-left font-mono"
    >
      {/* Terminal Title Bar */}
      <div className="flex items-center justify-between pb-3 border-b border-white/5 select-none">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-purple-400" />
          <span className="text-xs font-semibold tracking-wider text-slate-400 uppercase font-sans">
            AI Core Control Panel
          </span>
        </div>
        <div className="flex gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-slate-700"></span>
          <span className="w-2.5 h-2.5 rounded-full bg-slate-700"></span>
          <span className="w-2.5 h-2.5 rounded-full bg-slate-700"></span>
        </div>
      </div>

      {/* Terminal Console Output */}
      <div className="flex-1 overflow-y-auto py-4 space-y-1.5 no-scrollbar text-xs md:text-sm">
        {terminalLines.map((line, idx) => (
          <div 
            key={idx} 
            className={`${
              line.type === 'system' ? 'text-slate-400' 
              : line.type === 'input' ? 'text-purple-300 font-bold' 
              : line.type === 'success' ? 'text-emerald-400' 
              : line.type === 'error' ? 'text-rose-400' 
              : line.type === 'info' ? 'text-cyan-300 font-medium' 
              : 'text-slate-300'
            }`}
          >
            {line.text}
          </div>
        ))}
        <div ref={terminalEndRef} />
      </div>

      {/* Terminal Command Input Form */}
      <form onSubmit={handleTerminalSubmit} className="flex items-center border-t border-white/5 pt-3">
        <span className="text-purple-400 font-bold mr-2 text-xs md:text-sm select-none">&gt;</span>
        <input
          ref={inputRef}
          type="text"
          value={terminalInput}
          onChange={(e) => setTerminalInput(e.target.value)}
          placeholder="Type 'help'..."
          className="flex-1 bg-transparent text-slate-100 placeholder-slate-600 outline-none border-none text-xs md:text-sm"
        />
        <button type="submit" className="text-purple-500 hover:text-purple-400 transition-colors p-1">
          <CornerDownLeft className="w-3.5 h-3.5" />
        </button>
      </form>
    </div>
  );
};
