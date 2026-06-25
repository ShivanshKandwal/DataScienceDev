import React, { useState, useEffect } from 'react';
import { RotateCcw, Sliders } from 'lucide-react';

export const GradientDescent: React.FC = () => {
  const [lr, setLr] = useState<'optimal' | 'low' | 'high'>('optimal');
  const [epoch, setEpoch] = useState(0);
  const [ballPos, setBallPos] = useState({ x: 50, y: 35 });
  const [isPlaying, setIsPlaying] = useState(true);

  // Parabola curve coordinates: y = (x - 150)^2 / 80 + 50
  const width = 300;
  const height = 150;
  const cx = 150; // center x of parabola minimum

  const getParabolaY = (x: number) => {
    return Math.pow(x - cx, 2) / 100 + 40;
  };

  // Generate path points
  const points = [];
  for (let x = 30; x <= 270; x += 5) {
    points.push(`${x},${getParabolaY(x)}`);
  }
  const pathD = `M ${points.join(' L ')}`;

  useEffect(() => {
    if (!isPlaying) return;

    let timer: number;
    let step = 0;
    
    // Starting X coordinate of the ball
    let currentX = 45;
    setEpoch(0);

    const updatePhysics = () => {
      setEpoch((prev) => prev + 1);

      // Learning rate parameters
      const eta = lr === 'optimal' ? 12 : lr === 'low' ? 3 : 28;
      
      // Gradient of y = (x - 150)^2 / 100 + 40 is 2 * (x - 150) / 100
      const gradient = (2 * (currentX - cx)) / 100;

      // Update X with learning rate step: x_new = x_old - eta * grad
      currentX = currentX - eta * gradient;

      // Bound X
      if (currentX < 20) currentX = 20;
      if (currentX > 280) currentX = 280;

      setBallPos({
        x: currentX,
        y: getParabolaY(currentX)
      });

      step++;

      // Stop if converged or epoch reaches 30
      const distToMin = Math.abs(currentX - cx);
      if (distToMin < 1 && lr === 'optimal') {
        setIsPlaying(false);
      } else if (step < 35) {
        timer = window.setTimeout(updatePhysics, 150);
      } else {
        setIsPlaying(false);
      }
    };

    timer = window.setTimeout(updatePhysics, 150);

    return () => clearTimeout(timer);
  }, [lr, isPlaying]);

  const restartSimulation = () => {
    setBallPos({ x: 45, y: getParabolaY(45) });
    setIsPlaying(true);
  };

  return (
    <div className="glass-card rounded-3xl p-6 relative flex flex-col justify-between w-full h-[400px]">
      <div className="text-left select-none">
        <h4 className="font-sans text-sm font-bold tracking-wider text-slate-500 mb-1 uppercase flex items-center gap-1.5">
          <Sliders className="w-4.5 h-4.5 text-purple-500" />
          Gradient Descent Optimizer (Sim)
        </h4>
        <p className="text-sm text-slate-500 mb-3">
          Simulate standard parameter updates. See the weights ball descend to local minima.
        </p>

        {/* Controls */}
        <div className="flex gap-2 mb-4">
          {(['optimal', 'low', 'high'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => {
                setLr(mode);
                restartSimulation();
              }}
              className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold capitalize transition-all border ${
                lr === mode 
                  ? 'bg-purple-500 text-white border-purple-400' 
                  : 'bg-white/40 text-slate-650 border-slate-200 hover:bg-white/60'
              }`}
            >
              {mode} LR
            </button>
          ))}
          <button 
            onClick={restartSimulation}
            className="p-1.5 rounded-xl bg-white/40 hover:bg-white/60 text-slate-500 border border-slate-200 transition-all ml-auto"
            title="Restart simulation"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Parabola Canvas Graph */}
      <div className="flex-1 w-full relative overflow-hidden bg-white/30 rounded-2xl border border-black/5 p-4 flex flex-col justify-center items-center">
        
        {/* Math HUD */}
        <div className="absolute top-3 left-3 font-mono text-xs text-slate-500 space-y-0.5 text-left pointer-events-none">
          <div>Loss: <span className="font-bold text-slate-700">{getParabolaY(ballPos.x).toFixed(1)}</span></div>
          <div>Epoch: <span className="font-bold text-slate-700">{epoch}</span></div>
          <div>x_new = x_old - η * ∇L</div>
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
          {/* Parabola Path */}
          <path 
            d={pathD} 
            fill="none" 
            stroke="rgba(99, 102, 241, 0.2)" 
            strokeWidth="3" 
          />

          {/* Local Minima Target indicator */}
          <circle cx={cx} cy={getParabolaY(cx)} r="4" fill="#10B981" />
          <text x={cx} y={getParabolaY(cx) + 20} textAnchor="middle" fontSize="10" fill="#10B981" fontFamily="monospace">min(L)</text>

          {/* Physics Ball */}
          <circle 
            cx={ballPos.x} 
            cy={ballPos.y} 
            r="7" 
            fill={lr === 'high' ? '#F43F5E' : lr === 'low' ? '#38BDF8' : '#A855F7'}
            style={{
              filter: `drop-shadow(0px 0px 6px ${lr === 'high' ? '#F43F5E' : lr === 'low' ? '#38BDF8' : '#A855F7'})`,
              transition: 'cx 0.15s ease-out, cy 0.15s ease-out'
            }}
          />
        </svg>
      </div>
    </div>
  );
};
