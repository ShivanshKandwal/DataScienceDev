import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

interface EChartsCardProps {
  option: echarts.EChartsOption;
  className?: string;
  title?: string;
  height?: string;
}

export const EChartsCard: React.FC<EChartsCardProps> = ({
  option,
  className = '',
  title,
  height = '300px'
}) => {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const instanceRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    // Initialize ECharts instance
    const chart = echarts.init(chartRef.current, 'dark', {
      renderer: 'canvas',
    });
    instanceRef.current = chart;

    // Set options
    chart.setOption({
      backgroundColor: 'transparent',
      ...option
    });

    const handleResize = () => {
      chart.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
    };
  }, [option]);

  return (
    <div className={`glass-card rounded-3xl p-6 relative flex flex-col justify-between ${className}`}>
      {title && (
        <h4 className="font-sans text-sm font-semibold tracking-wider text-slate-400 mb-4 uppercase">
          {title}
        </h4>
      )}
      <div ref={chartRef} style={{ width: '100%', height }} />
    </div>
  );
};
