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

  // Effect 1: Initialize chart instance ONCE and handle cleanup
  useEffect(() => {
    if (!chartRef.current) return;

    const chart = echarts.init(chartRef.current, undefined, {
      renderer: 'canvas',
    });
    instanceRef.current = chart;

    const handleResize = () => {
      chart.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
      instanceRef.current = null;
    };
  }, []); // Empty deps — only runs on mount/unmount

  // Effect 2: Update options on the existing instance (no dispose/recreate)
  useEffect(() => {
    if (!instanceRef.current) return;

    instanceRef.current.setOption({
      backgroundColor: 'transparent',
      ...option
    }, { notMerge: false, lazyUpdate: true });
  }, [option]);

  return (
    <div className={`glass-card rounded-3xl p-6 relative flex flex-col justify-between ${className}`}>
      {title && (
        <h4 className="font-sans text-xs font-semibold tracking-wider text-slate-500 mb-4 uppercase">
          {title}
        </h4>
      )}
      <div ref={chartRef} style={{ width: '100%', height }} />
    </div>
  );
};
