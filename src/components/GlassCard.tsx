import React from 'react';
import { motion } from 'framer-motion';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  shadowColor?: string; // custom glowing shadow (e.g. rgba(168, 85, 247, 0.4) for purple)
  delay?: number;
  onClick?: () => void;
}

export const GlassCard: React.FC<GlassCardProps> = ({ 
  children, 
  className = '', 
  shadowColor = 'rgba(168, 85, 247, 0.25)', // Purple glow default
  delay = 0,
  onClick
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        duration: 0.6, 
        delay, 
        ease: [0.16, 1, 0.3, 1] 
      }}
      whileHover={{
        scale: 1.015,
        y: -5,
        backgroundColor: "rgba(255, 255, 255, 0.04)",
        borderColor: "rgba(255, 255, 255, 0.15)",
        boxShadow: `0 20px 40px -15px rgba(0, 0, 0, 0.5), 0 15px 30px -10px ${shadowColor}`,
      }}
      onClick={onClick}
      className={`glass-card rounded-3xl p-6 md:p-8 relative overflow-hidden ${className} ${onClick ? 'cursor-pointer' : ''}`}
    >
      {children}
    </motion.div>
  );
};
