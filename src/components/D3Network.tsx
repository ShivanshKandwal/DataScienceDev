import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface NetworkNode extends d3.SimulationNodeDatum {
  id: string;
  layer: number;
  label: string;
  val: number;
}

interface NetworkLink extends d3.SimulationLinkDatum<NetworkNode> {
  source: string | NetworkNode;
  target: string | NetworkNode;
  value: number;
}

export const D3Network: React.FC = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svgElement = svgRef.current;
    if (!svgElement) return;

    // Dimensions
    const width = 800;
    const height = 460;

    // Clear previous elements
    d3.select(svgElement).selectAll('*').remove();

    const svg = d3.select(svgElement)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('width', '100%')
      .attr('height', '100%');

    // Create Neural Network nodes (5 input, 6 hidden 1, 6 hidden 2, 3 output)
    const nodes: NetworkNode[] = [
      // Input Layer (layer 0)
      { id: 'i1', layer: 0, label: 'x1', val: 0.8 },
      { id: 'i2', layer: 0, label: 'x2', val: 0.5 },
      { id: 'i3', layer: 0, label: 'x3', val: 0.9 },
      { id: 'i4', layer: 0, label: 'x4', val: 0.7 },
      { id: 'i5', layer: 0, label: 'x5', val: 0.4 },
      // Hidden Layer 1 (layer 1)
      { id: 'h1_1', layer: 1, label: 'h1_1', val: 0.6 },
      { id: 'h1_2', layer: 1, label: 'h1_2', val: 0.4 },
      { id: 'h1_3', layer: 1, label: 'h1_3', val: 0.7 },
      { id: 'h1_4', layer: 1, label: 'h1_4', val: 0.3 },
      { id: 'h1_5', layer: 1, label: 'h1_5', val: 0.8 },
      { id: 'h1_6', layer: 1, label: 'h1_6', val: 0.5 },
      // Hidden Layer 2 (layer 2)
      { id: 'h2_1', layer: 2, label: 'h2_1', val: 0.5 },
      { id: 'h2_2', layer: 2, label: 'h2_2', val: 0.8 },
      { id: 'h2_3', layer: 2, label: 'h2_3', val: 0.2 },
      { id: 'h2_4', layer: 2, label: 'h2_4', val: 0.6 },
      { id: 'h2_5', layer: 2, label: 'h2_5', val: 0.4 },
      { id: 'h2_6', layer: 2, label: 'h2_6', val: 0.7 },
      // Output Layer (layer 3)
      { id: 'o1', layer: 3, label: 'y1', val: 0.95 },
      { id: 'o2', layer: 3, label: 'y2', val: 0.05 },
      { id: 'o3', layer: 3, label: 'y3', val: 0.34 },
    ];

    // Establish fully connected feedforward links
    const links: NetworkLink[] = [];
    nodes.forEach(sourceNode => {
      nodes.forEach(targetNode => {
        if (targetNode.layer === sourceNode.layer + 1) {
          links.push({
            source: sourceNode.id,
            target: targetNode.id,
            value: Math.random() * 2 + 0.5,
          });
        }
      });
    });

    // Positions based on layers (for starting coordinates)
    nodes.forEach((node) => {
      const colWidth = width / 4;
      const colNodes = nodes.filter(n => n.layer === node.layer);
      const idx = colNodes.indexOf(node);
      const rowGap = height / (colNodes.length + 1);
      
      node.x = colWidth * node.layer + colWidth / 2;
      node.y = rowGap * (idx + 1);
    });

    // Force simulation with column-restraining x-forces to prevent horizontal collapse
    const simulation = d3.forceSimulation<NetworkNode>(nodes)
      .force('link', d3.forceLink<NetworkNode, NetworkLink>(links).id(d => d.id).distance(180))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('x', d3.forceX<NetworkNode>().x(d => {
        const colWidth = width / 4;
        return colWidth * d.layer + colWidth / 2;
      }).strength(0.85))
      .force('y', d3.forceY<NetworkNode>().y(height / 2).strength(0.08))
      .force('collision', d3.forceCollide().radius(36));

    // Links container (soft dark line for light theme)
    const linkGroup = svg.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', 'rgba(15, 23, 42, 0.08)')
      .attr('stroke-width', d => d.value * 1.2)
      .style('stroke-dasharray', '3,3');

    // Pulsing signal elements (representing feedforward activations)
    const pulsesGroup = svg.append('g').attr('class', 'pulses');

    const spawnPulses = () => {
      pulsesGroup.selectAll('*').remove();
      
      pulsesGroup.selectAll('circle')
        .data(links)
        .enter()
        .append('circle')
        .attr('r', 4)
        .attr('fill', (d) => {
          // Color signal based on layers (vibrant colors)
          const src = d.source as NetworkNode;
          if (src.layer === 0) return '#10B981'; // Green for input
          if (src.layer === 1) return '#6366F1'; // Indigo for hidden 1
          return '#F43F5E'; // Rose for deep
        })
        .style('filter', 'drop-shadow(0px 0px 4px currentColor)')
        .each(function(d) {
          const self = d3.select(this);
          const src = d.source as NetworkNode;
          const tgt = d.target as NetworkNode;
          
          self.attr('cx', src.x || 0)
            .attr('cy', src.y || 0);

          self.transition()
            .duration(1200 + Math.random() * 800)
            .delay(Math.random() * 1000)
            .ease(d3.easeQuadInOut)
            .attr('cx', tgt.x || 0)
            .attr('cy', tgt.y || 0)
            .style('opacity', 0)
            .remove();
        });
    };

    // Periodically spawn pulses
    const pulseInterval = setInterval(spawnPulses, 2500);
    spawnPulses(); // initial call

    // Nodes container
    const nodeGroup = svg.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .call(
        d3.drag<SVGGElement, NetworkNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Node glowing shadows (Light Theme optimized)
    nodeGroup.append('circle')
      .attr('r', 28)
      .attr('fill', d => {
        if (d.layer === 0) return 'rgba(16, 185, 129, 0.08)'; // input
        if (d.layer === 1) return 'rgba(99, 102, 241, 0.08)'; // hidden 1
        if (d.layer === 2) return 'rgba(168, 85, 247, 0.08)'; // hidden 2
        return 'rgba(244, 63, 94, 0.08)'; // output
      })
      .attr('stroke', d => {
        if (d.layer === 0) return '#10B981';
        if (d.layer === 1) return '#6366F1';
        if (d.layer === 2) return '#A855F7';
        return '#F43F5E';
      })
      .attr('stroke-width', 2.5)
      .style('cursor', 'grab');

    // Inner core (Solid white for contrast in light theme)
    nodeGroup.append('circle')
      .attr('r', 16)
      .attr('fill', '#FFFFFF')
      .style('pointer-events', 'none');

    // Label text (Charcoal for light mode legibility)
    nodeGroup.append('text')
      .text(d => d.label)
      .attr('x', 0)
      .attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('fill', '#475569')
      .attr('font-size', '13px')
      .attr('font-family', 'ui-monospace, monospace')
      .style('pointer-events', 'none');

    // Value text inside node
    nodeGroup.append('text')
      .text(d => d.val.toFixed(1))
      .attr('x', 0)
      .attr('y', 4.5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#0F172A')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('font-family', 'sans-serif')
      .style('pointer-events', 'none');

    // Update positions on tick
    simulation.on('tick', () => {
      linkGroup
        .attr('x1', d => (d.source as NetworkNode).x || 0)
        .attr('y1', d => (d.source as NetworkNode).y || 0)
        .attr('x2', d => (d.target as NetworkNode).x || 0)
        .attr('y2', d => (d.target as NetworkNode).y || 0);

      nodeGroup.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });

    return () => {
      simulation.stop();
      clearInterval(pulseInterval);
    };
  }, []);

  return (
    <div className="glass-card rounded-3xl p-8 relative flex flex-col justify-between w-full h-[550px] md:h-[620px]">
      <div>
        <h4 className="font-sans text-sm font-bold tracking-wider text-slate-500 mb-1 uppercase">
          Neural Network Visualizer (Live D3 simulation)
        </h4>
        <p className="text-sm text-slate-400 mb-4">
          Interactive feedforward connection paths. Drag nodes to reshape and study activation values.
        </p>
      </div>
      <div className="flex-1 w-full relative overflow-hidden bg-white/20 rounded-2xl border border-black/5">
        <svg ref={svgRef} className="w-full h-full" />
      </div>
    </div>
  );
};
