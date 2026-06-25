import React, { useEffect, useState, useRef } from 'react';
import { X, Code, AlertCircle, Download } from 'lucide-react';
import Plotly from 'plotly.js-dist-min';

// ANSI escape sequence remover
const stripAnsi = (str: string) => {
  /* eslint-disable-next-line no-control-regex */
  return str.replace(/[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, '');
};

// Python Syntax Highlighter (Light Mode optimized)
const highlightPython = (code: string) => {
  const escapeHtml = (text: string) => {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  };

  // Group 1: Comment
  // Group 2: String
  // Group 3: Keyword
  // Group 4: Builtin
  // Group 5: Decorator
  // Group 6: Number
  // Group 7: Operator
  // Group 8: Identifier / word
  // Group 9: Whitespace
  // Group 10: Rest
  const tokenRegex = /(#.*)|('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")|(\b(?:def|class|import|from|return|if|else|elif|for|while|try|except|finally|with|as|in|is|not|and|or|lambda|pass|break|continue|global|assert)\b)|(\b(?:print|len|range|str|int|float|list|dict|set|tuple|enumerate|zip|sum|min|max|type|abs|round|open)\b)|(@\w+)|(\b\d+(?:\.\d+)?\b)|([+\-*/%=<>!&|^~]+)|(\w+)|(\s+)|(.)/g;

  return code.replace(tokenRegex, (match, comment, str, keyword, builtin, decorator, num, operator, word, space, char) => {
    if (comment) {
      return `<span class="py-comment">${escapeHtml(comment)}</span>`;
    }
    if (str) {
      return `<span class="py-string">${escapeHtml(str)}</span>`;
    }
    if (keyword) {
      return `<span class="py-keyword">${escapeHtml(keyword)}</span>`;
    }
    if (builtin) {
      return `<span class="py-builtin">${escapeHtml(builtin)}</span>`;
    }
    if (decorator) {
      return `<span class="py-decorator">${escapeHtml(decorator)}</span>`;
    }
    if (num) {
      return `<span class="py-number">${escapeHtml(num)}</span>`;
    }
    if (operator) {
      return `<span class="py-operator">${escapeHtml(operator)}</span>`;
    }
    if (word) {
      return escapeHtml(word);
    }
    if (space) {
      return space;
    }
    if (char) {
      return escapeHtml(char);
    }
    return escapeHtml(match);
  });
};

// Simple Markdown Renderer (Light Mode optimized)
const renderMarkdown = (source: string) => {
  const lines = source.split('\n');
  const renderedHtml = lines.map((line) => {
    let cleanLine = line.trim();
    if (!cleanLine) return '';

    // Headers
    if (cleanLine.startsWith('# ')) {
      return `<h1 class="font-serif text-2xl md:text-3xl font-bold text-slate-800 mt-6 mb-4">${cleanLine.slice(2)}</h1>`;
    }
    if (cleanLine.startsWith('## ')) {
      return `<h2 class="font-serif text-xl md:text-2xl font-semibold text-slate-700 mt-5 mb-3">${cleanLine.slice(3)}</h2>`;
    }
    if (cleanLine.startsWith('### ')) {
      return `<h3 class="font-serif text-lg md:text-xl font-medium text-slate-700 mt-4 mb-2">${cleanLine.slice(4)}</h3>`;
    }
    if (cleanLine.startsWith('#### ')) {
      return `<h4 class="font-sans text-base md:text-lg font-medium text-slate-600 mt-3 mb-1">${cleanLine.slice(5)}</h4>`;
    }

    // Unordered lists
    if (cleanLine.startsWith('- ') || cleanLine.startsWith('* ')) {
      const liContent = cleanLine.slice(2);
      const formattedLi = formatInlineMarkdown(liContent);
      return `<li class="ml-6 list-disc text-slate-600 my-1">${formattedLi}</li>`;
    }

    // Paragraph
    const formattedParagraph = formatInlineMarkdown(cleanLine);
    return `<p class="text-slate-600 text-sm md:text-base leading-relaxed my-2">${formattedParagraph}</p>`;
  }).join('');

  return renderedHtml;
};

// Inline markdown helper (bold, code, links) - Light Mode
const formatInlineMarkdown = (text: string) => {
  let formatted = text;
  // Bold **text**
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-800 font-semibold">$1</strong>');
  // Inline code `code`
  formatted = formatted.replace(/`(.*?)`/g, '<code class="bg-black/5 border border-black/10 px-1.5 py-0.5 rounded text-pink-700 font-mono text-xs">$1</code>');
  // Links [text](url)
  formatted = formatted.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="text-purple-600 hover:text-purple-500 underline font-medium">$1</a>');
  return formatted;
};

// Sub-component for rendering Plotly JSON outputs (Light Mode optimized)
const PlotlyChart: React.FC<{ plotlyData: any }> = ({ plotlyData }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    // Set a sleek light layout template
    const layout = {
      ...plotlyData.layout,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'rgba(0, 0, 0, 0.02)',
      font: {
        family: 'Outfit, sans-serif',
        color: '#475569', // slate 600
        size: 11
      },
      xaxis: {
        ...plotlyData.layout?.xaxis,
        gridcolor: 'rgba(0, 0, 0, 0.05)',
        zerolinecolor: 'rgba(0, 0, 0, 0.1)',
        linecolor: 'rgba(0, 0, 0, 0.1)'
      },
      yaxis: {
        ...plotlyData.layout?.yaxis,
        gridcolor: 'rgba(0, 0, 0, 0.05)',
        zerolinecolor: 'rgba(0, 0, 0, 0.1)',
        linecolor: 'rgba(0, 0, 0, 0.1)'
      },
      margin: { t: 40, r: 20, b: 40, l: 40 }
    };

    Plotly.newPlot(container, plotlyData.data, layout, {
      responsive: true,
      displayModeBar: false,
    });

    return () => {
      if (container) {
        Plotly.purge(container);
      }
    };
  }, [plotlyData]);

  return <div ref={containerRef} className="w-full min-h-[350px] my-3 rounded-2xl overflow-hidden border border-black/5 bg-slate-50/50" />;
};

interface NotebookViewerProps {
  notebookUrl: string;
  title: string;
  onClose: () => void;
  category: 'EDA' | 'ML' | 'AI';
}

export const NotebookViewer: React.FC<NotebookViewerProps> = ({
  notebookUrl,
  title,
  onClose,
  category
}) => {
  const [cells, setCells] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Category specific glow colors
  const glowClass = {
    EDA: 'rgba(16, 185, 129, 0.2)', // Emerald
    ML: 'rgba(99, 102, 241, 0.2)',  // Indigo
    AI: 'rgba(244, 63, 94, 0.2)'    // Rose
  }[category];

  useEffect(() => {
    setLoading(true);
    setError(null);
    
    fetch(notebookUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch notebook (HTTP ${res.status})`);
        return res.json();
      })
      .then((data) => {
        if (data && data.cells) {
          setCells(data.cells);
        } else {
          throw new Error("Invalid Jupyter notebook format (.ipynb)");
        }
      })
      .catch((err) => {
        console.error("Error loading notebook:", err);
        setError(err.message || "Failed to load the Jupyter notebook.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [notebookUrl]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm">
      <div 
        className="w-full max-w-5xl h-[85vh] flex flex-col glass-card rounded-3xl overflow-hidden relative border border-slate-200/80 bg-white/95"
        style={{
          boxShadow: `0 25px 50px -12px rgba(0, 0, 0, 0.15), 0 0 40px -5px ${glowClass}`
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-slate-50/50">
          <div className="flex items-center gap-3">
            <span className={`w-3 h-3 rounded-full ${
              category === 'EDA' ? 'bg-emerald-500' : category === 'ML' ? 'bg-indigo-500' : 'bg-rose-500'
            }`} />
            <h3 className="font-serif text-base md:text-lg font-semibold text-slate-800 truncate max-w-lg">
              {title}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <a 
              href={notebookUrl} 
              download 
              className="p-2 text-slate-500 hover:text-slate-800 rounded-full bg-slate-100 hover:bg-slate-200 transition-colors"
              title="Download Jupyter Notebook"
            >
              <Download className="w-4 h-4" />
            </a>
            <button 
              onClick={onClose} 
              className="p-2 text-slate-500 hover:text-slate-800 rounded-full bg-slate-100 hover:bg-slate-200 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Notebook Body */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 no-scrollbar bg-white/40">
          {loading && (
            <div className="flex flex-col items-center justify-center h-full space-y-4">
              <div className="w-8 h-8 rounded-full border-2 border-purple-500 border-t-transparent animate-spin" />
              <p className="text-sm font-sans tracking-wide text-slate-500">
                Parsing Jupyter Notebook cells dynamically...
              </p>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center h-full space-y-4 text-center">
              <AlertCircle className="w-12 h-12 text-rose-500 animate-bounce" />
              <h4 className="text-lg font-serif text-slate-700">Notebook Parsing Failed</h4>
              <p className="text-sm text-slate-500 max-w-md">{error}</p>
              <button 
                onClick={onClose} 
                className="px-5 py-2 text-sm bg-rose-500 hover:bg-rose-600 rounded-full text-white font-medium transition-colors"
              >
                Close Viewer
              </button>
            </div>
          )}

          {!loading && !error && cells.map((cell, cellIdx) => {
            const sourceText = Array.isArray(cell.source) ? cell.source.join('') : cell.source;

            if (cell.cell_type === 'markdown') {
              return (
                <div 
                  key={cellIdx} 
                  className="prose max-w-none text-left"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(sourceText) }}
                />
              );
            }

            if (cell.cell_type === 'code') {
              return (
                <div key={cellIdx} className="space-y-3 text-left">
                  {/* Code Block Container */}
                  <div className="ipynb-cell-code rounded-2xl overflow-hidden shadow-sm">
                    <div className="flex items-center justify-between px-4 py-2 border-b border-slate-100 bg-slate-50/50">
                      <div className="flex items-center gap-2">
                        <Code className="w-3.5 h-3.5 text-purple-600" />
                        <span className="text-[10px] font-mono text-slate-500">
                          In [{cell.execution_count || ' '}]
                        </span>
                      </div>
                      <span className="text-[9px] font-sans tracking-wider text-slate-400 uppercase font-semibold">
                        Python Cell
                      </span>
                    </div>
                    <pre className="p-4 overflow-x-auto font-mono text-xs md:text-sm leading-relaxed no-scrollbar text-slate-700">
                      <code dangerouslySetInnerHTML={{ __html: highlightPython(sourceText) }} />
                    </pre>
                  </div>

                  {/* Outputs Container */}
                  {cell.outputs && cell.outputs.length > 0 && (
                    <div className="space-y-3 pl-4 border-l border-slate-150 mt-2">
                      {cell.outputs.map((out: any, outIdx: number) => {
                        // Stream output (e.g. print statements)
                        if (out.output_type === 'stream') {
                          const streamText = Array.isArray(out.text) ? out.text.join('') : out.text;
                          return (
                            <pre 
                              key={outIdx} 
                              className="ipynb-cell-output-stream p-4 rounded-xl text-xs overflow-x-auto leading-relaxed shadow-sm whitespace-pre-wrap max-h-[300px] overflow-y-auto no-scrollbar"
                            >
                              {streamText}
                            </pre>
                          );
                        }

                        // Code execution results/display data
                        if (out.output_type === 'execute_result' || out.output_type === 'display_data') {
                          const mimeTypes = out.data || {};

                          // 1. Matplotlib / Seaborn PNG images
                          if (mimeTypes['image/png']) {
                            const b64 = mimeTypes['image/png'].replace(/\n/g, '');
                            return (
                              <div key={outIdx} className="my-4 flex justify-center">
                                <img 
                                  src={`data:image/png;base64,${b64}`} 
                                  alt="Matplotlib Plot output" 
                                  className="rounded-2xl border border-slate-200 max-w-full shadow-md bg-slate-50 p-2" 
                                 />
                              </div>
                            );
                          }

                          // 2. Plotly JSON Interactive chart representation
                          if (mimeTypes['application/vnd.plotly.v1+json']) {
                            const plotlyJSON = mimeTypes['application/vnd.plotly.v1+json'];
                            return <PlotlyChart key={outIdx} plotlyData={plotlyJSON} />;
                          }

                          // 3. Pandas DataFrame / HTML outputs
                          if (mimeTypes['text/html']) {
                            const html = Array.isArray(mimeTypes['text/html']) 
                              ? mimeTypes['text/html'].join('') 
                              : mimeTypes['text/html'];
                            return (
                              <div 
                                key={outIdx} 
                                className="overflow-x-auto my-3 rounded-xl border border-slate-200 bg-slate-50/30 no-scrollbar"
                                dangerouslySetInnerHTML={{ __html: html }}
                              />
                            );
                          }

                          // 4. Text output
                          if (mimeTypes['text/plain']) {
                            const plainText = Array.isArray(mimeTypes['text/plain'])
                              ? mimeTypes['text/plain'].join('')
                              : mimeTypes['text/plain'];
                            return (
                              <pre 
                                key={outIdx} 
                                className="p-3 bg-slate-50 border border-slate-200 rounded-xl font-mono text-xs text-slate-500 overflow-x-auto no-scrollbar"
                              >
                                {plainText}
                              </pre>
                            );
                          }
                        }

                        // Errors/tracebacks
                        if (out.output_type === 'error') {
                          const errorText = Array.isArray(out.traceback) 
                            ? out.traceback.join('\n') 
                            : out.traceback;
                          const cleanTrace = stripAnsi(errorText);
                          return (
                            <pre 
                              key={outIdx} 
                              className="ipynb-cell-output-error p-4 rounded-xl text-xs overflow-x-auto whitespace-pre leading-relaxed shadow-sm max-h-[300px] overflow-y-auto no-scrollbar"
                            >
                              {cleanTrace}
                            </pre>
                          );
                        }

                        return null;
                      })}
                    </div>
                  )}
                </div>
              );
            }

            return null;
          })}
        </div>
      </div>
    </div>
  );
};
