import os
import re
import sys
import io
import json
import base64
import traceback
import yaml

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.void, np.integer, np.floating)):
                return obj.item()
            import pandas as pd
            if isinstance(obj, pd.Index):
                return obj.tolist()
        except ImportError:
            pass
        return super(NumpyEncoder, self).default(obj)

# Set matplotlib backend to a non-interactive one (Agg) to prevent GUI windows from opening and blocking
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

# Global list to capture Plotly figures via a custom renderer
PLOTLY_FIGS = []

try:
    import plotly.io as pio
    import plotly.io._renderers as r
    class CapturedRenderer(r.ExternalRenderer):
        def render(self, fig, **kwargs):
            fig_dict = fig.to_plotly_json() if hasattr(fig, 'to_plotly_json') else fig
            PLOTLY_FIGS.append(fig_dict)
            
    pio.renderers['captured'] = CapturedRenderer()
    pio.renderers.default = 'captured'
except ImportError:
    pass


def execute_cell(code_str, glob_env):
    # Clear captured Plotly figures from previous cells
    PLOTLY_FIGS.clear()
    
    # Setup standard output and error capture
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = mystdout = io.StringIO()
    sys.stderr = mystderr = io.StringIO()
    
    result = None
    tb = None
    
    try:
        # Run code statefully using AST parsing (matching Jupyter's last-expression evaluation behavior)
        import ast
        tree = ast.parse(code_str)
        
        if tree.body:
            last_node = tree.body[-1]
            if isinstance(last_node, ast.Expr):
                # Compile and exec all statements except the last expression
                if len(tree.body) > 1:
                    exec_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(exec_tree, filename="<cell>", mode="exec"), glob_env)
                
                # Compile and eval the last expression to capture its return value
                eval_tree = ast.Expression(body=last_node.value)
                result = eval(compile(eval_tree, filename="<cell>", mode="eval"), glob_env)
            else:
                # If the last statement is not an expression (e.g. def, class, loop), run the whole cell
                exec(compile(tree, filename="<cell>", mode="exec"), glob_env)
        else:
            exec(code_str, glob_env)
    except Exception as e:
        tb = traceback.format_exc()
        sys.stderr.write(tb)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    # Capture Matplotlib / Seaborn figures
    images = []
    try:
        # Check if matplotlib.pyplot is loaded in the globals
        if 'plt' in glob_env or 'matplotlib' in glob_env or 'sns' in glob_env:
            import matplotlib.pyplot as plt
            for num in plt.get_fignums():
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                images.append(img_b64)
                plt.close(fig)
    except Exception as e:
        sys.stderr.write(f"Error capturing matplotlib figure: {e}\n")
        
    return {
        'stdout': mystdout.getvalue(),
        'stderr': mystderr.getvalue(),
        'result': result,
        'traceback': tb,
        'images': images,
        'plotly_figs': list(PLOTLY_FIGS)  # Make a copy of the captured Plotly figures
    }


def compile_script_to_notebook(script_path, output_dir):
    print(f"Reading script: {script_path}")
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 1. Parse metadata header (YAML frontmatter block)
    # Starts with # ---, ends with # ---
    metadata = {}
    remaining_content = content
    
    frontmatter_match = re.match(r'^#\s*---\s*\n(.*?)\n#\s*---\s*\n', content, re.DOTALL)
    if frontmatter_match:
        yaml_content = frontmatter_match.group(1)
        # Strip leading # from YAML content lines
        yaml_lines = []
        for line in yaml_content.splitlines():
            # Remove leading '#' and optional whitespace
            clean_line = re.sub(r'^# ?', '', line)
            yaml_lines.append(clean_line)
        try:
            metadata = yaml.safe_load("\n".join(yaml_lines))
        except Exception as e:
            print(f"Error parsing YAML frontmatter in {script_path}: {e}")
        remaining_content = content[frontmatter_match.end():]
        
    # 2. Split cells by "# %%"
    # We want to match "# %%" at the beginning of a line
    cell_raw_blocks = re.split(r'^#\s*%%\s*', remaining_content, flags=re.MULTILINE)
    
    # Filter empty blocks (e.g. before the first # %%)
    cell_raw_blocks = [b for b in cell_raw_blocks if b.strip()]
    
    cells = []
    glob_env = {
        '__name__': '__main__',
        '__file__': script_path
    }
    
    # Track execution count for notebook cells
    exec_count = 1
    
    for block in cell_raw_blocks:
        lines = block.splitlines()
        if not lines:
            continue
            
        first_line = lines[0].strip()
        is_markdown = first_line.startswith('[markdown]')
        
        if is_markdown:
            # Markdown cell
            # The remaining lines are markdown content. Strip leading comment markers '# ' or '#'
            md_lines = []
            for line in lines[1:]:
                if line.startswith('# '):
                    md_lines.append(line[2:])
                elif line.startswith('#'):
                    md_lines.append(line[1:])
                else:
                    md_lines.append(line)
            cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': [l + '\n' for l in md_lines]
            })
        else:
            # Code cell
            # Source code is all lines of the block
            source_lines = [l + '\n' for l in lines]
            code_str = block
            
            # Execute cell code
            execution_result = execute_cell(code_str, glob_env)
            
            outputs = []
            
            # 1. Standard Output
            if execution_result['stdout']:
                outputs.append({
                    'output_type': 'stream',
                    'name': 'stdout',
                    'text': [execution_result['stdout']]
                })
                
            # 2. Standard Error (excluding tracebacks)
            if execution_result['stderr'] and not execution_result['traceback']:
                outputs.append({
                    'output_type': 'stream',
                    'name': 'stderr',
                    'text': [execution_result['stderr']]
                })
                
            # 3. Execution traceback errors
            if execution_result['traceback']:
                # Strip ANSI escape sequences from traceback for clean rendering in JSON
                clean_tb = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', execution_result['traceback'])
                outputs.append({
                    'output_type': 'error',
                    'ename': 'Exception',
                    'evalue': 'Error during cell execution',
                    'traceback': [clean_tb]
                })
                
            # 4. Matplotlib/Seaborn images
            for img in execution_result['images']:
                outputs.append({
                    'output_type': 'display_data',
                    'data': {
                        'image/png': img
                    },
                    'metadata': {}
                })
                
            # 5. Plotly interactive charts
            for plotly_json in execution_result['plotly_figs']:
                outputs.append({
                    'output_type': 'display_data',
                    'data': {
                        'application/vnd.plotly.v1+json': plotly_json
                    },
                    'metadata': {}
                })
                
            # 6. Last expression result value representation (only if no Plotly figures captured)
            if not execution_result['plotly_figs']:
                res = execution_result['result']
                if res is not None:
                    is_df = False
                    try:
                        import pandas as pd
                        if isinstance(res, pd.DataFrame):
                            is_df = True
                    except ImportError:
                        pass
                        
                    data = {}
                    if is_df:
                        data['text/html'] = [res.to_html(classes='dataframe ipynb-dataframe')]
                        data['text/plain'] = [repr(res)]
                    else:
                        data['text/plain'] = [repr(res)]
                        
                    outputs.append({
                        'output_type': 'execute_result',
                        'execution_count': exec_count,
                        'data': data,
                        'metadata': {}
                    })
                
            cells.append({
                'cell_type': 'code',
                'execution_count': exec_count,
                'metadata': {},
                'outputs': outputs,
                'source': source_lines
            })
            
            exec_count += 1
            
    # 3. Build full notebook structure
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }
    
    # Save notebook
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(script_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, cls=NumpyEncoder, indent=1)
        
    print(f"Successfully compiled {script_path} -> {output_path}")
    return metadata, filename


def rebuild_projects_ts(projects_list, output_path):
    ts_content = "export interface ProjectData {\n"
    ts_content += "  id: string;\n"
    ts_content += "  title: string;\n"
    ts_content += "  description: string;\n"
    ts_content += "  category: 'EDA' | 'ML' | 'AI';\n"
    ts_content += "  notebookUrl: string;\n"
    ts_content += "  dashboardUrl?: string;\n"
    ts_content += "  dashboardImage?: string;\n"
    ts_content += "  tags: string[];\n"
    ts_content += "  date: string;\n"
    ts_content += "  metrics: { [key: string]: string };\n"
    ts_content += "}\n\n"
    
    ts_content += "export const projectsData: ProjectData[] = [\n"
    
    entries = []
    for proj in projects_list:
        meta = proj['metadata']
        filename = proj['filename']
        
        title = meta.get('title', filename.replace('_', ' ').title())
        desc = meta.get('description', '')
        category = meta.get('category', 'EDA')
        tags = meta.get('tags', [])
        date = meta.get('date', '2026-06-25')
        metrics = meta.get('metrics', {})
        
        entry = "  {\n"
        entry += f"    id: '{filename}',\n"
        entry += f"    title: {json.dumps(title)},\n"
        entry += f"    description: {json.dumps(desc)},\n"
        entry += f"    category: '{category}',\n"
        entry += f"    notebookUrl: './notebooks/{filename}.ipynb',\n"
        
        if 'dashboardUrl' in meta:
            entry += f"    dashboardUrl: '{meta['dashboardUrl']}',\n"
        if 'dashboardImage' in meta:
            entry += f"    dashboardImage: '{meta['dashboardImage']}',\n"
            
        entry += f"    tags: {json.dumps(tags)},\n"
        entry += f"    date: '{date}',\n"
        entry += f"    metrics: {json.dumps(metrics, indent=6).replace('{', '{\n     ').replace('}', '\n    }')}\n"
        entry += "  }"
        entries.append(entry)
        
    ts_content += ",\n".join(entries)
    ts_content += "\n];\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ts_content)
    print(f"Successfully rebuilt projects registry at {output_path}")


def main():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(workspace_root, 'python_scripts')
    notebooks_dir = os.path.join(workspace_root, 'public', 'notebooks')
    projects_ts = os.path.join(workspace_root, 'src', 'data', 'projects.ts')
    
    if not os.path.exists(scripts_dir):
        print(f"Scripts directory '{scripts_dir}' does not exist.")
        return
        
    compiled_projects = []
    
    for file in sorted(os.listdir(scripts_dir)):
        if file.endswith('.py'):
            script_path = os.path.join(scripts_dir, file)
            try:
                metadata, filename = compile_script_to_notebook(script_path, notebooks_dir)
                compiled_projects.append({
                    'metadata': metadata,
                    'filename': filename
                })
            except Exception as e:
                print(f"Failed compiling {file}: {e}")
                traceback.print_exc()
                
    if compiled_projects:
        rebuild_projects_ts(compiled_projects, projects_ts)
        # Also copy notebooks to dist/notebooks so they are present in build directory
        dist_notebooks_dir = os.path.join(workspace_root, 'dist', 'notebooks')
        if os.path.exists(os.path.join(workspace_root, 'dist')):
            os.makedirs(dist_notebooks_dir, exist_ok=True)
            for proj in compiled_projects:
                filename = proj['filename']
                src_file = os.path.join(notebooks_dir, f"{filename}.ipynb")
                dst_file = os.path.join(dist_notebooks_dir, f"{filename}.ipynb")
                import shutil
                shutil.copy2(src_file, dst_file)
                print(f"Copied compiled notebook to build folder: {dst_file}")
    else:
        print("No script files were found/compiled.")

if __name__ == '__main__':
    main()
