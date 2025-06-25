from agno.agent import Agent
from agno.tools import tool
from agno.workflow import Workflow, RunResponse
import subprocess, os, re
from dotenv import load_dotenv

load_dotenv()

@tool
def md_to_pdf(file_path: str, style: str = "academic") -> str:
    """One-step Markdown to academic PDF conversion"""
    
    # Step 1: Convert to LaTeX
    tex_file = file_path.replace(".md", ".tex")
    cmd = ["pandoc", file_path, "-o", tex_file, "--standalone"]
    subprocess.run(cmd, check=True)
    
    # Step 2: Add academic formatting
    with open(tex_file, 'r') as f:
        content = f.read()
    
    formatting = r"""
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{setspace}
\onehalfspacing
\usepackage{titlesec}
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}
"""
    
    content = content.replace(r'\begin{document}', formatting + r'\begin{document}')
    
    # Fix Unicode issues
    fixes = {'≥': r'$\geq$', '≤': r'$\leq$', '≈': r'$\approx$', 'Δ': r'$\Delta$', 'α': r'$\alpha$', '∕': r'/', '⟦': r'[', '⟧': r']', '\u2003': ' '}
    for char, fix in fixes.items():
        content = content.replace(char, fix)
    
    with open(tex_file, 'w') as f:
        f.write(content)
    
    # Step 3: Generate PDF
    env = os.environ.copy()
    env["PATH"] = "/usr/local/texlive/2025/bin/universal-darwin:" + env.get("PATH", "")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False, env=env)
    
    pdf_file = tex_file.replace(".tex", ".pdf")
    return f"Generated: {pdf_file}" if os.path.exists(pdf_file) else "Failed"

@tool  
def update_metadata(tex_file: str, title: str = "", authors: str = "", date: str = "") -> str:
    """Update LaTeX metadata"""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    if title:
        content = re.sub(r'\\title\{[^}]*\}', f'\\\\title{{{title}}}', content)
    if authors:
        content = re.sub(r'\\author\{[^}]*\}', f'\\\\author{{{authors}}}', content)
    if date:
        content = re.sub(r'\\date\{[^}]*\}', f'\\\\date{{{date}}}', content)
    
    with open(tex_file, 'w') as f:
        f.write(content)
    return tex_file

class SimpleWorkflow(Workflow):
    """Minimal paper processing workflow"""
    description: str = "Simple MD to PDF conversion"
    
    def run(self, markdown_file: str, title: str = "", authors: str = "") -> RunResponse:
        try:
            result = md_to_pdf.entrypoint(markdown_file, "academic")
            if title or authors:
                tex_file = markdown_file.replace(".md", ".tex")
                update_metadata.entrypoint(tex_file, title, authors, "\\today")
                # Regenerate PDF
                md_to_pdf.entrypoint(markdown_file, "academic")
            return RunResponse(content=f"✅ {result}")
        except Exception as e:
            return RunResponse(content=f"❌ {str(e)}")

agent = Agent(
    name="SimpleAgent",
    introduction="Minimal paper processing",
    tools=[md_to_pdf, update_metadata]
)

if __name__ == "__main__":
    agent.cli_app()