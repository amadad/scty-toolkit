from agno.agent import Agent  
from agno.tools import tool
from agno.workflow import Workflow, RunResponse
import subprocess
from dotenv import load_dotenv

load_dotenv()

@tool
def convert_md_to_tex(file_path: str, template: str = "default") -> str:
    out = file_path.replace(".md", ".tex")
    cmd = ["pandoc", file_path, "-o", out, "--standalone"]
    
    # Add template-specific options
    if template == "ieee":
        cmd.extend(["--template=ieee", "-V", "documentclass=IEEEtran"])
    elif template == "acm":
        cmd.extend(["-V", "documentclass=acmart"])
    elif template == "neurips":
        cmd.extend(["-V", "documentclass=neurips_2024"])
    elif template == "arxiv":
        cmd.extend(["-V", "documentclass=article", "-V", "geometry:margin=1in"])
    
    subprocess.run(cmd, check=True)
    return out

@tool
def add_latex_formatting(tex_file: str, style: str = "academic") -> str:
    """Add professional formatting to LaTeX file"""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Define style templates
    headers = {
        "academic": r"""
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{setspace}
\onehalfspacing
\usepackage{titlesec}
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}
""",
        "ieee": r"""
\usepackage[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
""",
        "arxiv": r"""
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{natbib}
\usepackage{url}
\linespread{1.1}
"""
    }
    
    # Insert formatting after \documentclass
    if style in headers:
        content = content.replace(
            r'\begin{document}', 
            headers[style] + r'\begin{document}'
        )
    
    # Write back
    formatted_file = tex_file.replace('.tex', '_formatted.tex')
    with open(formatted_file, 'w') as f:
        f.write(content)
    
    return formatted_file

@tool
def clean_latex(file_path: str) -> str:
    import os
    # arxiv-latex-cleaner expects a directory, so pass the directory containing the .tex file
    tex_dir = os.path.dirname(file_path)
    subprocess.run(["python", "-m", "arxiv_latex_cleaner", tex_dir], check=True)
    return f"{tex_dir}_arXiv"

@tool
def generate_pdf(tex_file: str) -> str:
    import os
    # Add LaTeX to PATH
    env = os.environ.copy()
    env["PATH"] = "/usr/local/texlive/2025/bin/universal-darwin:" + env.get("PATH", "")
    
    # Change to directory containing the tex file for proper PDF placement
    tex_dir = os.path.dirname(tex_file)
    tex_basename = os.path.basename(tex_file)
    pdf_file = tex_file.replace(".tex", ".pdf")
    
    try:
        if tex_dir:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False, env=env, cwd=tex_dir)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False, env=env, cwd=tex_dir)
        else:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False, env=env)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=False, env=env)
    except subprocess.CalledProcessError:
        pass
    
    # Check multiple possible PDF locations
    possible_pdfs = [
        pdf_file,  # Same directory as tex file
        os.path.basename(pdf_file),  # Current working directory
        os.path.join(tex_dir, os.path.basename(pdf_file)) if tex_dir else None
    ]
    
    for pdf_path in possible_pdfs:
        if pdf_path and os.path.exists(pdf_path):
            return f"PDF generated: {pdf_path}"
    
    raise Exception(f"PDF generation failed for {tex_file}")

@tool
def update_latex_field(tex_file: str, field: str, new_value: str) -> str:
    """Update specific LaTeX fields like author, date, title, etc."""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    import re
    
    if field.lower() == "author":
        # Replace author block - handle both \author{} and corrupted variants
        content = re.sub(r'\\author\{[^}]*\}', f'\\\\author{{{new_value}}}', content)
    elif field.lower() == "date":
        content = re.sub(r'\\date\{[^}]*\}', f'\\\\date{{{new_value}}}', content)
    elif field.lower() == "title":
        content = re.sub(r'\\title\{[^}]*\}', f'\\\\title{{{new_value}}}', content)
    elif field.lower() == "keywords":
        # Add or replace keywords before \begin{document}
        if r'\keywords{' in content:
            content = re.sub(r'\\keywords\{[^}]*\}', f'\\keywords{{{new_value}}}', content)
        else:
            content = content.replace(r'\begin{document}', f'\\keywords{{{new_value}}}\n\n\\begin{{document}}')
    
    # Write updated file
    updated_file = tex_file.replace('.tex', f'_{field}_updated.tex')
    with open(updated_file, 'w') as f:
        f.write(content)
    
    return updated_file

@tool
def insert_latex_section(tex_file: str, section_title: str, section_content: str, insert_after: str = "") -> str:
    """Insert a new section or subsection at specified location"""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Format the section
    if section_title.startswith("\\"):
        # Already formatted LaTeX command
        new_section = f"{section_title}\n{section_content}\n\n"
    else:
        # Plain title, add LaTeX formatting
        new_section = f"\\subsection{{{section_title}}}\n{section_content}\n\n"
    
    if insert_after:
        # Insert after specific content
        content = content.replace(insert_after, insert_after + "\n\n" + new_section)
    else:
        # Append before \end{document}
        content = content.replace(r'\end{document}', new_section + r'\end{document}')
    
    # Write updated file
    updated_file = tex_file.replace('.tex', '_with_section.tex')
    with open(updated_file, 'w') as f:
        f.write(content)
    
    return updated_file

@tool
def generate_gap_report(tex_file: str) -> str:
    """Analyze current LaTeX file against gap checklist"""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    gaps = []
    
    # Check front matter
    if r'\author{}' in content or 'Ali Madad' not in content:
        gaps.append("❌ Authors block needs completion")
    else:
        gaps.append("✅ Authors block fixed")
    
    if r'\date{v0.1' in content:
        gaps.append("❌ Date placeholder needs replacement")
    else:
        gaps.append("✅ Date updated")
    
    if r'\keywords{' not in content:
        gaps.append("❌ Keywords missing")
    else:
        gaps.append("✅ Keywords added")
    
    # Check for figures
    if r'⟦add-diagram⟧' in content:
        gaps.append("❌ System architecture figure placeholder")
    else:
        gaps.append("✅ System architecture figure referenced")
    
    # Check for tables
    if 'demographics table' not in content.lower():
        gaps.append("❌ Demographics table missing")
    
    # Check references count
    ref_count = content.count(r'\cite{') + content.count(r'\citep{')
    if ref_count < 20:
        gaps.append(f"❌ Need more references (current: ~{ref_count}, target: 35+)")
    else:
        gaps.append(f"✅ Sufficient references ({ref_count})")
    
    report = "📊 GAP ANALYSIS REPORT\n" + "="*50 + "\n" + "\n".join(gaps)
    
    # Write report to file
    report_file = tex_file.replace('.tex', '_gap_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    return report_file

@tool
def apply_flairs_formatting(tex_file: str) -> str:
    """Apply FLAIRS conference formatting requirements to LaTeX file"""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Fix Unicode characters that cause LaTeX errors
    unicode_fixes = {
        '≥': r'$\geq$',
        '≤': r'$\leq$',
        '≈': r'$\approx$',
        'Δ': r'$\Delta$',
        'α': r'$\alpha$',
        '∕': r'/',
        '⟦': r'[',
        '⟧': r']',
        '\u2003': ' ',  # em space
    }
    
    for unicode_char, latex_replacement in unicode_fixes.items():
        content = content.replace(unicode_char, latex_replacement)
    
    # Replace missing image placeholder
    content = content.replace(r'\includegraphics{⟦add-diagram⟧}', r'% System architecture diagram placeholder')
    content = content.replace(r'\includegraphics{[add-diagram]}', r'% System architecture diagram placeholder')
    
    # FLAIRS-specific preamble
    flairs_preamble = r"""
% FLAIRS Required Packages
\usepackage{flairs}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}

% PDFINFO for PDFLaTeX
\pdfinfo{
/Title (Your Paper Title Here)
/Author (Author Names Here)
/Keywords (Your keywords here)
}

% Section Numbers (optional - uncomment if needed)
% \setcounter{secnumdepth}{0}
"""
    
    # Replace document class and add FLAIRS requirements
    content = content.replace(
        r'\documentclass{article}',
        r'\documentclass[letterpaper]{article}'
    )
    
    # Insert FLAIRS preamble after \documentclass
    if r'\usepackage{flairs}' not in content:
        content = content.replace(
            r'\begin{document}',
            flairs_preamble + r'\begin{document}'
        )
    
    # Remove keywords command that's causing issues (FLAIRS handles this differently)
    content = content.replace(r'\keywords{caregiving, SMS, LLM, ecological momentary assessment}', '')
    
    # Fix calc package errors with brackets and math
    content = content.replace(r'[Add remaining citations to reach $\approx$ 35]', r'% Add remaining citations to reach approximately 35')
    content = content.replace(r'[Insert item tables]', r'% Insert item tables')
    content = content.replace(r'[PDF placeholder]', r'% PDF placeholder')
    content = content.replace(r'[Author]', r'Author et al.')
    
    # Write formatted file
    flairs_file = tex_file.replace('.tex', '_flairs.tex')
    with open(flairs_file, 'w') as f:
        f.write(content)
    
    return flairs_file

@tool
def check_flairs_compliance(tex_file: str) -> str:
    """Check LaTeX file against FLAIRS formatting requirements"""
    with open(tex_file, 'r') as f:
        content = f.read()
    
    compliance_issues = []
    compliance_ok = []
    
    # Check document class
    if '[letterpaper]' in content:
        compliance_ok.append("✅ Letter paper size specified")
    else:
        compliance_issues.append("❌ Document class must include [letterpaper]")
    
    # Check required packages
    required_packages = ['flairs', 'times', 'helvet', 'courier']
    for pkg in required_packages:
        if f'\\usepackage{{{pkg}}}' in content:
            compliance_ok.append(f"✅ {pkg} package included")
        else:
            compliance_issues.append(f"❌ Missing required package: {pkg}")
    
    # Check PDF dimensions
    if 'pdfpagewidth}{8.5in}' in content and 'pdfpageheight}{11in}' in content:
        compliance_ok.append("✅ PDF page dimensions set correctly")
    else:
        compliance_issues.append("❌ PDF page dimensions not set to US letter (8.5x11)")
    
    # Check for forbidden packages
    forbidden = ['hyperref', 'natbib', 'geometry', 'titlesec']
    for pkg in forbidden:
        if f'\\usepackage{{{pkg}}}' in content:
            compliance_issues.append(f"❌ Forbidden package detected: {pkg}")
    
    # Check for forbidden commands  
    forbidden_cmds = ['\\input', '\\vspace', '\\addtolength', '\\columnsep']
    for cmd in forbidden_cmds:
        if cmd in content:
            compliance_issues.append(f"❌ Forbidden command detected: {cmd}")
    
    # Check PDF metadata
    if '\\pdfinfo{' in content:
        compliance_ok.append("✅ PDF metadata included")
    else:
        compliance_issues.append("❌ Missing PDF metadata (\\pdfinfo)")
    
    # Generate report
    report = "📋 FLAIRS COMPLIANCE REPORT\n" + "="*50 + "\n\n"
    report += "COMPLIANT ITEMS:\n" + "\n".join(compliance_ok) + "\n\n"
    report += "ISSUES TO FIX:\n" + "\n".join(compliance_issues)
    
    # Write report
    report_file = tex_file.replace('.tex', '_flairs_compliance.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    return report_file

@tool
def create_flairs_template(title: str = "Your Paper Title", authors: str = "Author Names") -> str:
    """Create a FLAIRS-compliant LaTeX template from scratch"""
    
    template = f"""\\documentclass[letterpaper]{{article}}

% FLAIRS Required Packages
\\usepackage{{flairs}}
\\usepackage{{times}}
\\usepackage{{helvet}}
\\usepackage{{courier}}
\\setlength{{\\pdfpagewidth}}{{8.5in}}
\\setlength{{\\pdfpageheight}}{{11in}}

% PDFINFO for PDFLaTeX
\\pdfinfo{{
/Title ({title})
/Author ({authors})
/Keywords (Add your keywords here)
}}

% Section Numbers (uncomment if needed)
% \\setcounter{{secnumdepth}}{{0}}

% Title and Authors
\\title{{{title}}}
\\author{{{authors}}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
Your abstract goes here. Keep it under 250 words.
\\end{{abstract}}

\\section{{Introduction}}
Your introduction goes here.

\\section{{Related Work}}
Related work section.

\\section{{Methodology}}
Your methodology.

\\section{{Results}}
Your results.

\\section{{Discussion}}
Discussion of results.

\\section{{Conclusion}}
Your conclusions.

\\section{{Acknowledgments}}
Acknowledgments go here.

\\bibliographystyle{{flairs}}
\\bibliography{{references}}

\\end{{document}}"""

    template_file = "flairs_template.tex"
    with open(template_file, 'w') as f:
        f.write(template)
    
    return template_file

class PaperProcessingWorkflow(Workflow):
    """Agno workflow for complete paper processing pipeline"""
    
    description: str = "Complete paper processing pipeline for academic formats"
    
    def run(self, markdown_file: str, title: str = "", authors: str = "", keywords: str = "", target_format: str = "flairs") -> RunResponse:
        """Execute the complete paper processing workflow"""
        
        try:
            # Step 1: Convert Markdown to LaTeX
            tex_file = convert_md_to_tex.entrypoint(markdown_file, "default")
            
            # Step 2: Apply academic formatting
            formatted_file = add_latex_formatting.entrypoint(tex_file, "academic")
            
            # Step 3: Clean with ArXiv cleaner
            clean_dir = clean_latex.entrypoint(formatted_file)
            clean_file = f"{clean_dir}/01_formatted.tex"
            
            # Step 4: Apply target format (conditional logic)
            if target_format.lower() == "flairs":
                final_file = apply_flairs_formatting.entrypoint(clean_file)
            else:
                final_file = clean_file
            
            # Step 5: Update metadata if provided
            if title:
                final_file = update_latex_field.entrypoint(final_file, "title", title)
            if authors:
                final_file = update_latex_field.entrypoint(final_file, "author", authors)
            if keywords:
                final_file = update_latex_field.entrypoint(final_file, "keywords", keywords)
            
            # Step 6: Generate PDF
            pdf_file = generate_pdf.entrypoint(final_file)
            
            # Step 7: Generate compliance report
            compliance_report = check_flairs_compliance.entrypoint(final_file) if target_format.lower() == "flairs" else generate_gap_report.entrypoint(final_file)
            
            result_content = f"✅ AGNO WORKFLOW COMPLETE\n{'='*50}\n\n🎯 WORKFLOW RESULTS:\n📄 LaTeX File: {final_file}\n📖 PDF File: {pdf_file}\n📊 Compliance Report: {compliance_report}\n🔄 Workflow ID: {self.workflow_id if hasattr(self, 'workflow_id') else 'N/A'}"
            
            return RunResponse(content=result_content)
            
        except Exception as e:
            return RunResponse(content=f"❌ Workflow failed: {str(e)}")

@tool
def run_paper_workflow(markdown_file: str, title: str = "", authors: str = "", keywords: str = "", target_format: str = "flairs") -> str:
    """Execute the complete paper processing workflow using Agno workflows"""
    
    try:
        # Create and execute workflow
        workflow = PaperProcessingWorkflow()
        
        # Execute workflow
        result = workflow.run(
            markdown_file=markdown_file,
            title=title,
            authors=authors,
            keywords=keywords,
            target_format=target_format
        )
        
        # Create summary file
        summary_file = markdown_file.replace('.md', '_workflow_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(result.content)
            
        return summary_file
        
    except Exception as e:
        return f"❌ Workflow failed: {str(e)} - Falling back to manual pipeline"

@tool
def quick_paper_process(markdown_file: str, format_type: str = "academic") -> str:
    """Quick processing: Markdown → LaTeX → Academic Format → PDF"""
    
    pipeline_log = []
    
    try:
        # Step 1: Convert to LaTeX
        pipeline_log.append("🔄 Converting Markdown to LaTeX...")
        tex_file = convert_md_to_tex.entrypoint(markdown_file, "default")
        
        # Step 2: Apply formatting
        pipeline_log.append(f"🔄 Applying {format_type} formatting...")
        if format_type.lower() == "flairs":
            formatted_file = apply_flairs_formatting.entrypoint(tex_file)
        else:
            formatted_file = add_latex_formatting.entrypoint(tex_file, format_type)
        
        # Step 3: Generate PDF
        pipeline_log.append("🔄 Generating PDF...")
        pdf_result = generate_pdf.entrypoint(formatted_file)
        
        summary = "✅ QUICK PROCESSING COMPLETE\n" + "="*40 + "\n"
        summary += f"📄 LaTeX: {formatted_file}\n"
        summary += f"📖 PDF: {pdf_result}\n"
        
        summary_file = markdown_file.replace('.md', '_quick_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        return summary_file
        
    except Exception as e:
        return f"❌ Quick processing failed: {str(e)}"

agent = Agent(
    name="ArxivHelperAgent", 
    introduction="Complete paper processing using Agno workflows and manual pipelines - from Markdown to publication-ready PDF",
    tools=[convert_md_to_tex, add_latex_formatting, clean_latex, generate_pdf, update_latex_field, insert_latex_section, generate_gap_report, apply_flairs_formatting, check_flairs_compliance, create_flairs_template, run_paper_workflow, quick_paper_process],
)

def process_paper(md="drafts/01.md"):
    # Call the tool functions directly
    tex = convert_md_to_tex.entrypoint(md)
    clean_latex.entrypoint(tex)
    return generate_pdf.entrypoint(tex)

if __name__ == "__main__":
    import sys
    if "--process" in sys.argv:
        print(f"🎉 Complete! Final PDF: {process_paper()}")
    else:
        agent.cli_app()