<div align="center">

# SCTY Toolkit 🧰

### AI Research & Analysis

<p>
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/amadad/scty-toolkit" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/amadad/scty-toolkit" />
<img alt="GitHub Repo Size" src="https://img.shields.io/github/repo-size/amadad/scty-toolkit" />
<img alt="GitHub Stars" src="https://img.shields.io/github/stars/amadad/scty-toolkit" />
<img alt="GitHub Forks" src="https://img.shields.io/github/forks/amadad/scty-toolkit" />
<img alt="Github License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
<img alt="Twitter" src="https://img.shields.io/twitter/follow/amadad?style=social" />
</p>

</div>

-----

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#user-input">User Input</a> •
  <a href="#-custom-agents">Custom Agents</a> •
  <a href="#-task">Task</a> •
  <a href="#-tools">Tools</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-license">License</a>
</p>

-----

...

-----

## 📖 Overview

**SCTY Toolkit** is a curated collection of AI-powered tools and frameworks designed for research, analysis, and content creation. This repository contains multiple distinct projects that work together to provide a comprehensive toolkit for various AI and machine learning tasks.

## 🏗️ Project Structure

This repository is organized as a monorepo with shared dependencies and individual project folders:

```
scty-toolkit/
├── .venv/                    # Shared virtual environment
├── paper/                    # Academic paper processing tools
├── finetuning/              # ML model fine-tuning framework
├── scrape/                  # Web scraping and content extraction
├── sub2mark/               # Substack to Markdown converter
├── fire/                   # Firecrawl-based web scraper
├── rag-mm/                 # Multi-modal RAG system
├── video-to-blog/          # Video content to blog conversion
├── ui-gen/                 # UI generation tools
├── medi/                   # Medical AI tools
├── assistant/              # AI assistant frameworks
├── chains/                 # LLM chain implementations
└── ... (other projects)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/amadad/scty-toolkit.git
cd scty-toolkit

# Install dependencies
uv sync

# Run individual projects
python paper/main.py
python finetuning/main.py
python scrape/main.py
```

## 📊 Featured Projects

### 📄 Paper Processing (`paper/`)
Academic paper conversion and processing tools using Pandoc and ArXiv LaTeX cleaner.
- Convert Markdown to LaTeX/PDF
- ArXiv-ready document preparation
- Automated citation formatting

### 🔧 Fine-tuning Framework (`finetuning/`)
Comprehensive ML model fine-tuning and evaluation system.
- Model training pipelines
- Evaluation metrics and benchmarks
- Support for various model architectures

### 🕷️ Web Scraping (`scrape/`, `fire/`, `sub2mark/`)
Multiple web scraping solutions for different use cases.
- General web content extraction
- Substack newsletter conversion
- Enhanced crawling with Firecrawl integration

### 🧠 RAG System (`rag-mm/`)
Multi-modal Retrieval-Augmented Generation system.
- Document ingestion and processing
- Vector database integration
- Multi-modal content support

### 🎥 Content Creation (`video-to-blog/`)
AI-powered content transformation tools.
- Video to blog post conversion
- Automated content summarization
- Multi-format output support

## 🛠️ Tools & Technologies

- **AI/ML**: OpenAI API, Transformers, PyTorch
- **Web Scraping**: Selenium, BeautifulSoup, Firecrawl
- **Document Processing**: Pandoc, LaTeX, Markdown
- **Data Processing**: Pandas, NumPy, SQLAlchemy
- **UI/UX**: Streamlit, Gradio
- **Package Management**: UV, Python 3.11+

## 📈 Roadmap

- [ ] Enhanced multi-modal capabilities
- [ ] Improved model fine-tuning workflows
- [ ] Advanced RAG system features
- [ ] Better documentation and examples
- [ ] Performance optimizations
- [ ] Additional content format support


--

## Contributing

Please ensure to follow the project's code standards and submit pull requests for review. Contact [Ali Madad](mailto:ali@scty.org) for any questions or issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details