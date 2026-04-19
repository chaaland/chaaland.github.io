# CLAUDE.md - AI Agent Guide

## Project Overview

This is a **Jekyll-based technical blog** deployed to GitHub Pages at `chaaland.github.io`. The blog focuses on mathematics, machine learning, programming, and optimization algorithms.

## Directory Structure

```
├── _config.yml          # Jekyll configuration (theme, plugins, site settings)
├── _posts/              # Published blog posts (Markdown with frontmatter)
├── _drafts/             # Unpublished draft posts
├── _includes/           # Reusable HTML partials
│   ├── head/custom.html     # Google Fonts imports
│   ├── footer.html          # Custom footer with social links
│   └── scripts.html         # Analytics scripts
├── _sass/               # Custom SCSS stylesheets
│   ├── _custom-variables.scss  # Theme colors and typography
│   └── _custom-styles.scss     # Component styling
├── _site/               # Generated output (do not edit directly)
├── assets/              # Static assets organized by year
│   ├── shared/          # Shared resources (avatar, css, data)
│   ├── 2019/            # Assets for 2019 posts
│   ├── 2020/            # Assets for 2020 posts
│   ├── 2021/            # Assets for 2021 posts
│   ├── 2025/            # Assets for 2025 posts
│   ├── 2026/            # Assets for 2026 posts
│   └── drafts/          # Assets for unpublished draft posts
│       └── <topic>/     # Each post has its own asset directory
│           ├── generate_plots.py   # Python scripts for visualizations
│           ├── plots/              # Generated HTML/PNG plots
│           └── __marimo__/         # Marimo notebook sessions
├── categories/          # Category archive pages
├── tags/                # Tag archive pages
├── CONVENTIONS.md       # Authoring conventions (math, widgets, figures, footnotes)
├── index.html           # Homepage
├── about.md             # About page
├── Gemfile              # Ruby dependencies
└── .ruby-version        # Ruby 3.1.4
```

## Key Technologies

- **Theme**: [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) (dark skin)
- **Markdown**: Kramdown with GFM support
- **Math**: MathJax for LaTeX rendering
- **Visualizations**: Plotly (interactive HTML charts), inline SVG widgets
- **Notebooks**: Marimo for computational exploration
- **Fonts**: JetBrains Mono (headings/code), Source Serif Pro (body)

## Common Commands

```bash
# Install dependencies
bundle install

# Run development server
bundle exec jekyll serve

# Build site
bundle exec jekyll build
```

## Authoring Conventions

**Before writing or editing posts, read [CONVENTIONS.md](CONVENTIONS.md).** It is the authoritative reference for:

- Frontmatter fields and their required values
- Math notation choices (`\beta`, `\lvert...\rvert`, `$$...$$` for both inline and display)
- Footnote syntax (Kramdown `[^fn1]`)
- Figure numbering and `<figure>` / `<figcaption>` HTML patterns
- Interactive widget HTML/JS structure, SVG layout constants, and color palette
- Asset path conventions for published vs. draft posts

## Important Notes

- Ruby version 3.1.4 is required (see `.ruby-version`)
- Analytics: Google Universal Analytics (G-FL4HTN0JH2)
- Permalinks: `/:categories/:title/`
- Do NOT enter plan mode for informational questions or simple single-file edits — only use it for multi-step work spanning multiple files
