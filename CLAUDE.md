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
├── index.html           # Homepage
├── about.md             # About page
├── Gemfile              # Ruby dependencies
└── .ruby-version        # Ruby 3.1.4
```

## Key Technologies

- **Theme**: [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) (dark skin)
- **Markdown**: Kramdown with GFM support
- **Math**: MathJax for LaTeX rendering
- **Visualizations**: Plotly (interactive HTML charts)
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

## Writing Posts

### Frontmatter Template

```yaml
---
title: "Post Title"
date: YYYY-MM-DD
categories:
  - category-name
tags:
  - tag1
  - tag2
mathjax: true
toc: true
excerpt: "Brief description for previews"
---
```

## Asset Organization

Assets are organized by year, matching post publication dates:
- Published posts: `assets/<year>/<topic>/` (e.g., `assets/2025/cordic/`)
- Draft posts: `assets/drafts/<topic>/`
- Shared resources: `assets/shared/` (avatar, css, data)

Each post's asset directory typically contains:
- Python generation scripts (`generate_plots.py`)
- Generated plots (`plots/` subdirectory)
- Marimo notebooks (`__marimo__/`)
- Supporting data files

## Style Conventions

- **Color palette**: GitHub dark theme inspired (background #0d1117, accent #58a6ff)
- **Code blocks**: Use Rouge syntax highlighting
- **Math**: Inline with `$...$`, display with `$$...$$`
- **Permalinks**: `/:categories/:title/`

## Important Notes

- MathJax must be enabled via `mathjax: true` in frontmatter
- Posts require `toc: true` for table of contents
- Analytics: Google Universal Analytics (G-FL4HTN0JH2)
- Ruby version 3.1.4 is required