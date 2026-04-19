# Blog Post Conventions

Derived from the 5 most recent posts and the shared widget library. Use this as the reference when writing or editing posts.

---

## Frontmatter

```yaml
---
title: "Post Title"
categories:
  - Optimization          # one category, capitalized
date: YYYY-MM-DD HH:MM:SS +0000
mathjax: true
tags:
  - Tag One               # title-case, multi-word tags use spaces not hyphens
  - Tag Two
toc: true
classes: wide             # use for posts with interactive widgets or wide figures
excerpt: "One sentence description used in previews."
---
```

- Always include `mathjax: true` when the post contains math.
- Always include `toc: true`.
- Use `classes: wide` for posts with interactive widgets (removes the right sidebar and gives more horizontal space).

---

## Math Notation

**Inline math:** use `$$...$$` (double dollars, Kramdown convention — not single `$`).

**Display math:** also `$$...$$` but on its own line/block.

### Regression coefficients

| Symbol | Usage |
|--------|-------|
| `\beta` | Vector of slope parameters (multidimensional case) |
| `\beta_k` | k-th slope parameter |
| `\beta_0` | Intercept / bias |
| `\beta_1` | Slope in the 2D case |
| `\beta^\star` | Optimal parameter value |

- Always use `\beta` (not `b`, `w`, or `\theta`) for regression parameters.
- Use `\beta^T x^{(i)}` for the inner product in multi-dimensional regression.
- Superscript `(i)` for sample index: `x^{(i)}`, `y^{(i)}`.
- Subscript for feature/coordinate index: `x^{(i)}_k`, `\beta_k`.

### Vector spaces and norms

- Real vector space: `\mathbf{R}^d`
- Absolute value: `\lvert x \rvert` (not `|x|`)
- L2 norm: `\|...\|_2`
- L1 norm: `\|...\|_1`

### Optimization problem layout

```latex
$$
\underset{\beta, \beta_0}{\text{minimize}}\quad \frac{1}{N} \sum_{i=1}^N \lvert \beta^T x^{(i)} + \beta_0 - y^{(i)}\rvert
$$
```

- Use `\underset{...}{\text{minimize}}` (not `\min`).
- Use `\quad` after `\text{minimize}` before the objective.
- Always include `\frac{1}{N}` scaling for empirical loss functions.

### Sign / subgradient

- Sign function: `\mathbf{sign}(x)`
- Subgradient: `\partial \lvert x \rvert`

---

## Footnotes

Use Kramdown footnote syntax throughout. **Never** use HTML `<sup>` tags or parenthetical asides for footnotes.

**In-text reference:**
```markdown
This is the main claim.[^fn1]
```

**Definition (at the very end of the post, after all body text):**
```markdown
[^fn1]: Footnote text here, which can include math like $$\beta^\star = (A^TA)^{-1}A^Tb$$.
```

- Label convention: `[^fn1]`, `[^fn2]`, etc. for numbered footnotes; `[^keyword]` for named footnotes (e.g. `[^denormal]`, `[^twoscomp]`).
- Definitions go at the very end of the file, after `{% include widget-scripts.html %}` and `<script>` blocks when those are present.
- No blank line required between footnote definitions.

---

## Figure Numbering

Figures are numbered **globally and sequentially** throughout the post, starting at 1.

- Reference in body text: "Figure 1 shows...", "...as shown in Figure 3."
- `<figcaption>` text: `"Figure N: Description of what is shown."`
- Numbering is shared between static image figures and interactive widget figures.

---

## Static Image Figures

Use `<figure>` / `<figcaption>` HTML, not bare Markdown `![]()`:

```html
<figure>
  <a href="/assets/YEAR/TOPIC/images/filename.png">
    <img src="/assets/YEAR/TOPIC/images/filename.png" alt="Descriptive alt text">
  </a>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure N: Caption text.</figcaption>
</figure>
```

- Always wrap the image in an `<a>` link so users can click to enlarge.
- For full-width images without a click-to-enlarge: add `style="max-width: 100%;"` to the `<img>`.
- For side-by-side images: use `<figure class="half">` with two `<a><img></a>` pairs inside.
- Old CORDIC-style posts use `<figure class>` (bare class attribute) — new posts prefer the explicit patterns above.

**Figcaption style (always use exactly):**
```html
style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;"
```

---

## Interactive Widget Structure

### HTML skeleton

```html
<style>
#PREFIX-widget { max-width: 100%; }
.PREFIX-panels { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
.PREFIX-panel  { flex: 1; min-width: 280px; }
.PREFIX-panel-title { text-align: center; font-size: 0.85rem; color: #c9d1d9; margin-top: 6px; font-style: italic; }
@media (max-width: 600px) { .PREFIX-panels { flex-direction: column; align-items: center; } }
</style>

<div class="widget-container" id="PREFIX-widget">
  <div class="widget-controls">
    <label>
      β₁
      <input type="range" min="..." max="..." value="..." step="..." id="PREFIX-slope-slider">
      <span class="widget-readout" id="PREFIX-slope-readout">...</span>
    </label>
    <!-- more sliders -->
    <button type="button" class="widget-button" id="PREFIX-action-btn">Button label</button>
  </div>

  <!-- For multi-panel layouts: -->
  <div class="PREFIX-panels">
    <div class="PREFIX-panel">
      <svg class="widget-plot" id="PREFIX-left-svg" viewBox="0 0 320 280" preserveAspectRatio="xMidYMid meet">
        <!-- ... -->
      </svg>
      <div class="PREFIX-panel-title">Panel title</div>
    </div>
  </div>

  <!-- For single-panel layouts: -->
  <svg class="widget-plot" id="PREFIX-svg" viewBox="0 0 500 280" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="500" height="280" fill="#0d1117"></rect>
    <!-- named groups for each layer -->
    <g id="PREFIX-grid"></g>
    <g id="PREFIX-axes"></g>
    <path id="PREFIX-curve" fill="none" stroke="#58a6ff" stroke-width="2.5"></path>
  </svg>

  <div class="widget-info" id="PREFIX-info">Initial info text</div>
  <figcaption style="font-size: 0.9rem; color: #8b949e; margin-top: 12px;">Figure N: Caption.</figcaption>
</div>
```

### SVG layout constants

```js
var W = 320, H = 280;                           // two-panel SVG
// var W = 500, H = 280;                        // single-panel medium
// var W = 720, H = 300;                        // single-panel wide
var PD = { t: 25, r: 20, b: 40, l: 50 };       // padding: top/right/bottom/left
var PW = W - PD.l - PD.r;                       // plot width  (e.g. 250)
var PH = H - PD.t - PD.b;                       // plot height (e.g. 215)
```

### JavaScript IIFE pattern

Each widget gets its own IIFE to avoid polluting global scope:

```js
{% include widget-scripts.html %}   <!-- loads WidgetUtils — must appear before <script> -->
<script>
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';

  // Helper: create SVG element
  function el(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]);
    return e;
  }
  function txt(content, attrs) { var e = el('text', attrs); e.textContent = content; return e; }

  // Coordinate transforms
  function toSvgX(x) { return PD.l + (x - X0) / (X1 - X0) * PW; }
  function toSvgY(y) { return PD.t + (Y1 - y) / (Y1 - Y0) * PH; }

  function render() { /* update all SVG elements */ }

  render();
  document.getElementById('PREFIX-slider').addEventListener('input', render);
})();
</script>
```

- **ID prefix:** every element ID begins with the widget prefix (`lad-`, `ols-`, `gd1d-`).
- **`data-param` / `data-readout` attributes** (newer style in gradient-flow post): use `data-param="name"` on sliders and `data-readout="name"` on readout spans to enable generic event wiring.
- **Multiple widgets per post:** stack multiple IIFEs inside a single `<script>` block at the bottom of the file.

### Viridis colormap (canonical implementation)

```js
var VIR = [[68,1,84],[59,82,139],[33,144,140],[93,201,99],[253,231,37]];
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  var s = t * 4, lo = Math.floor(s), hi = Math.min(lo + 1, 4), f = s - lo;
  return 'rgb(' +
    Math.round(VIR[lo][0] + f*(VIR[hi][0]-VIR[lo][0])) + ',' +
    Math.round(VIR[lo][1] + f*(VIR[hi][1]-VIR[lo][1])) + ',' +
    Math.round(VIR[lo][2] + f*(VIR[hi][2]-VIR[lo][2])) + ')';
}
```

### Color palette

| Role | Hex |
|------|-----|
| Background | `#0d1117` |
| Primary line / accent | `#58a6ff` |
| Secondary line | `#3fb950` |
| Tertiary / warning line | `#f78166` |
| Axis / tick labels | `#8b949e` |
| Data points / text | `#c9d1d9` |
| Grid lines | `#21262d` |

### Axis drawing pattern

```js
function drawAxes(gId, xVals, xFmt, xToSvg, yVals, yFmt, yToSvg, xLabel, yLabel) {
  var g = document.getElementById(gId); g.innerHTML = '';
  // x-axis line
  g.appendChild(el('line', { x1: PD.l, y1: H-PD.b, x2: W-PD.r, y2: H-PD.b, stroke: '#8b949e', 'stroke-width': 1 }));
  // y-axis line
  g.appendChild(el('line', { x1: PD.l, y1: PD.t, x2: PD.l, y2: H-PD.b, stroke: '#8b949e', 'stroke-width': 1 }));
  // ticks + labels
  xVals.forEach(function(v) {
    var px = xToSvg(v);
    g.appendChild(el('line', { x1:px, y1:H-PD.b, x2:px, y2:H-PD.b+4, stroke:'#8b949e', 'stroke-width':1 }));
    g.appendChild(txt(xFmt(v), { x:px, y:H-PD.b+15, 'text-anchor':'middle', fill:'#8b949e', 'font-size':10, 'font-family':'monospace' }));
  });
  // axis labels
  g.appendChild(txt(xLabel, { x: PD.l+PW/2, y: H-5, 'text-anchor':'middle', fill:'#8b949e', 'font-size':11, 'font-family':'monospace' }));
  var mid = PD.t + PH/2;
  g.appendChild(txt(yLabel, { x:12, y:mid, 'text-anchor':'middle', fill:'#8b949e', 'font-size':11, 'font-family':'monospace', transform:'rotate(-90,12,'+mid+')' }));
}
```

### Current-position crosshair marker

```js
function updateMarker(gId, px, py) {
  var g = document.getElementById(gId); g.innerHTML = '';
  var r = 6;
  g.appendChild(el('line', { x1:px-r, y1:py,   x2:px+r, y2:py,   stroke:'white', 'stroke-width':1.5 }));
  g.appendChild(el('line', { x1:px,   y1:py-r, x2:px,   y2:py+r, stroke:'white', 'stroke-width':1.5 }));
  g.appendChild(el('circle', { cx:px, cy:py, r:r, fill:'none', stroke:'white', 'stroke-width':1.5 }));
}
```

### widget-info bar format

```
β₁ = 0.50 | β₀ = 10.00 | OLS loss = 12.34
```

Use Unicode `β₀` / `β₁` (not LaTeX) in widget info bars.

---

## Code Blocks in Posts

Use Liquid highlight tags (not fenced ``` blocks) for inline code examples:

```liquid
{% highlight python %}
def my_function(x):
    return x * 2
{% endhighlight %}
```

Wrap in `<details>/<summary>` when the snippet is long and optional reading:

```html
<details>
<summary>Click for code</summary>

{% highlight python %}
...
{% endhighlight %}

</details>
<br>
```

---

## Asset Paths

| Post state | Path pattern |
|------------|-------------|
| Published | `/assets/YEAR/TOPIC/images/filename.png` |
| Draft | `/assets/drafts/TOPIC/images/filename.png` |
| Shared JS | `/assets/shared/js/filename.js` |

When a draft is published, update all `/assets/drafts/TOPIC/` paths to `/assets/YEAR/TOPIC/`.
