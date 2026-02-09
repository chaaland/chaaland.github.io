/**
 * Golden Ratio Widget
 * Interactive demonstration of the golden ratio φ ≈ 1.618
 */
(function() {
  'use strict';

  // Golden ratio and threshold
  const PHI = 1.618033988749895;
  const TOLERANCE = 0.05; // 5% tolerance
  const PHI_MIN = PHI * (1 - TOLERANCE);
  const PHI_MAX = PHI * (1 + TOLERANCE);

  // Color scheme
  const COLORS = {
    defaultPoint: '#58a6ff',
    goldenPoint: '#ffd700',
    leftSegment: '#58a6ff',
    rightSegment: '#3fb950',
    goldenText: '#ffd700',
    defaultText: '#8b949e'
  };

  // DOM elements
  const widget = document.getElementById('golden-ratio-widget');
  if (!widget) return; // Widget not present

  const svg = document.getElementById('golden-ratio-svg');
  const point = document.getElementById('gr-point');
  const leftSegment = document.getElementById('gr-left-segment');
  const rightSegment = document.getElementById('gr-right-segment');
  const ratio1El = document.getElementById('gr-ratio1');
  const ratio2El = document.getElementById('gr-ratio2');
  const resetBtn = document.getElementById('golden-ratio-reset');
  const labelsG = document.getElementById('gr-labels');

  // SVG dimensions
  const BAR_X = 80;
  const BAR_Y = 50;
  const BAR_WIDTH = 440;
  const BAR_HEIGHT = 30;
  const POINT_RADIUS = 8;

  // State
  let isDragging = false;
  let currentPosition = BAR_WIDTH / 2; // Start at middle

  /**
   * Calculate ratios given a point position
   * @param {number} position - Position along bar [0, BAR_WIDTH]
   * @returns {Object} {wholeToLonger, longerToShorter, isGolden}
   */
  function calculateRatios(position) {
    const clampedPos = Math.max(0, Math.min(BAR_WIDTH, position));
    const leftLen = clampedPos;
    const rightLen = BAR_WIDTH - clampedPos;

    const longerLen = Math.max(leftLen, rightLen);
    const shorterLen = Math.min(leftLen, rightLen);

    // Avoid division by zero
    if (shorterLen === 0) {
      return {
        wholeToLonger: 1,
        longerToShorter: Infinity,
        isGolden: false
      };
    }

    const wholeToLonger = BAR_WIDTH / longerLen;
    const longerToShorter = longerLen / shorterLen;

    // Check if either ratio is close to golden ratio
    const isGolden = (wholeToLonger >= PHI_MIN && wholeToLonger <= PHI_MAX) ||
                     (longerToShorter >= PHI_MIN && longerToShorter <= PHI_MAX);

    return {
      wholeToLonger,
      longerToShorter,
      isGolden
    };
  }

  /**
   * Update the widget display
   */
  function update() {
    const ratios = calculateRatios(currentPosition);

    // Update point position and color
    const pointX = BAR_X + currentPosition;
    point.setAttribute('cx', pointX);
    point.setAttribute('fill', ratios.isGolden ? COLORS.goldenPoint : COLORS.defaultPoint);

    // Update segments
    leftSegment.setAttribute('width', currentPosition);
    rightSegment.setAttribute('x', BAR_X + currentPosition);
    rightSegment.setAttribute('width', BAR_WIDTH - currentPosition);

    // Format and display ratios
    const ratio1Text = ratios.wholeToLonger === Infinity ? '∞' : ratios.wholeToLonger.toFixed(3);
    const ratio2Text = ratios.longerToShorter === Infinity ? '∞' : ratios.longerToShorter.toFixed(3);

    ratio1El.textContent = ratio1Text;
    ratio2El.textContent = ratio2Text;

    // Highlight text if golden ratio found
    const textColor = ratios.isGolden ? COLORS.goldenText : COLORS.defaultText;
    ratio1El.style.color = textColor;
    ratio2El.style.color = textColor;
  }

  /**
   * Handle mouse down on draggable point
   */
  function handleMouseDown(e) {
    isDragging = true;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    e.preventDefault();
  }

  /**
   * Handle mouse move while dragging
   */
  function handleMouseMove(e) {
    if (!isDragging) return;

    const svgRect = svg.getBoundingClientRect();
    const mouseX = e.clientX - svgRect.left;

    // Convert to SVG coordinates (accounting for viewBox scaling)
    const viewBoxWidth = parseFloat(svg.getAttribute('viewBox').split(' ')[2]);
    const scale = viewBoxWidth / svgRect.width;
    const svgMouseX = mouseX * scale;

    // Clamp position within bar bounds
    currentPosition = Math.max(0, Math.min(BAR_WIDTH, svgMouseX - BAR_X));
    update();
  }

  /**
   * Handle mouse up to stop dragging
   */
  function handleMouseUp() {
    isDragging = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }

  /**
   * Reset widget to initial state (middle position)
   */
  function reset() {
    currentPosition = BAR_WIDTH / 2;
    update();
  }

  // Initialize event listeners
  point.addEventListener('mousedown', handleMouseDown);
  resetBtn.addEventListener('click', reset);

  // Initial update
  update();
})();
