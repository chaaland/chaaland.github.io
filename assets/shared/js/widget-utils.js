/**
 * Widget Utilities - Shared functions for interactive SVG widgets
 *
 * Usage: All functions are exposed on window.WidgetUtils
 * Example: const rng = WidgetUtils.mulberry32(12345);
 */
(function() {
  'use strict';

  var utils = {
    // =========================================================================
    // Array Utilities
    // =========================================================================

    /**
     * Create an array of n evenly spaced values from start to end (inclusive)
     * @param {number} start - First value
     * @param {number} end - Last value
     * @param {number} n - Number of values
     * @returns {number[]} Array of evenly spaced values
     */
    linspace: function(start, end, n) {
      var step = (end - start) / (n - 1);
      return Array.from({length: n}, function(_, i) { return start + i * step; });
    },

    /**
     * Calculate ranks of array elements (1-based)
     * @param {number[]} arr - Input array
     * @returns {number[]} Array of ranks
     */
    ranks: function(arr) {
      var indexed = arr.map(function(v, i) { return { v: v, i: i }; });
      indexed.sort(function(a, b) { return a.v - b.v; });
      var r = new Array(arr.length);
      for (var rank = 1; rank <= indexed.length; rank++) {
        r[indexed[rank - 1].i] = rank;
      }
      return r;
    },

    // =========================================================================
    // Random Number Generation
    // =========================================================================

    /**
     * Mulberry32 - seeded 32-bit random number generator
     * @param {number} seed - Integer seed value
     * @returns {function} Function that returns random numbers in [0, 1)
     */
    mulberry32: function(seed) {
      return function() {
        var t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    },

    /**
     * Generate a standard normal random variable using Box-Muller transform
     * @param {function} rng - Random number generator returning values in [0, 1)
     * @returns {number} Standard normal random variable
     */
    normalRandom: function(rng) {
      var u1 = rng();
      var u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    },

    // =========================================================================
    // Statistical Functions
    // =========================================================================

    /**
     * Calculate the arithmetic mean of an array
     * @param {number[]} arr - Input array
     * @returns {number} Mean value
     */
    mean: function(arr) {
      var sum = 0;
      for (var i = 0; i < arr.length; i++) {
        sum += arr[i];
      }
      return sum / arr.length;
    },

    /**
     * Calculate sample variance
     * @param {number[]} arr - Input array
     * @returns {number} Sample variance
     */
    variance: function(arr) {
      var m = utils.mean(arr);
      var sumSq = 0;
      for (var i = 0; i < arr.length; i++) {
        var d = arr[i] - m;
        sumSq += d * d;
      }
      return sumSq / (arr.length - 1);
    },

    /**
     * Calculate Pearson correlation coefficient
     * @param {number[]} xs - First array
     * @param {number[]} ys - Second array
     * @returns {number} Correlation coefficient in [-1, 1]
     */
    pearsonCorr: function(xs, ys) {
      var n = xs.length;
      var xMean = utils.mean(xs);
      var yMean = utils.mean(ys);

      var covXY = 0, varX = 0, varY = 0;
      for (var i = 0; i < n; i++) {
        var dx = xs[i] - xMean;
        var dy = ys[i] - yMean;
        covXY += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
      }

      return covXY / Math.sqrt(varX * varY);
    },

    /**
     * Calculate Spearman rank correlation coefficient
     * @param {number[]} xs - First array
     * @param {number[]} ys - Second array
     * @returns {number} Rank correlation coefficient in [-1, 1]
     */
    spearmanCorr: function(xs, ys) {
      var n = xs.length;
      var rankX = utils.ranks(xs);
      var rankY = utils.ranks(ys);

      var sumSqDiff = 0;
      for (var i = 0; i < n; i++) {
        var d = rankX[i] - rankY[i];
        sumSqDiff += d * d;
      }

      return 1 - (6 * sumSqDiff) / (n * (n * n - 1));
    },

    // =========================================================================
    // SVG Coordinate Transforms
    // =========================================================================

    /**
     * Create coordinate transform functions for SVG plotting
     * @param {Object} config - Configuration object
     * @param {Object} config.margin - Margins {top, right, bottom, left}
     * @param {number} config.width - Total SVG width
     * @param {number} config.height - Total SVG height
     * @param {number} config.xMin - Minimum x data value
     * @param {number} config.xMax - Maximum x data value
     * @param {number} config.yMin - Minimum y data value
     * @param {number} config.yMax - Maximum y data value
     * @returns {Object} Object with toSvgX and toSvgY functions
     */
    createCoordTransforms: function(config) {
      var margin = config.margin;
      var plotW = config.width - margin.left - margin.right;
      var plotH = config.height - margin.top - margin.bottom;
      var xMin = config.xMin;
      var xMax = config.xMax;
      var yMin = config.yMin;
      var yMax = config.yMax;

      return {
        toSvgX: function(x) {
          return margin.left + (x - xMin) / (xMax - xMin) * plotW;
        },
        toSvgY: function(y) {
          return margin.top + (yMax - y) / (yMax - yMin) * plotH;
        },
        plotW: plotW,
        plotH: plotH
      };
    },

    /**
     * Build an SVG path string from arrays of x and y values
     * @param {number[]} xs - Array of x values
     * @param {number[]} ys - Array of y values
     * @param {function} toSvgX - X coordinate transform
     * @param {function} toSvgY - Y coordinate transform
     * @returns {string} SVG path d attribute
     */
    buildPath: function(xs, ys, toSvgX, toSvgY) {
      if (xs.length === 0) return '';
      var d = 'M ' + toSvgX(xs[0]) + ' ' + toSvgY(ys[0]);
      for (var i = 1; i < xs.length; i++) {
        d += ' L ' + toSvgX(xs[i]) + ' ' + toSvgY(ys[i]);
      }
      return d;
    },

    // =========================================================================
    // SVG Element Creation
    // =========================================================================

    /**
     * Create an SVG element with given attributes
     * @param {string} tag - SVG element tag name (e.g., 'line', 'circle', 'path')
     * @param {Object} attrs - Object of attribute name/value pairs
     * @returns {SVGElement} The created SVG element
     */
    createSvgElement: function(tag, attrs) {
      var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (var key in attrs) {
        if (attrs.hasOwnProperty(key)) {
          el.setAttribute(key, attrs[key]);
        }
      }
      return el;
    },

    /**
     * Draw a grid on an SVG group element
     * @param {SVGGElement} gridG - SVG group to draw into
     * @param {Object} config - Configuration matching createCoordTransforms
     * @param {Object} options - Grid options
     * @param {number} options.xStep - X grid line spacing
     * @param {number} options.yStep - Y grid line spacing
     * @param {string} [options.stroke='#21262d'] - Grid line color
     */
    drawGrid: function(gridG, config, options) {
      gridG.innerHTML = '';
      var xStep = options.xStep;
      var yStep = options.yStep;
      var stroke = options.stroke || '#21262d';
      var margin = config.margin;
      var coords = utils.createCoordTransforms(config);

      // Vertical lines
      for (var x = config.xMin; x <= config.xMax; x += xStep) {
        var line = utils.createSvgElement('line', {
          x1: coords.toSvgX(x),
          y1: margin.top,
          x2: coords.toSvgX(x),
          y2: config.height - margin.bottom,
          stroke: stroke,
          'stroke-width': 1
        });
        gridG.appendChild(line);
      }

      // Horizontal lines
      for (var y = config.yMin; y <= config.yMax; y += yStep) {
        var line = utils.createSvgElement('line', {
          x1: margin.left,
          y1: coords.toSvgY(y),
          x2: config.width - margin.right,
          y2: coords.toSvgY(y),
          stroke: stroke,
          'stroke-width': 1
        });
        gridG.appendChild(line);
      }
    }
  };

  // Expose utilities globally
  window.WidgetUtils = utils;
})();
