/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        /* Suno theme colors — matching hot-step-9000 */
        suno: {
          DEFAULT: '#09090b',
          sidebar: '#000000',
          panel: '#121214',
          card: '#18181b',
          hover: '#27272a',
          border: '#27272a',
        },


        /* UI kit (docs/dev/ui-design.md). The tint is CSS-variable channels,
           so bg-tint, bg-tint/20 and ring-tint re-colour under [data-tint]. */
        tint: {
          DEFAULT: 'rgb(var(--tint) / <alpha-value>)',
          hover: 'rgb(var(--tint-hover) / <alpha-value>)',
          glow: 'rgb(var(--tint-glow) / <alpha-value>)',
          deep: 'rgb(var(--tint-deep) / <alpha-value>)',
          ink: 'rgb(var(--tint-ink) / <alpha-value>)', // tint-coloured text and icons (AA on the light panel)
        },
        // Kit colours are theme variables (index.css): dark values on :root, light under html:not(.dark).
        surface: {
          well: 'rgb(var(--k-well) / <alpha-value>)',
          chassis: 'rgb(var(--k-chassis) / <alpha-value>)',
          panel: 'rgb(var(--k-panel) / <alpha-value>)',
          raised: 'rgb(var(--k-raised) / <alpha-value>)',
        },
        ink: { DEFAULT: 'rgb(var(--k-ink) / <alpha-value>)', muted: 'rgb(var(--k-muted) / <alpha-value>)', onTint: 'rgb(var(--k-on-tint) / <alpha-value>)' },
        status: { ok: 'rgb(var(--k-ok) / <alpha-value>)', warn: 'rgb(var(--k-warn) / <alpha-value>)', error: 'rgb(var(--k-error) / <alpha-value>)' },

        /* pink / rose read CSS variables (defaults in index.css, equal to Tailwind stock). */
        pink: {50: 'rgb(var(--pink-50) / <alpha-value>)', 100: 'rgb(var(--pink-100) / <alpha-value>)', 200: 'rgb(var(--pink-200) / <alpha-value>)', 300: 'rgb(var(--pink-300) / <alpha-value>)', 400: 'rgb(var(--pink-400) / <alpha-value>)', 500: 'rgb(var(--pink-500) / <alpha-value>)', 600: 'rgb(var(--pink-600) / <alpha-value>)', 700: 'rgb(var(--pink-700) / <alpha-value>)', 800: 'rgb(var(--pink-800) / <alpha-value>)', 900: 'rgb(var(--pink-900) / <alpha-value>)', 950: 'rgb(var(--pink-950) / <alpha-value>)'},
        rose: {50: 'rgb(var(--rose-50) / <alpha-value>)', 100: 'rgb(var(--rose-100) / <alpha-value>)', 200: 'rgb(var(--rose-200) / <alpha-value>)', 300: 'rgb(var(--rose-300) / <alpha-value>)', 400: 'rgb(var(--rose-400) / <alpha-value>)', 500: 'rgb(var(--rose-500) / <alpha-value>)', 600: 'rgb(var(--rose-600) / <alpha-value>)', 700: 'rgb(var(--rose-700) / <alpha-value>)', 800: 'rgb(var(--rose-800) / <alpha-value>)', 900: 'rgb(var(--rose-900) / <alpha-value>)', 950: 'rgb(var(--rose-950) / <alpha-value>)'},

        /* Semantic colors */
        brand: {
          DEFAULT: '#ec4899',
          hover: '#db2777',
          light: '#f472b6',
        },
      },
      borderRadius: { ctl: '6px', panel: '10px' },
      boxShadow: {
        bevel: 'inset 0 1px 0 rgb(var(--k-hi) / 0.08), 0 2px 4px rgb(var(--k-shade) / 0.35)',
        press: 'inset 0 2px 4px rgb(var(--k-shade) / 0.55)',
        glow: '0 0 14px rgb(var(--tint-glow) / 0.24)',
      },
      transitionDuration: { glow: '140ms', press: '70ms' },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      animation: {
        'gradient-x': 'gradient-x 15s ease infinite',
      },
      keyframes: {
        'gradient-x': {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center',
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center',
          },
        },
      },
    },
  },
  plugins: [
    // Portrait screens at desktop width (e.g. 1080x1920): stacked layout + mobile nav.
    ({ addVariant }) => addVariant('portrait-wide', '@media (orientation: portrait) and (min-width: 1024px)'),
    // Short viewports (laptops, phone landscape): the sidebar compacts so its
    // whole menu stays on screen instead of clipping below the fold.
    ({ addVariant }) => addVariant('short', '@media (max-height: 1040px)'),
  ],
};
