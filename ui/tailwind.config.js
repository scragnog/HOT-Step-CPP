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
        },
        surface: { well: '#0d0d0f', chassis: '#111113', panel: '#19191c', raised: '#232326' },
        ink: { DEFAULT: '#eeeae5', muted: '#a6a19b', onTint: '#111113' },
        status: { ok: '#4ADE80', warn: '#60A5FA', error: '#FB7185' },

        /* Semantic colors */
        brand: {
          DEFAULT: '#ec4899',
          hover: '#db2777',
          light: '#f472b6',
        },
      },
      borderRadius: { ctl: '6px', panel: '10px' },
      boxShadow: {
        bevel: 'inset 0 1px 0 rgb(255 255 255 / 0.08), 0 2px 4px rgb(0 0 0 / 0.35)',
        press: 'inset 0 2px 4px rgb(0 0 0 / 0.55)',
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
