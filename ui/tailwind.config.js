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

        /* Semantic colors */
        brand: {
          DEFAULT: '#ec4899',
          hover: '#db2777',
          light: '#f472b6',
        },
      },
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
