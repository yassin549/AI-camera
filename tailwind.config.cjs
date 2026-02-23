/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f1724",
        accent: "#7c5cff",
        accentSoft: "#9e8cff",
        glass: "rgba(255,255,255,0.08)"
      },
      boxShadow: {
        premium: "0 20px 60px rgba(0,0,0,0.35)",
        soft: "0 6px 20px rgba(0,0,0,0.25)"
      },
      backdropBlur: {
        xs: "2px"
      }
    }
  },
  plugins: []
};
