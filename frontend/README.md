# Wardrobe Circle — Frontend

React + Vite + Tailwind CSS frontend for the Wardrobe Circle AI platform.

## Pages

| Page | Route | Description |
|---|---|---|
| Dashboard | `/` | Overview, quick stats, navigation |
| My Wardrobe | `/wardrobe` | Browse and delete wardrobe items |
| Add Clothing | `/add` | Upload a clothing photo — AI analyses and saves it |
| Generate Outfit | `/outfit` | Recommend compatible outfit combinations |
| Find Missing Item | `/missing-item` | Fill-in-the-blank item recommendation |

## Stack

- **React 18** with React Router DOM v7
- **Vite** build tool
- **Tailwind CSS v3** for styling
- **Lucide React** for icons
- **Fontsource** (Inter, Playfair Display, Cormorant Garamond)

## Setup

```bash
npm install
cp .env.example .env       # adjust VITE_API_BASE_URL if needed
npm run dev                # http://localhost:5173
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | URL of the FastAPI backend |

Copy `.env.example` to `.env` and edit the value if your backend runs on a different host or port.

## Available Scripts

```bash
npm run dev      # Start development server (hot reload)
npm run build    # Production build → dist/
npm run preview  # Preview production build locally
npm run lint     # ESLint check
```

## Notes

- The frontend makes all API calls through `src/api/client.js`, which reads `VITE_API_BASE_URL`.
- The backend must be running before using any AI feature. See the root [README](../README.md) for backend setup.
- Wardrobe images are served directly by the FastAPI backend at `/wardrobe/images/{filename}`.
