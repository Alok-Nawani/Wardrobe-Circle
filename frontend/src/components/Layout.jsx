/**
 * src/components/Layout.jsx
 * =========================
 * Premium top navigation layout.
 * Desktop: elegant header with subtle links.
 * Mobile: compact header with a clean sliding drawer.
 */

import { useState, useEffect } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { Search, Menu, X, User } from 'lucide-react';

const NAV_LINKS = [
  { to: '/wardrobe',      label: 'Wardrobe' },
  { to: '/outfit',        label: 'Style Studio' },
  { to: '/add',           label: 'Add Item' },
];

function NavItem({ to, label, onClick }) {
  return (
    <NavLink
      to={to}
      onClick={onClick}
      className={({ isActive }) =>
        `relative px-1 py-2 text-sm uppercase tracking-widest font-medium transition-colors duration-300 ease-editorial
         ${isActive ? 'text-brand-text' : 'text-brand-subtext hover:text-brand-text'}
        `
      }
    >
      {({ isActive }) => (
        <>
          {label}
          {isActive && (
            <span className="absolute bottom-0 left-0 w-full h-[1px] bg-brand-text animate-fade-in" />
          )}
        </>
      )}
    </NavLink>
  );
}

export default function Layout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu on navigation
  useEffect(() => {
    setMobileOpen(false);
  }, [location]);

  return (
    <div className="flex flex-col min-h-screen bg-brand-bg">
      {/* ── Top Navigation ── */}
      <header
        className={`fixed top-0 left-0 right-0 z-40 transition-all duration-500 ease-editorial border-b
          ${scrolled ? 'bg-brand-bg/90 backdrop-blur-md border-brand-border py-4' : 'bg-transparent border-transparent py-6'}
        `}
      >
        <div className="max-w-7xl mx-auto px-6 md:px-12 flex items-center justify-between">
          
          {/* Mobile Menu Button */}
          <button 
            className="md:hidden btn-icon -ml-2"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
          >
            <Menu size={22} strokeWidth={1.5} />
          </button>

          {/* Brand */}
          <Link to="/" className="flex flex-col items-center justify-center relative z-50">
            <span className="font-serif text-lg tracking-widest uppercase font-semibold leading-none">
              Wardrobe
            </span>
            <span className="font-sans text-[9px] tracking-[0.3em] uppercase text-brand-subtext mt-1 leading-none">
              Circle
            </span>
          </Link>

          {/* Desktop Links - Center */}
          <nav className="hidden md:flex absolute left-1/2 -translate-x-1/2 items-center gap-10">
            {NAV_LINKS.map(link => (
              <NavItem key={link.to} {...link} />
            ))}
          </nav>

          {/* Desktop Actions - Right */}
          <div className="flex items-center gap-3">
            <button className="btn-icon hidden md:flex" aria-label="Search">
              <Search size={18} strokeWidth={1.5} />
            </button>
            <button className="btn-icon" aria-label="Profile">
              <User size={18} strokeWidth={1.5} />
            </button>
          </div>

        </div>
      </header>

      {/* ── Mobile Menu Overlay ── */}
      <div 
        className={`fixed inset-0 z-50 bg-brand-bg/95 backdrop-blur-sm transition-all duration-500 ease-editorial flex flex-col justify-center items-center md:hidden
          ${mobileOpen ? 'opacity-100 visible' : 'opacity-0 invisible'}
        `}
      >
        <button 
          className="absolute top-6 left-6 btn-icon"
          onClick={() => setMobileOpen(false)}
        >
          <X size={24} strokeWidth={1.5} />
        </button>
        
        <nav className="flex flex-col items-center gap-8">
          <Link to="/" onClick={() => setMobileOpen(false)} className="text-2xl font-serif tracking-widest uppercase mb-4">Home</Link>
          {NAV_LINKS.map(link => (
            <Link 
              key={link.to} 
              to={link.to}
              onClick={() => setMobileOpen(false)}
              className="text-2xl font-serif tracking-widest uppercase text-brand-subtext hover:text-brand-text transition-colors"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>

      {/* ── Main content ── */}
      <main className="flex-1 flex flex-col w-full max-w-7xl mx-auto pt-28 md:pt-36 pb-20 px-6 md:px-12">
        {children}
      </main>
      
      {/* ── Footer ── */}
      <footer className="w-full max-w-7xl mx-auto px-6 md:px-12 py-10 border-t border-brand-border mt-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex flex-col items-center md:items-start gap-1">
            <span className="font-serif text-sm tracking-widest uppercase font-semibold">Wardrobe Circle</span>
            <span className="text-[10px] uppercase tracking-widest text-brand-subtext">AI-Powered Digital Wardrobe</span>
        </div>
        <div className="text-xs text-brand-subtext tracking-wider uppercase">
            © {new Date().getFullYear()} Wardrobe Circle
        </div>
      </footer>
    </div>
  );
}
